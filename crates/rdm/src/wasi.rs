//! The host's network for a language model, as a module built for WASI 0.2 (`wasm32-wasip2`)
//! is given it: `wasi:http`. lm15 has no network of its own in a module, and this is the
//! transport its router is given there. The host opens the connection and does the TLS;
//! whatever runs the module has to grant it: `wasmtime run -S http`.

// An error here is lm15's own, returned to lm15 as its transport is to return one.
#![allow(clippy::result_large_err)]

use std::time::Duration;

use lm15::{ErrorMeta, Lm15Error, TransportRequest, TransportResponse};
use wasip2::clocks::monotonic_clock;
use wasip2::http::outgoing_handler;
use wasip2::http::types::{ErrorCode, Fields, Method, OutgoingBody, OutgoingRequest, RequestOptions, Scheme};
use wasip2::io::poll;
use wasip2::io::streams::StreamError;

/// HTTP through the host. It keeps nothing between requests: the connections are the host's.
pub(crate) struct WasiHttp;

/// As much as is written to, or read from, the host at a time.
const AT_A_TIME: usize = 4096;
/// How long a connection may take to open, and how long a reply may say nothing, unless the
/// request says otherwise: what lm15's own network allows.
const CONNECT: Duration = Duration::from_secs(10);
const IDLE: Duration = Duration::from_secs(600);

impl lm15::Transport for WasiHttp {
    fn send(&self, request: TransportRequest) -> lm15::transport::BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        // The host is waited on here, and what it answered is whole when this returns.
        Box::pin(std::future::ready(sent(&request)))
    }
}

fn failed(what: impl Into<String>) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(what))
}

fn late() -> Lm15Error {
    Lm15Error::TimeoutError(ErrorMeta::new("the host's HTTP did not answer in time"))
}

fn sent(request: &TransportRequest) -> Result<TransportResponse, Lm15Error> {
    let (scheme, rest) = request.url.split_once("://").ok_or_else(|| failed("the URL names no scheme"))?;
    let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    let scheme = match scheme {
        "https" => Scheme::Https,
        "http" => Scheme::Http,
        other => Scheme::Other(other.to_owned()),
    };
    // Encoding the parameters is the transport's to do.
    let mut path = if path.is_empty() { "/".to_owned() } else { path.to_owned() };
    for (name, value) in &request.params {
        path.push(if path.contains('?') { '&' } else { '?' });
        path.push_str(&format!("{}={}", encoded(name), encoded(value)));
    }
    let body = (request.body.is_some() || request.raw.is_some()).then(|| request.body_bytes());
    let mut headers: Vec<(String, Vec<u8>)> = request
        .headers
        .iter()
        .map(|(name, value)| (name.clone(), value.clone().into_bytes()))
        .collect();
    if let Some(body) = &body {
        headers.push(("content-length".to_owned(), body.len().to_string().into_bytes()));
    }
    let headers = Fields::from_list(&headers).map_err(|_| failed("the host does not let a module send one of the request's headers"))?;

    let outgoing = OutgoingRequest::new(headers);
    let method = match request.method.to_ascii_uppercase().as_str() {
        "GET" => Method::Get,
        "POST" => Method::Post,
        "PUT" => Method::Put,
        "PATCH" => Method::Patch,
        "DELETE" => Method::Delete,
        other => Method::Other(other.to_owned()),
    };
    let named = outgoing
        .set_method(&method)
        .and(outgoing.set_scheme(Some(&scheme)))
        .and(outgoing.set_authority(Some(authority)))
        .and(outgoing.set_path_with_query(Some(&path)));
    named.map_err(|()| failed("the URL is not one the host can send a request to"))?;
    let idle = request.read_timeout.unwrap_or(IDLE);
    let nanoseconds = |time: Duration| Some(u64::try_from(time.as_nanos()).unwrap_or(u64::MAX));
    let options = RequestOptions::new();
    // A host that keeps no such time is one that waits as long as it does: the reply is timed below either way.
    let _ = options.set_connect_timeout(nanoseconds(CONNECT));
    let _ = options.set_first_byte_timeout(nanoseconds(idle));
    let _ = options.set_between_bytes_timeout(nanoseconds(idle));

    let sending = outgoing.body().map_err(|()| failed("the host gave the request no body to write"))?;
    let answering = outgoing_handler::handle(outgoing, Some(options)).map_err(unanswered)?;
    if let Some(body) = &body {
        let stream = sending.write().map_err(|()| failed("the host gave the request no body to write"))?;
        for part in body.chunks(AT_A_TIME) {
            stream.blocking_write_and_flush(part).map_err(|error| unread(&error))?;
        }
    }
    OutgoingBody::finish(sending, None).map_err(unanswered)?;

    let answered = answering.subscribe();
    let response = loop {
        if let Some(answer) = answering.get() {
            break answer
                .map_err(|()| failed("the host answered once, and was asked twice"))?
                .map_err(unanswered)?;
        }
        let waited = monotonic_clock::subscribe_duration(nanoseconds(idle.saturating_add(CONNECT)).unwrap_or(u64::MAX));
        if poll::poll(&[&answered, &waited]).contains(&1) && answering.get().is_none() {
            return Err(late());
        }
    };
    let status = response.status();
    let headers = response.headers().entries().into_iter();
    let headers = headers.filter_map(|(name, value)| Some((name, String::from_utf8(value).ok()?)));
    let headers: Vec<(String, String)> = headers.collect();
    let incoming = response.consume().map_err(|()| failed("the host gave the reply no body to read"))?;
    let stream = incoming.stream().map_err(|()| failed("the host gave the reply no body to read"))?;
    let (mut body, arriving) = (Vec::new(), stream.subscribe());
    loop {
        // Each part of the reply is waited for as long as a reply may say nothing.
        let waited = monotonic_clock::subscribe_duration(nanoseconds(idle).unwrap_or(u64::MAX));
        if poll::poll(&[&arriving, &waited]).contains(&1) {
            return Err(late());
        }
        match stream.read(AT_A_TIME as u64) {
            Ok(part) => body.extend(part),
            Err(StreamError::Closed) => break,
            Err(error) => return Err(unread(&error)),
        }
    }
    Ok(TransportResponse::buffered(status, headers, body))
}

/// A name or a value as it is written in a query: every byte but a letter, a digit and `-._~` as `%XX`.
fn encoded(text: &str) -> String {
    let mut written = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => written.push(byte as char),
            other => written.push_str(&format!("%{other:02X}")),
        }
    }
    written
}

/// A request the host could not get answered.
fn unanswered(error: ErrorCode) -> Lm15Error {
    match error {
        ErrorCode::ConnectionTimeout
        | ErrorCode::ConnectionReadTimeout
        | ErrorCode::ConnectionWriteTimeout
        | ErrorCode::HttpResponseTimeout => late(),
        other => failed(format!("the host's HTTP failed: {other:?}")),
    }
}

/// A body that could not be written, or read, to its end.
fn unread(error: &StreamError) -> Lm15Error {
    failed(match error {
        StreamError::Closed => "the host closed the body before its end".to_owned(),
        StreamError::LastOperationFailed(error) => error.to_debug_string(),
    })
}
