//! The host's network, as a module built for WASI 0.2 (`wasm32-wasip2`) is given it:
//! `wasi:http`. The host opens the connection and does the TLS; nothing of either is in the
//! module. Whatever runs the module has to grant it: `wasmtime run -S http`.

use std::time::Duration;

use wasip2::clocks::monotonic_clock;
use wasip2::http::outgoing_handler;
use wasip2::http::types::{ErrorCode, Fields, Method, OutgoingBody, OutgoingRequest, RequestOptions, Scheme};
use wasip2::io::poll;
use wasip2::io::streams::StreamError;

use crate::transport::{HttpRequest, HttpResponse, Transport, TransportError};

/// HTTP through the host. It keeps nothing between requests: the connections are the host's.
pub(crate) struct WasiHttp;

/// As much as is written to, or read from, the host at a time.
const AT_A_TIME: usize = 4096;

impl Transport for WasiHttp {
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let failed = |what: &str| TransportError::Failed(what.to_owned());
        let (scheme, rest) = request.url.split_once("://").ok_or_else(|| failed("the URL names no scheme"))?;
        let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
        let scheme = match scheme {
            "https" => Scheme::Https,
            "http" => Scheme::Http,
            other => Scheme::Other(other.to_owned()),
        };
        let mut headers: Vec<(String, Vec<u8>)> = request
            .headers
            .iter()
            .map(|(name, value)| (name.clone(), value.clone().into_bytes()))
            .collect();
        // The body is whole and in memory, so its length is sent and it is not chunked.
        if let Some(body) = &request.body {
            headers.push(("Content-Length".to_owned(), body.len().to_string().into_bytes()));
        }
        headers.push(("User-Agent".to_owned(), concat!("one-system-rs/", env!("CARGO_PKG_VERSION")).into()));
        let headers =
            Fields::from_list(&headers).map_err(|_| failed("the host does not let a module send one of the request's headers"))?;

        let outgoing = OutgoingRequest::new(headers);
        let method = if request.body.is_some() { Method::Post } else { Method::Get };
        let named = outgoing
            .set_method(&method)
            .and(outgoing.set_scheme(Some(&scheme)))
            .and(outgoing.set_authority(Some(authority)))
            .and(outgoing.set_path_with_query(Some(if path.is_empty() { "/" } else { path })));
        named.map_err(|()| failed("the URL is not one the host can send a request to"))?;
        let nanoseconds = |time: Duration| Some(u64::try_from(time.as_nanos()).unwrap_or(u64::MAX));
        let options = RequestOptions::new();
        // A host that keeps no such time is one that waits as long as it does: the whole
        // exchange is timed below either way.
        let _ = options.set_connect_timeout(nanoseconds(request.timeout.min(Duration::from_secs(10))));
        let _ = options.set_first_byte_timeout(nanoseconds(request.timeout));
        let _ = options.set_between_bytes_timeout(nanoseconds(request.timeout));

        let sending = outgoing.body().map_err(|()| failed("the host gave the request no body to write"))?;
        let answering = outgoing_handler::handle(outgoing, Some(options)).map_err(unanswered)?;
        if let Some(body) = &request.body {
            let stream = sending.write().map_err(|()| failed("the host gave the request no body to write"))?;
            for part in body.chunks(AT_A_TIME) {
                stream.blocking_write_and_flush(part).map_err(|error| unread(&error))?;
            }
        }
        OutgoingBody::finish(sending, None).map_err(unanswered)?;

        // The whole of it is timed here: the host is waited on for the answer, and for the
        // time the request was given, and whichever is first ends the wait.
        let late = monotonic_clock::subscribe_duration(nanoseconds(request.timeout).unwrap_or(u64::MAX));
        let answered = answering.subscribe();
        let response = loop {
            if let Some(answer) = answering.get() {
                break answer
                    .map_err(|()| failed("the host answered once, and was asked twice"))?
                    .map_err(unanswered)?;
            }
            if poll::poll(&[&answered, &late]).contains(&1) && answering.get().is_none() {
                return Err(TransportError::TimedOut);
            }
        };
        let status = response.status();
        let headers = response.headers().entries().into_iter();
        let headers = headers.filter_map(|(name, value)| Some((name, String::from_utf8(value).ok()?)));
        let headers: Vec<(String, String)> = headers.collect();
        let incoming = response
            .consume()
            .map_err(|()| failed("the host gave the answer no body to read"))?;
        let stream = incoming.stream().map_err(|()| failed("the host gave the answer no body to read"))?;
        let (mut body, arriving) = (Vec::new(), stream.subscribe());
        // One byte more than is wanted is read for, to know that there was one.
        while body.len() <= request.limit {
            if poll::poll(&[&arriving, &late]).contains(&1) {
                return Err(TransportError::TimedOut);
            }
            match stream.read(AT_A_TIME as u64) {
                Ok(part) => body.extend(part),
                Err(StreamError::Closed) => break,
                Err(error) => return Err(unread(&error)),
            }
        }
        body.truncate(request.limit.saturating_add(1));
        Ok(HttpResponse { status, headers, body })
    }
}

/// A request the host could not get answered.
fn unanswered(error: ErrorCode) -> TransportError {
    match error {
        ErrorCode::ConnectionTimeout
        | ErrorCode::ConnectionReadTimeout
        | ErrorCode::ConnectionWriteTimeout
        | ErrorCode::HttpResponseTimeout => TransportError::TimedOut,
        other => TransportError::Failed(format!("{other:?}")),
    }
}

/// A body that could not be written, or read, to its end.
fn unread(error: &StreamError) -> TransportError {
    TransportError::Failed(match error {
        StreamError::Closed => "the host closed the body before its end".to_owned(),
        StreamError::LastOperationFailed(error) => error.to_debug_string(),
    })
}
