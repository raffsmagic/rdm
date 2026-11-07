//! This crate's own network: blocking HTTP, where there is an operating system to open a
//! connection on.

use std::io::Read;
use std::time::Duration;

use crate::proxy;
use crate::transport::{HttpRequest, HttpResponse, Transport, TransportError};

/// HTTP to one server, with the connections it keeps open.
///
/// A proxy is taken from the environment when this is made, for the server it is made for.
#[derive(Clone)]
pub(crate) struct Http {
    agent: ureq::Agent,
}

impl Http {
    /// The transport for the server at a base URL. The environment is read here, once, for a proxy.
    pub(crate) fn to(base_url: &str) -> Self {
        let through = proxy::for_server(base_url, |variable| std::env::var(variable).ok());
        let agent = ureq::Agent::config_builder()
            // The proxy this server is reached through, or none: never the one ureq would
            // pick for every server alike.
            .proxy(through.and_then(|proxy| ureq::Proxy::new(&proxy).ok()))
            // A status is read here, with its body, and not turned into an error by the transport.
            .http_status_as_error(false)
            // A redirect is answered to the caller as a refusal, and the key goes nowhere new.
            .max_redirects(0)
            .max_redirects_will_error(false)
            .timeout_connect(Some(Duration::from_secs(10)))
            .user_agent(concat!("one-system-rs/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();
        Self { agent }
    }
}

impl Transport for Http {
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let mut response = match &request.body {
            Some(body) => {
                let mut call = self.agent.post(&request.url);
                for (name, value) in &request.headers {
                    call = call.header(name, value);
                }
                // The body is whole and in memory, so its length is sent and it is not chunked.
                call.config().timeout_global(Some(request.timeout)).build().send(&body[..])
            }
            None => {
                let mut call = self.agent.get(&request.url);
                for (name, value) in &request.headers {
                    call = call.header(name, value);
                }
                call.config().timeout_global(Some(request.timeout)).build().call()
            }
        }
        .map_err(failed)?;
        let status = response.status().as_u16();
        let headers = response.headers().iter();
        let headers = headers.filter_map(|(name, value)| Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned())));
        let headers: Vec<(String, String)> = headers.collect();
        // One byte more than is wanted is read for, to know that there was one.
        let most = request.limit as u64 + 1;
        let most = match closing(&response) {
            // An HTTP/1.0 server closes the connection after its reply. The transport would
            // keep it for the next request all the same, which then fails on a connection
            // that is closed or closing. It keeps a connection when a read finds the body
            // at its end: so the body is read to its last byte and no further, and the
            // connection goes with the response.
            Some(length) => length.min(most),
            None => most,
        };
        let mut body = Vec::new();
        let mut reader = response.body_mut().as_reader().take(most);
        reader.read_to_end(&mut body).map_err(|error| failed(ureq::Error::Io(error)))?;
        Ok(HttpResponse { status, headers, body })
    }
}

/// A request that got no response, as the client is told of it.
fn failed(error: ureq::Error) -> TransportError {
    let timed_out = match &error {
        ureq::Error::Timeout(_) => true,
        ureq::Error::Io(io) => matches!(io.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock),
        _ => false,
    };
    if timed_out {
        TransportError::TimedOut
    } else {
        TransportError::Failed(error.to_string())
    }
}

/// The length of a body whose connection the server closes after it and does not say so:
/// a reply in HTTP/1.0 that does not name `keep-alive`. None for any other reply. One with no
/// length ends where the connection does, and the transport does not keep that.
fn closing(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let headers = response.headers();
    let kept = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|token| token.trim().eq_ignore_ascii_case("keep-alive"));
    if response.version() != ureq::http::Version::HTTP_10 || kept {
        return None;
    }
    headers.get("content-length")?.to_str().ok()?.trim().parse().ok()
}
