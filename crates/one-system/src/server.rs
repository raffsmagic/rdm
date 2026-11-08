//! A gateway, served over HTTP.
//!
//! Three endpoints behind one bearer key: `POST /v1/systemone`, `GET /v1/models` and
//! `GET /v1/capabilities`. Every response is JSON, and every error is
//! `{"detail":[{"loc":["body"],"msg":"...","type":"..."}]}` with a fixed message that holds
//! nothing of the request, of a backend's reply, or of what went wrong underneath.

use std::io::Read;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::json;
use sha2::{Digest, Sha256};
use systemone::Error;

use crate::error::{ConfigError, api};
use crate::gateway::{Gateway, Picked, Route, Stage};
use crate::registry::valid_key;
use crate::schema::{MAX_BODY_BYTES, MISMATCHED, complete, invalid_upstream, mismatched, parse_request};

/// The largest body a request may say it has and still be heard out: 64 MiB, eight times what
/// is accepted. Between the two, a request is told it is too large. Over this, it is not
/// answered at all (see `Shared::serve`).
const MAX_DECLARED_BYTES: usize = 64 << 20;

/// What goes back for one request.
struct Reply {
    status: u16,
    body: Vec<u8>,
    /// A header the status calls for: `WWW-Authenticate`, `Allow`, or how the cache answered.
    header: Option<(&'static str, &'static str)>,
    route: Option<Route>,
}

impl Reply {
    fn ok(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            body,
            header: None,
            route: None,
        }
    }

    fn error(status: u16, code: &str, message: &str) -> Self {
        let mut body = json!({"detail": [{"loc": ["body"], "msg": message, "type": code}]})
            .to_string()
            .into_bytes();
        body.push(b'\n');
        Self {
            status,
            body,
            header: None,
            route: None,
        }
    }

    fn header(mut self, name: &'static str, value: &'static str) -> Self {
        self.header = Some((name, value));
        self
    }
}

/// What a caller is told of an error, given where it arose.
///
/// The gateway's own errors already are what a caller is told. A backend's error is the
/// backend's business: its words may hold what it was sent, so only its status is passed on,
/// and of those not the ones that are about the gateway's own request to the backend (a 400,
/// a 401 or a 403 there is no fault of the caller's), which become a 502. A backend that
/// answered, but not with a response, is `invalid_upstream_response`, and one that answered
/// other questions than it was asked is `mismatched_answers`, whichever of the two found it
/// out: the gateway, or a backend that is a client and checks what its server sent.
fn public(error: &Error, stage: Stage) -> Reply {
    let told = |error: &Error| match error {
        Error::Api { status, code, message } => Reply::error(*status, code, message),
        _ => Reply::error(500, "internal_error", "Request could not be completed"),
    };
    if stage == Stage::Gateway {
        return told(error);
    }
    match error {
        Error::Transport(_) | Error::Api { status: 407, .. } => Reply::error(502, "upstream_unavailable", "Upstream request failed"),
        Error::Api { status, .. } => {
            let status = match status {
                400 | 401 | 403 => 502,
                400..=599 => *status,
                _ => 502,
            };
            Reply::error(status, "upstream_rejected", "Upstream could not complete the request")
        }
        Error::Evidence(found) if found.starts_with(MISMATCHED) => told(&mismatched()),
        Error::Evidence(_) => told(&invalid_upstream()),
        Error::Invalid(_) => Reply::error(500, "upstream_configuration", "Upstream configuration is invalid"),
        // A backend that replays what was kept has nothing for this request.
        Error::Unrecorded(_) => Reply::error(502, "upstream_unavailable", "Upstream request failed"),
    }
}

/// A path with its percent escapes decoded, or nothing for a path with an escape that is
/// none, or that decodes to what is not text. Such a path is no endpoint.
fn decoded(path: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(path.len());
    let mut rest = path.as_bytes();
    while let Some((&byte, after)) = rest.split_first() {
        if byte != b'%' {
            bytes.push(byte);
            rest = after;
            continue;
        }
        let digits = after.get(..2).and_then(|digits| std::str::from_utf8(digits).ok())?;
        // `from_str_radix` would take a sign.
        if !digits.bytes().all(|digit| digit.is_ascii_hexdigit()) {
            return None;
        }
        bytes.push(u8::from_str_radix(digits, 16).ok()?);
        rest = &after[2..];
    }
    String::from_utf8(bytes).ok()
}

/// Compare two digests without stopping at the first byte that differs.
fn same(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter().zip(right).fold(0u8, |difference, (a, b)| difference | (a ^ b)) == 0
}

struct Shared {
    gateway: Gateway,
    /// SHA-256 of the whole `Authorization` value a caller must send.
    authorization: [u8; 32],
    log: bool,
}

impl Shared {
    /// The checks, in the order callers rely on: the path, then the key, then the method,
    /// then the body.
    fn reply(&self, request: &mut tiny_http::Request) -> Reply {
        // The path as it reads once its escapes are decoded, which is how the Go gateway
        // matches it: `/v1/%6dodels` is `/v1/models`.
        let path = decoded(request.url().split(['?', '#']).next().unwrap_or_default());
        let path = path.as_deref().unwrap_or_default();
        let method = match path {
            "/v1/models" | "/v1/capabilities" => "GET",
            "/v1/systemone" => "POST",
            _ => return Reply::error(404, "not_found", "Endpoint not found"),
        };
        let given = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("Authorization"))
            .map(|header| header.value.as_str().to_string())
            .unwrap_or_default();
        let given: [u8; 32] = Sha256::digest(given.as_bytes()).into();
        if !same(&given, &self.authorization) {
            return Reply::error(401, "unauthorized", "A valid bearer API key is required").header("WWW-Authenticate", "Bearer");
        }
        if request.method().as_str() != method {
            return Reply::error(405, "method_not_allowed", "Method not allowed").header("Allow", method);
        }
        match path {
            "/v1/models" => return Reply::ok(self.gateway.models().to_string().into_bytes()),
            "/v1/capabilities" => return Reply::ok(self.gateway.capabilities().to_string().into_bytes()),
            _ => {}
        }

        // One byte more than is allowed is read, to know that there was one. A body that
        // stops short of the length it declared was not read either.
        let declared = request.body_length();
        let mut body = Vec::new();
        let read = request.as_reader().take(MAX_BODY_BYTES as u64 + 1).read_to_end(&mut body);
        let short = declared.is_some_and(|declared| body.len() < declared.min(MAX_BODY_BYTES + 1));
        if read.is_err() || short || body.len() > MAX_BODY_BYTES {
            return Reply::error(422, "invalid_body", "Request body is unreadable or exceeds 8 MiB");
        }
        let asked = match parse_request(&body) {
            Ok(asked) => asked,
            Err(error) => return public(&error, Stage::Gateway),
        };
        drop(body);
        // In the process, a request that names no model lets the gateway choose. Over HTTP
        // there is no such default: the gateway is asked for by its name.
        if asked.model.is_empty() {
            return public(&api(422, "unsupported_model", "Unknown model; see GET /v1/models"), Stage::Gateway);
        }
        let routed = self.gateway.route(&asked);
        let mut reply = match &routed.result {
            // What goes out is a whole response by the official schema, or it does not go out.
            Ok(response) => {
                let mut response = response.clone();
                match complete(&asked.questions, &mut response).and_then(|()| {
                    serde_json::to_vec(&response)
                        .map_err(|_| api(502, "invalid_response", "Upstream response cannot be returned as SystemOneResponse"))
                }) {
                    Ok(body) => Reply::ok(body),
                    Err(error) => public(&error, Stage::Gateway),
                }
            }
            Err(error) => public(error, routed.route.failed.unwrap_or(Stage::Gateway)),
        };
        if routed.route.cached {
            reply = reply.header("X-One-System-Cache", "hit");
        }
        reply.route = Some(routed.route);
        reply
    }

    fn serve(&self, mut request: tiny_http::Request) {
        // tiny_http 0.12 reads whatever is left of a body when a request is dropped, into a
        // buffer it allocates as large as what is left. A request that declares terabytes and
        // sends none would have it ask for terabytes, and a refused allocation ends the
        // process. Such a request cannot be answered without being dropped, so it is neither:
        // it is left where it is, its connection with it, and the server goes on.
        if request.body_length().is_some_and(|declared| declared > MAX_DECLARED_BYTES) {
            std::mem::forget(request);
            return;
        }
        let started = Instant::now();
        let reply = self.reply(&mut request);
        if self.log
            && let Some(route) = &reply.route
        {
            log_route(route, reply.status, started.elapsed());
        }
        let header = |name: &str, value: &str| tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes());
        let mut response = tiny_http::Response::from_data(reply.body).with_status_code(reply.status);
        let fixed = [("Content-Type", "application/json"), ("Cache-Control", "no-store")];
        for (name, value) in fixed.into_iter().chain(reply.header) {
            if let Ok(header) = header(name, value) {
                response.add_header(header);
            }
        }
        // A caller that has gone is not an error of the server's.
        let _ = request.respond(response);
    }
}

/// One line for one routed request: where it went and why, and nothing of what it said.
fn log_route(route: &Route, status: u16, took: Duration) {
    let (how, confidence) = match &route.picked {
        None if route.cached => ("cache", None),
        None => ("none", None),
        Some(Picked::Direct) => ("direct", None),
        Some(Picked::Only) => ("only", None),
        Some(Picked::Selected { confidence }) => ("selected", Some(*confidence)),
        Some(Picked::Withheld { confidence, .. }) => ("withheld", Some(*confidence)),
        Some(Picked::Escalated { value, .. }) => ("escalated", Some(*value)),
        Some(Picked::Stayed) => ("stayed", None),
        Some(Picked::Fallback { .. }) => ("selector_failed", None),
    };
    let line = json!({
        "level": if status == 200 { "INFO" } else { "WARN" },
        "msg": "route",
        "backend": route.backend,
        "picked": how,
        "confidence": confidence,
        "eligible": route.eligible,
        "latency_ms": took.as_millis() as u64,
        "status": status,
    });
    eprintln!("{line}");
}

/// A gateway listening on an address.
///
/// It answers `POST /v1/systemone` as the gateway does, `GET /v1/models` with the gateway's
/// name and its backends' ids, and `GET /v1/capabilities` with what each backend declared.
/// All three want `Authorization: Bearer <key>`. A request is checked in this order, and the
/// first check it fails is the one it hears about: the path (404), the key (401), the method
/// (405), the size of the body (422 `invalid_body` over 8 MiB), its shape (422
/// `schema_validation`), the model (422 `unsupported_model`), then what the gateway finds.
/// A path is matched as it reads once decoded (`/v1/%6dodels` is `/v1/models`), and the
/// gateway is asked for by its name: over HTTP an empty `model` is an unknown one.
///
/// A 200 is always a whole response by TypeSafe's published schema. What that schema requires
/// and a backend left out (a choice's `choice` and `confidence`; a score's `score`,
/// `confidence` and `legend`) is worked out from the probabilities and the question, and what
/// a backend said is sent as it said it. A backend whose answer cannot be made whole is a 502
/// `invalid_upstream_response`, and a number that JSON cannot write a 502 `invalid_response`.
///
/// Each request is answered on a thread of its own, so one that waits on a slow backend, or
/// on a caller slow to send, holds up no other. It is a small server all the same, for this
/// machine or a network that is trusted: it sets no time limit on a caller and no bound on
/// how many it serves at once. A burst of callers that connect in the same instant and then
/// send nothing can, rarely, hold up the one after them, since tiny_http's pool may queue a
/// connection behind a thread they hold. Put something sturdier in front of it before the
/// open internet.
///
/// ```no_run
/// use one_system::{Gateway, Server};
///
/// let gateway = Gateway::load("backends.json")?;
/// let server = Server::bind("127.0.0.1:8090", gateway, "the-key-callers-present")?;
/// eprintln!("listening on {}", server.url());
/// server.run();
/// # Ok::<(), one_system::ConfigError>(())
/// ```
pub struct Server {
    http: tiny_http::Server,
    gateway: Gateway,
    authorization: [u8; 32],
    address: SocketAddr,
    log: bool,
}

/// Without its key.
impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("address", &self.address)
            .field("log", &self.log)
            .finish_non_exhaustive()
    }
}

impl Server {
    /// Listen on an address (port 0 for any free port) and serve a gateway to callers who
    /// present `key`. The key is a nonempty bearer token with no whitespace in it.
    pub fn bind(address: impl ToSocketAddrs, gateway: Gateway, key: &str) -> Result<Self, ConfigError> {
        if !valid_key(key) {
            return Err(ConfigError::new(
                "ONE_SYSTEM_API_KEY must contain a nonempty bearer key without whitespace",
            ));
        }
        let unavailable = || ConfigError::new("could not listen on ONE_SYSTEM_ADDR");
        let http = tiny_http::Server::http(address).map_err(|_| unavailable())?;
        let address = http.server_addr().to_ip().ok_or_else(unavailable)?;
        Ok(Self {
            http,
            gateway,
            authorization: Sha256::digest(format!("Bearer {key}").as_bytes()).into(),
            address,
            log: false,
        })
    }

    /// Write one line to standard error for each request routed: the backend, how it was
    /// picked, the status and how long it took. Nothing of the request is written.
    pub fn log_routes(mut self, on: bool) -> Self {
        self.log = on;
        self
    }

    /// The address it is listening on.
    pub fn addr(&self) -> SocketAddr {
        self.address
    }

    /// The base URL a [`Client`](crate::Client) reaches it at.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Serve until the process ends.
    pub fn run(self) {
        self.spawn().wait();
    }

    /// Serve from threads of its own, and return at once with a handle that stops it.
    pub fn spawn(self) -> Running {
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared {
            gateway: self.gateway,
            authorization: self.authorization,
            log: self.log,
        });
        let (http, stopping) = (self.http, stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                // Wake now and then to see whether to stop.
                let Ok(Some(request)) = http.recv_timeout(Duration::from_millis(50)) else {
                    continue;
                };
                let shared = shared.clone();
                // A request no thread can be started for is dropped, which answers it 500.
                let _ = std::thread::Builder::new()
                    .name("one-system request".into())
                    .spawn(move || shared.serve(request));
            }
        });
        Running {
            address: self.address,
            stop,
            thread: Some(thread),
        }
    }
}

/// A server that is serving. Dropping it stops it.
pub struct Running {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Running {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Running").field("address", &self.address).finish_non_exhaustive()
    }
}

impl Running {
    /// The address it is listening on.
    pub fn addr(&self) -> SocketAddr {
        self.address
    }

    /// The base URL a [`Client`](crate::Client) reaches it at.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Stop taking requests. Those being answered are answered.
    pub fn stop(mut self) {
        self.halt();
    }

    fn wait(mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.halt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn told(error: Error, stage: Stage) -> (u16, String) {
        let reply = public(&error, stage);
        let body: serde_json::Value = serde_json::from_slice(&reply.body).unwrap();
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        assert_eq!(reply.body.last(), Some(&b'\n'));
        (reply.status, body["detail"][0]["type"].as_str().unwrap().to_string())
    }

    #[test]
    fn what_a_caller_is_told_of_a_backend() {
        let api = |status: u16| Error::Api {
            status,
            code: "theirs".into(),
            message: "PRIVATE".into(),
        };
        for (status, public_status) in [
            (400, 502),
            (401, 502),
            (403, 502),
            (422, 422),
            (429, 429),
            (503, 503),
            (529, 529),
            (302, 502),
            (700, 502),
        ] {
            assert_eq!(told(api(status), Stage::Backend), (public_status, "upstream_rejected".to_string()));
        }
        assert_eq!(told(api(407), Stage::Selector), (502, "upstream_unavailable".to_string()));
        assert_eq!(
            told(Error::Transport("PRIVATE".into()), Stage::Backend),
            (502, "upstream_unavailable".to_string())
        );
        assert_eq!(
            told(Error::Evidence("PRIVATE".into()), Stage::Backend),
            (502, "invalid_upstream_response".to_string())
        );
        // A backend that is a client found its server answering other questions.
        let found = crate::schema::Fault::Mismatched("PRIVATE".into()).evidence();
        assert_eq!(told(found.clone(), Stage::Selector), (502, "mismatched_answers".to_string()));
        assert!(!String::from_utf8_lossy(&public(&found, Stage::Backend).body).contains("PRIVATE"));
        let found = crate::schema::Fault::Malformed("PRIVATE".into()).evidence();
        assert_eq!(told(found, Stage::Backend), (502, "invalid_upstream_response".to_string()));
        assert_eq!(
            told(Error::Invalid("PRIVATE".into()), Stage::Backend),
            (500, "upstream_configuration".to_string())
        );
        let reply = public(&api(422), Stage::Backend);
        assert!(!String::from_utf8_lossy(&reply.body).contains("PRIVATE"));
        // The gateway's own errors are already public.
        assert_eq!(told(api(422), Stage::Gateway), (422, "theirs".to_string()));
    }

    #[test]
    fn paths_decode() {
        assert_eq!(decoded("/v1/models").as_deref(), Some("/v1/models"));
        assert_eq!(decoded("/v1/%6dodels").as_deref(), Some("/v1/models"));
        assert_eq!(decoded("/v1/%6Dodels").as_deref(), Some("/v1/models"));
        assert_eq!(decoded("%2Fv1%2fmodels").as_deref(), Some("/v1/models"));
        assert_eq!(decoded("/%E2%98%83").as_deref(), Some("/\u{2603}"));
        assert_eq!(decoded("").as_deref(), Some(""));
        for bad in ["/%", "/%6", "/%zz", "/%+6", "/%-1", "/%FF", "/a%2"] {
            assert_eq!(decoded(bad), None, "{bad}");
        }
    }

    #[test]
    fn digests_compare() {
        let a: [u8; 32] = Sha256::digest(b"Bearer a").into();
        let b: [u8; 32] = Sha256::digest(b"Bearer b").into();
        assert!(same(&a, &a) && !same(&a, &b));
    }
}
