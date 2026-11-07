//! What carries a request to a server, and brings back what it answered.
//!
//! A [`Client`](crate::Client) says what is asked and reads what comes back, and a transport is
//! the network between the two. Where there is an operating system it is this crate's own:
//! blocking HTTP, with TLS by rustls. A WebAssembly module can open no connection of its own,
//! so there the host carries the request: through WASI's `wasi:http` where the module is built
//! for it (`wasm32-wasip2`), and anywhere else as a [`Transport`] the client is given.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// One HTTP request, whole: the body is in memory, and its length is known.
///
/// Its `Debug` output names the headers and shows no value of one: a key is among them.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// `GET` or `POST`.
    pub method: &'static str,
    /// Where it goes, in full: `https://api.typesafe.ai/v1/systemone`.
    pub url: String,
    /// The headers, by name and value. `Authorization` is one of them when a key is sent.
    pub headers: Vec<(String, String)>,
    /// The body of a `POST`: JSON, as its `Content-Type` says.
    pub body: Option<Vec<u8>>,
    /// How long the whole of it may take, from connecting to the last byte of the answer.
    pub timeout: Duration,
    /// The most of the answer's body that is wanted, in bytes. One that is longer is cut
    /// here, or anywhere after: the client then knows it was too long, and reads none of it.
    pub limit: usize,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let named: Vec<&str> = self.headers.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &named)
            .field("body", &self.body.as_ref().map(Vec::len))
            .field("timeout", &self.timeout)
            .field("limit", &self.limit)
            .finish()
    }
}

/// What a server answered: its status, whatever that is, with its headers and its body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HttpResponse {
    /// The status. A refusal is a response, and not a failure of the transport.
    pub status: u16,
    /// The headers, by name and value. Names are read whatever their case.
    pub headers: Vec<(String, String)>,
    /// The body, or as much of it as was asked for.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// The first header of a name, whatever case it was sent in.
    pub fn header(&self, name: &str) -> Option<&str> {
        let found = self.headers.iter().find(|(named, _)| named.eq_ignore_ascii_case(name));
        found.map(|(_, value)| value.as_str())
    }
}

/// A request that got no response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransportError {
    /// Nothing came back in the time the request was given.
    TimedOut,
    /// Anything else, in words. A client takes its key out of them before they go further.
    Failed(String),
}

/// Its words as they came, before a client takes its key out of them: for the host that sent it.
impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::TimedOut => f.write_str("no response in the time given"),
            TransportError::Failed(words) => f.write_str(words),
        }
    }
}

impl std::error::Error for TransportError {}

/// What carries one request and waits for its answer.
///
/// A transport does not follow a redirect: it answers with the redirect as it came, since the
/// request carries a key and would take it wherever it was sent on to. A status is never an
/// error here. It may be asked by several threads at once.
///
/// A function is one, which is how a host gives its own: here one that answers by itself.
///
/// ```
/// use one_system::{Client, HttpRequest, HttpResponse, TransportError};
/// use systemone::{Answer, DecisionModel, Question, Request, Response};
///
/// let host = |request: HttpRequest| -> Result<HttpResponse, TransportError> {
///     assert_eq!((request.method, request.url.as_str()), ("POST", "https://decisions.example/v1/systemone"));
///     let answers = [("refund".to_owned(), Answer::noul(0.9))].into_iter().collect();
///     let answered = Response { model: "jev-1".into(), answers, usage: Default::default() };
///     Ok(HttpResponse { status: 200, headers: vec![], body: serde_json::to_vec(&answered).unwrap() })
/// };
/// let client = Client::new("https://decisions.example").transport(host);
/// let request = Request::new("I was charged twice.").question("refund", Question::noul("Is a refund asked for?"));
/// assert_eq!(client.decide(&request)?.model, "jev-1");
/// # Ok::<(), systemone::Error>(())
/// ```
pub trait Transport: Send + Sync {
    /// Send one request, and wait for the whole of what is answered.
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError>;
}

impl<F: Fn(HttpRequest) -> Result<HttpResponse, TransportError> + Send + Sync> Transport for F {
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        self(request)
    }
}

/// The transport a client has until it is given another: this crate's own HTTP.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn of_this_build(base_url: &str) -> Arc<dyn Transport> {
    Arc::new(crate::http::Http::to(base_url))
}

/// The transport a client has until it is given another: the host's, through `wasi:http`.
#[cfg(all(target_os = "wasi", target_env = "p2"))]
pub(crate) fn of_this_build(_base_url: &str) -> Arc<dyn Transport> {
    Arc::new(crate::wasi::WasiHttp)
}

/// The transport a client has until it is given another: none. Every request says so.
#[cfg(all(target_family = "wasm", not(all(target_os = "wasi", target_env = "p2"))))]
pub(crate) fn of_this_build(_base_url: &str) -> Arc<dyn Transport> {
    Arc::new(|_: HttpRequest| -> Result<HttpResponse, TransportError> {
        Err(TransportError::Failed(
            "this WebAssembly module has no network of its own: the host carries the request, \
             as a transport the client is given (`Client::transport`)"
                .to_owned(),
        ))
    })
}
