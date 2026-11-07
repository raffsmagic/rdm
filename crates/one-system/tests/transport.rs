//! The client, through a transport that is a function: what a host is asked to carry, and what
//! the client makes of what it brings back. Nothing here opens a connection, so it runs
//! wherever the crate is built, a WebAssembly module included.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{api, body};
use one_system::{Backend, Client, Gateway, HttpRequest, HttpResponse, MAX_BODY_BYTES, Retry, Transport, TransportError};
use systemone::{DecisionModel, Error, Question, Request};

const KEY: &str = "sk-test-SECRET-0123456789";

fn request() -> Request {
    Request::new("I was charged twice.").question("refund", Question::noul("Is a refund asked for?"))
}

/// What a host answers a request with, given the request and how many came before it.
type Answers = dyn Fn(&HttpRequest, usize) -> Result<HttpResponse, TransportError> + Send + Sync;

/// A host that answers each request as a function says, and keeps what it was asked to carry.
#[derive(Clone)]
struct Host {
    carried: Arc<Mutex<Vec<HttpRequest>>>,
    answers: Arc<Answers>,
}

impl Host {
    fn answering(answers: impl Fn(&HttpRequest, usize) -> Result<HttpResponse, TransportError> + Send + Sync + 'static) -> Self {
        Self {
            carried: Arc::default(),
            answers: Arc::new(answers),
        }
    }

    /// A host in front of a server that answers every question, well formed.
    fn of_a_server() -> Self {
        Self::answering(|carried, _| {
            let asked: Request = serde_json::from_slice(carried.body.as_deref().unwrap_or_default()).unwrap();
            Ok(said(200, &body(&asked, "jev-1.13.0", 291)))
        })
    }

    fn carried(&self) -> Vec<HttpRequest> {
        self.carried.lock().unwrap().clone()
    }
}

impl Transport for Host {
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let before = {
            let mut carried = self.carried.lock().unwrap();
            carried.push(request.clone());
            carried.len() - 1
        };
        (self.answers)(&request, before)
    }
}

fn said(status: u16, body: &str) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![("Content-Type".into(), "application/json".into())],
        body: body.as_bytes().to_vec(),
    }
}

#[test]
fn a_host_is_asked_to_carry_the_request_as_the_client_would_send_it() {
    let host = Host::of_a_server();
    let client = Client::new("https://decisions.example/base/")
        .model("jev-latest")
        .key(KEY)
        .timeout(Duration::from_secs(7))
        .transport(host.clone());
    let response = client.decide(&request()).unwrap();
    assert_eq!(
        (response.model.as_str(), response.answers["refund"].noul),
        ("jev-1.13.0", Some(0.9))
    );

    let carried = host.carried();
    assert_eq!(carried.len(), 1);
    assert_eq!(
        (carried[0].method, carried[0].url.as_str()),
        ("POST", "https://decisions.example/base/v1/systemone")
    );
    let header = |name: &str| {
        carried[0]
            .headers
            .iter()
            .find(|(named, _)| named == name)
            .map(|(_, value)| value.as_str())
    };
    assert_eq!(header("Authorization"), Some(format!("Bearer {KEY}").as_str()));
    assert_eq!(
        (header("Content-Type"), header("Accept")),
        (Some("application/json"), Some("application/json"))
    );
    assert_eq!((carried[0].timeout, carried[0].limit), (Duration::from_secs(7), MAX_BODY_BYTES));
    // The client's model is named, since the request named none.
    let sent: Request = serde_json::from_slice(carried[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(sent, request().model("jev-latest"));
    // What a host is handed can be printed, and the key is not in it.
    let shown = format!("{:?}", carried[0]);
    assert!(shown.contains("Authorization") && !shown.contains("SECRET"), "{shown}");

    // The list of models is a `GET`, with no body and no type for one.
    let host = Host::answering(|_, _| Ok(said(200, r#"{"data": [{"id": "jev-latest"}]}"#)));
    let listed = Client::new("https://decisions.example").transport(host.clone()).models().unwrap();
    assert_eq!(listed["data"][0]["id"], "jev-latest");
    let carried = host.carried();
    assert_eq!(
        (carried[0].method, carried[0].url.as_str()),
        ("GET", "https://decisions.example/v1/models")
    );
    assert_eq!((carried[0].body.as_ref(), carried[0].headers.len()), (None, 1));
}

#[test]
fn what_a_host_brings_back_is_read_as_what_a_server_answered() {
    let ask = |answers: fn(&HttpRequest, usize) -> Result<HttpResponse, TransportError>| {
        let client = Client::new("https://decisions.example").key(KEY);
        client.transport(Host::answering(answers)).decide(&request())
    };
    // A refusal is the server's, whichever body it wrote.
    let refused = ask(|_, _| {
        Ok(said(
            401,
            r#"{"detail": {"error_type": "unauthorized", "message": "No such key."}}"#,
        ))
    });
    assert_eq!(api(&refused.unwrap_err()), (401, "unauthorized".into(), "No such key.".into()));
    // A redirect a host did not follow is a refusal, and one it must not follow.
    let moved = ask(|_, _| Ok(said(302, "")));
    assert_eq!(moved.unwrap_err().status(), Some(302));
    // Answers that are not for the questions asked are not evidence.
    let other = ask(|_, _| Ok(said(200, r#"{"model": "m", "answers": {}}"#)));
    assert!(matches!(other.unwrap_err(), Error::Evidence(_)));
    // A body longer than was wanted is not read, however much of it the host brought back.
    let long = ask(|carried, _| Ok(said(200, &" ".repeat(carried.limit + 1))));
    assert!(matches!(long.unwrap_err(), Error::Evidence(message) if message.contains("8 MiB")));
    let long = ask(|carried, _| Ok(said(503, &"x".repeat(carried.limit + 1))));
    assert_eq!(
        api(&long.unwrap_err()),
        (503, String::new(), "The server answered with status 503.".into())
    );
    // A host that could not carry it, or ran out of time, is a transport error, without the key.
    let late = ask(|_, _| Err(TransportError::TimedOut));
    assert!(matches!(late.unwrap_err(), Error::Transport(message) if message.contains("did not answer within")));
    let lost = ask(|carried, _| Err(TransportError::Failed(format!("fetch failed for {:?}", carried.headers))));
    let lost = lost.unwrap_err();
    assert!(
        matches!(&lost, Error::Transport(message) if message.contains("fetch failed")),
        "{lost:?}"
    );
    assert!(!format!("{lost} {lost:?}").contains("SECRET"), "{lost:?}");
}

#[test]
fn a_failure_that_may_pass_is_carried_again_when_retries_are_on() {
    let twice_busy = |_: &HttpRequest, before: usize| match before {
        0 => Err(TransportError::Failed("connection reset".into())),
        1 => Ok(HttpResponse {
            headers: vec![("Retry-After-Ms".into(), "1".into())],
            ..said(429, r#"{"error": {"message": "Slow down.", "type": "rate_limited"}}"#)
        }),
        _ => Ok(said(200, &body(&request(), "jev-1.13.0", 291))),
    };
    let quick = Retry {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(2),
        ..Retry::default()
    };
    let host = Host::answering(twice_busy);
    let client = Client::new("https://decisions.example").transport(host.clone());
    // Nothing is asked twice unless the client is told to.
    assert!(matches!(client.decide(&request()).unwrap_err(), Error::Transport(_)));
    assert_eq!(host.carried().len(), 1);
    let host = Host::answering(twice_busy);
    let patient = Client::new("https://decisions.example").retry(quick).transport(host.clone());
    assert_eq!(patient.decide(&request()).unwrap().model, "jev-1.13.0");
    assert_eq!(host.carried().len(), 3);
}

#[test]
fn a_client_on_a_hosts_transport_is_a_backend_like_any_other() {
    let host = Host::of_a_server();
    let hosted = Client::new("https://decisions.example").model("jev-latest").transport(host.clone());
    let gateway = Gateway::builder("demo").backend(Backend::new("hosted", hosted)).build().unwrap();
    assert_eq!(gateway.decide(&request()).unwrap().model, "jev-1.13.0");
    assert_eq!(host.carried().len(), 1);
    // And a clone of a client asks through the same host.
    let client = Client::new("https://decisions.example").transport(host.clone());
    client.clone().decide(&request()).unwrap();
    assert_eq!(host.carried().len(), 2);
}

/// Where the client is given no transport: its own network where there is one, and, in a
/// WebAssembly module with no `wasi:http` under it, a refusal that says whose the network is.
#[test]
#[cfg(all(target_family = "wasm", not(all(target_os = "wasi", target_env = "p2"))))]
fn a_module_with_no_network_says_so_and_is_still_there() {
    let error = Client::new("https://decisions.example").key(KEY).decide(&request()).unwrap_err();
    assert!(
        matches!(&error, Error::Transport(message) if message.contains("the host carries the request")),
        "{error:?}"
    );
    assert!(!error.to_string().contains("SECRET"));
}
