//! What the tests share: answers for any question, a server that is a function, and HTTP by hand.
#![allow(dead_code, unused_imports)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use indexmap::IndexMap;
use systemone::{Answer, Error, Kind, Question, Request, Response, Scripted, Usage};

/// A well-formed answer to one question: a likely yes, the first option, or the lowest level.
pub fn answer(question: &Question) -> Answer {
    let labels = question.labels();
    match question.kind {
        Kind::Noul => Answer::noul(0.9),
        Kind::Choice => Answer::chose(&labels[0], labels.iter().map(String::as_str)),
        Kind::Score => Answer::score(labels.iter().enumerate().map(|(level, _)| if level == 0 { 1.0 } else { 0.0 })),
    }
}

/// A well-formed answer to every question of a request.
pub fn answers(request: &Request) -> IndexMap<String, Answer> {
    request
        .questions
        .iter()
        .map(|(name, question)| (name.clone(), answer(question)))
        .collect()
}

/// A model that answers anything, well formed, under a name, reporting `tokens` each time.
pub fn model(name: &str, tokens: u64) -> Arc<Scripted> {
    Arc::new(Scripted::new(answers).named(name).tokens(tokens))
}

/// A model that always fails the same way.
pub fn failing(name: &str, error: Error) -> Arc<Scripted> {
    Arc::new(Scripted::fallible(move |_| Err(error.clone())).named(name))
}

/// A response body for a request, as a server would write it.
pub fn body(request: &Request, model: &str, tokens: u64) -> String {
    let response = Response {
        model: model.into(),
        answers: answers(request),
        usage: Usage {
            input_tokens: tokens,
            output_tokens: 0,
        },
    };
    serde_json::to_string(&response).unwrap()
}

/// One request a test server received.
#[derive(Clone, Debug)]
pub struct Seen {
    pub method: String,
    pub url: String,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Seen {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn request(&self) -> Request {
        serde_json::from_slice(&self.body).expect("the body is a request")
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("the body is JSON")
    }
}

/// What a test server answers with.
#[derive(Clone, Debug)]
pub struct Reply {
    pub status: u16,
    pub body: String,
    pub headers: Vec<(String, String)>,
    /// How long to wait before answering.
    pub after: Duration,
}

impl Reply {
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            headers: Vec::new(),
            after: Duration::ZERO,
        }
    }

    pub fn ok(body: impl Into<String>) -> Self {
        Self::new(200, body)
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn after(mut self, wait: Duration) -> Self {
        self.after = wait;
        self
    }
}

/// A System One server that is a function, on 127.0.0.1 and a port of its own. The function
/// is given each request and how many came before it. A WebAssembly module can be no server.
#[cfg(not(target_family = "wasm"))]
pub struct Upstream {
    address: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

#[cfg(not(target_family = "wasm"))]
impl Upstream {
    pub fn start(answer: impl Fn(&Seen, usize) -> Reply + Send + 'static) -> Self {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("a free port on this machine");
        let address = server.server_addr().to_ip().expect("an address");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (record, stopping) = (seen.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let Ok(Some(mut request)) = server.recv_timeout(Duration::from_millis(20)) else {
                    continue;
                };
                let mut body = Vec::new();
                let _ = request.as_reader().read_to_end(&mut body);
                let seen = Seen {
                    method: request.method().as_str().to_string(),
                    url: request.url().to_string(),
                    headers: request
                        .headers()
                        .iter()
                        .map(|header| {
                            (
                                header.field.as_str().as_str().to_ascii_lowercase(),
                                header.value.as_str().to_string(),
                            )
                        })
                        .collect(),
                    body,
                };
                let before = {
                    let mut record = record.lock().unwrap();
                    record.push(seen.clone());
                    record.len() - 1
                };
                let reply = answer(&seen, before);
                std::thread::sleep(reply.after);
                let mut response = tiny_http::Response::from_string(reply.body).with_status_code(reply.status);
                for (name, value) in &reply.headers {
                    response.add_header(tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap());
                }
                let _ = request.respond(response);
            }
        });
        Self {
            address,
            seen,
            stop,
            thread: Some(thread),
        }
    }

    /// A server that answers every request well, as the model `name`, reporting `tokens`.
    pub fn answering(name: &'static str, tokens: u64) -> Self {
        Self::start(move |seen, _| Reply::ok(body(&seen.request(), name, tokens)))
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Every request received so far.
    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    pub fn hits(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

#[cfg(not(target_family = "wasm"))]
impl Drop for Upstream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A response read off a socket: for what a client of the protocol would never send.
#[derive(Debug)]
pub struct Raw {
    pub status: u16,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Raw {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).expect("the body is JSON")
    }

    /// The `type` of the one error in the body.
    pub fn code(&self) -> String {
        self.json()["detail"][0]["type"].as_str().unwrap_or_default().to_string()
    }
}

/// Send one HTTP/1.1 request by hand and read the whole response.
pub fn http(address: SocketAddr, method: &str, path: &str, key: Option<&str>, body: Option<&[u8]>) -> Raw {
    let mut stream = TcpStream::connect(address).expect("the server is listening");
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n");
    if let Some(key) = key {
        head.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    if let Some(body) = body {
        head.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).unwrap();
    if let Some(body) = body {
        // A server may answer, and stop reading, before the whole of a body too large is sent.
        let _ = stream.write_all(body);
    }
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("a response with headers");
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split(' ').nth(1))
        .and_then(|code| code.parse().ok())
        .expect("a status");
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let chunked = headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.contains("chunked"));
    Raw {
        status,
        headers,
        body: if chunked { unchunk(body) } else { body.to_string() },
    }
}

fn unchunk(mut body: &str) -> String {
    let mut whole = String::new();
    while let Some((size, rest)) = body.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if size == 0 || rest.len() < size {
            break;
        }
        whole.push_str(&rest[..size]);
        body = rest[size..].trim_start_matches("\r\n");
    }
    whole
}

/// The status, code and message of an error a server answered with.
pub fn api(error: &Error) -> (u16, String, String) {
    match error {
        Error::Api { status, code, message } => (*status, code.clone(), message.clone()),
        other => panic!("not a refusal from a server: {other:?}"),
    }
}
