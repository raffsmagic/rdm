//! A System One server, over HTTP.

use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::time::Duration;

use systemone::{DecisionModel, Error, Json, Request, Response};

use crate::schema::{MAX_BODY_BYTES, mismatch};
use crate::transport::{self, HttpRequest, HttpResponse, Transport, TransportError};

/// The API key a client sends, or where it finds it. It is never printed.
#[derive(Clone, PartialEq, Eq)]
enum Key {
    /// A server that asks for none.
    None,
    /// A key handed over directly.
    Given(String),
    /// The name of the environment variable that holds it, read at each call.
    Env(String),
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::None => f.write_str("none"),
            Key::Given(_) => f.write_str("[given]"),
            Key::Env(name) => write!(f, "${name}"),
        }
    }
}

impl Key {
    /// The key as it is sent. One that could not be a bearer token is refused here, before
    /// any request, and the refusal does not repeat it.
    fn reveal(&self) -> Result<Option<String>, Error> {
        let (key, source) = match self {
            Key::None => return Ok(None),
            Key::Given(key) => (key.clone(), "The API key given".to_string()),
            Key::Env(name) => match std::env::var(name) {
                Ok(key) => (key, format!("The API key in the environment variable {name}")),
                Err(_) => return Err(Error::Invalid(format!("The environment variable {name} holds no API key."))),
            },
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(Error::Invalid(format!("{source} is empty.")));
        }
        if key.chars().any(|character| !character.is_ascii_graphic()) {
            return Err(Error::Invalid(format!(
                "{source} is not usable: a key is printable ASCII with no space in it."
            )));
        }
        Ok(Some(key.to_string()))
    }
}

/// When a failed request is asked again, and how long is waited first.
///
/// The default is what TypeSafe's own SDKs do: up to 2 more attempts after a 408, a 429, any
/// 5xx (a 529 among them) or a connection that failed; a wait of half a second that doubles
/// up to 5 seconds, less a random part of up to a quarter; and a server's `retry-after`
/// honoured in place of that wait when it asks for a minute or less.
///
/// A [`Client`] does not retry unless it is given one of these. A gateway must not: a request
/// that is asked twice is paid for twice, and the caller is the one who decides that.
///
/// ```
/// use one_system::{Client, Retry};
/// use std::time::Duration;
///
/// let patient = Client::typesafe().retry(Retry { max_retries: 4, ..Retry::default() });
/// assert_eq!(Retry::default().initial, Duration::from_millis(500));
/// # let _ = patient;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Retry {
    /// How many more times a request is asked after it first fails.
    pub max_retries: u32,
    /// The wait before the first retry. Each one after waits twice as long.
    pub initial: Duration,
    /// The longest a wait grows to.
    pub max: Duration,
    /// The share of each wait, from 0 to 1, up to which a random part is taken off, so that
    /// clients that failed together do not all come back together.
    pub jitter: f64,
    /// The longest wait a server may ask for with `retry-after`. A longer one is not honoured.
    pub max_retry_after: Duration,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial: Duration::from_millis(500),
            max: Duration::from_secs(5),
            jitter: 0.25,
            max_retry_after: Duration::from_secs(60),
        }
    }
}

impl Retry {
    /// The wait before retry number `attempt` (from 0), given what the server asked for.
    fn delay(&self, attempt: u32, asked: Option<Duration>) -> Duration {
        if let Some(asked) = asked
            && asked <= self.max_retry_after
        {
            return asked;
        }
        let doubled = self.initial.saturating_mul(2u32.saturating_pow(attempt)).min(self.max);
        let jitter = self.jitter.clamp(0.0, 1.0) * random_unit();
        doubled.mul_f64(1.0 - jitter)
    }
}

/// A number from 0 to 1, from the randomness the standard library seeds its hash maps with.
fn random_unit() -> f64 {
    let bits = std::collections::hash_map::RandomState::new().build_hasher().finish();
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

/// A decision model reached over HTTP: `POST {base}/v1/systemone` with a bearer key.
///
/// It talks to anything that serves the protocol: hosted TypeSafe, OpenRouter, a local
/// llama.cpp serving Laya, or a one-system gateway. A request that names no model is sent
/// with the client's own.
///
/// One call is one request. Nothing is retried unless [`Client::retry`] says so, a redirect
/// is never followed (it would carry the key somewhere else), and a response is read only up
/// to 8 MiB. A refusal comes back as [`systemone::Error::Api`] whichever of the three error
/// bodies in use the server wrote, and answers that are not for the questions asked come back
/// as [`systemone::Error::Evidence`].
///
/// The key is given directly or named as an environment variable, which is read at each
/// call. It is not in the client's `Debug` output and not in any error. A base URL is no
/// place for one: credentials written into it (`https://user:password@host`) are not kept,
/// not printed and not sent, and a client built from such a URL refuses every call with
/// [`systemone::Error::Invalid`].
///
/// A proxy is taken from the environment when the client is built, the way Go's
/// `http.ProxyFromEnvironment` takes one: `HTTPS_PROXY` for an `https` server and `HTTP_PROXY`
/// for an `http` one (or the same names in lower case), with `ALL_PROXY` for a scheme that
/// has none of its own, and not for a host `NO_PROXY` names. A server on this machine
/// (`localhost`, or a loopback address) is never asked through a proxy, whatever is set.
///
/// In WebAssembly the network is the host's. A module built for `wasm32-wasip2` asks through
/// WASI's `wasi:http`, and whether a proxy is used is the host's to say. Any other module is
/// given its transport ([`Client::transport`]), and refuses every call until it is.
///
/// ```no_run
/// use one_system::Client;
/// use systemone::{DecisionModel, Question, Request};
///
/// let jev = Client::typesafe(); // https://api.typesafe.ai, jev-latest, $TYPESAFE_API_KEY
/// let request = Request::new("I was charged twice.").question("refund", Question::noul("Is a refund asked for?"));
/// let response = jev.decide(&request)?;
/// println!("{:?}", response.answers["refund"].noul);
/// # Ok::<(), systemone::Error>(())
/// ```
#[derive(Clone)]
pub struct Client {
    base_url: String,
    model: String,
    key: Key,
    timeout: Duration,
    retry: Option<Retry>,
    /// Whether the base URL it was given held credentials. They are not kept.
    credentials: bool,
    /// Whether the server reads the pictures of a state ([`Client::images`]).
    images: bool,
    transport: Arc<dyn Transport>,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("key", &self.key)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .field("images", &self.images)
            .finish()
    }
}

impl Client {
    /// A client for the server at a base URL, such as `http://127.0.0.1:8080`. A path in the
    /// base is kept, and `/v1/systemone` is added to it. It sends no key and names no model
    /// until it is given them.
    ///
    /// The environment is read here, once, for a proxy (see [`Client`]).
    pub fn new(base_url: impl Into<String>) -> Self {
        let (base_url, credentials) = without_credentials(base_url.into());
        let base_url = base_url.trim_end_matches('/').to_string();
        let transport = transport::of_this_build(&base_url);
        let images = !typesafe(&base_url);
        Self {
            base_url,
            model: String::new(),
            key: Key::None,
            timeout: Duration::from_secs(180),
            retry: None,
            credentials,
            images,
            transport,
        }
    }

    /// Hosted TypeSafe: `https://api.typesafe.ai`, the model `jev-latest`, and the key in
    /// `TYPESAFE_API_KEY`.
    ///
    /// It refuses a request with pictures in it ([`Client::images`]): TypeSafe's JSON endpoint
    /// does not read an image part of a state. Checked 2026-10-17: Jev there gave a red picture
    /// and a green one the same answer, as if it had seen neither. TypeSafe takes pictures as
    /// multipart assets, which this client does not send. OpenRouter's decisions read them.
    ///
    /// ```
    /// use one_system::Client;
    /// use systemone::{DecisionModel, Question, Request};
    ///
    /// let picture = Request::new(serde_json::json!([{"inputs": {"window": "<image: window>"}}, systemone::image("image/png", b"png")]))
    ///     .question("editor", Question::noul("Is `inputs.window` an editor?"));
    /// let refused = Client::typesafe().key("unused").decide(&picture).unwrap_err();
    /// assert!(refused.to_string().contains("does not read pictures"));
    /// ```
    pub fn typesafe() -> Self {
        Self::new("https://api.typesafe.ai").model("jev-latest").key_env("TYPESAFE_API_KEY")
    }

    /// Jev through OpenRouter: `https://openrouter.ai/api`, the model `~typesafe/jev-latest`,
    /// and the key in `OPENROUTER_API_KEY`.
    pub fn openrouter() -> Self {
        Self::new("https://openrouter.ai/api")
            .model("~typesafe/jev-latest")
            .key_env("OPENROUTER_API_KEY")
    }

    /// A server on this machine that asks for no key and has one model loaded, as
    /// `llama-server -hf ggml-org/Laya-GGUF` is at `http://127.0.0.1:8080`. Laya reads text,
    /// so it refuses a request with pictures in it unless told otherwise ([`Client::images`]).
    pub fn local(base_url: impl Into<String>) -> Self {
        Self::new(base_url).images(false)
    }

    /// Whether the server reads the pictures of a state ([`systemone::image`]). One that does
    /// not is never sent a request with pictures in it: the client refuses it, before anything
    /// is sent, with [`systemone::Error::Invalid`].
    ///
    /// A client reads them unless this says otherwise, but for one at TypeSafe's address
    /// ([`Client::typesafe`]), whose JSON endpoint does not, and one made with [`Client::local`].
    pub fn images(mut self, read: bool) -> Self {
        self.images = read;
        self
    }

    /// The model asked for when a request names none.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// The API key itself.
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Key::Given(key.into());
        self
    }

    /// The name of the environment variable that holds the API key. It is read at each call,
    /// so a client can be built before the variable is set.
    pub fn key_env(mut self, variable: impl Into<String>) -> Self {
        self.key = Key::Env(variable.into());
        self
    }

    /// Send no key at all.
    pub fn no_key(mut self) -> Self {
        self.key = Key::None;
        self
    }

    /// How long one request may take, from connecting to the last byte of the answer.
    /// 180 seconds unless set. With retries on, each attempt has this long.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Ask again after a failure that asking again could cure. `Retry::default()` is what the
    /// TypeSafe SDKs do.
    pub fn retry(mut self, retry: Retry) -> Self {
        self.retry = Some(retry);
        self
    }

    /// What carries the requests, in place of this crate's own network: a host's, where the
    /// client is in a WebAssembly module, or anything else that answers as a server would.
    /// See [`Transport`].
    pub fn transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Arc::new(transport);
        self
    }

    /// The base URL requests go to, with no `/` at the end and no credentials in it.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The key as it is sent, if the client is one that may send anything.
    fn ready(&self) -> Result<Option<String>, Error> {
        if self.credentials {
            return Err(Error::Invalid(
                "The base URL holds credentials, which are not sent. A key is given with `key` or `key_env`.".into(),
            ));
        }
        self.key.reveal()
    }

    /// The models the server lists: the body of `GET {base}/v1/models`, as it came.
    pub fn models(&self) -> Result<Json, Error> {
        let key = self.ready()?;
        let (status, body, _) = self.exchange("/v1/models", None, key.as_deref())?;
        if status != 200 {
            return Err(refusal(status, &body, key.as_deref()));
        }
        serde_json::from_slice(&body).map_err(|_| Error::Evidence("The server's list of models is not JSON.".into()))
    }

    /// One request and its response: the status, the body, and the wait the server asked for.
    fn exchange(&self, path: &str, body: Option<&[u8]>, key: Option<&str>) -> Result<(u16, Vec<u8>, Option<Duration>), Error> {
        let mut headers = vec![("Accept".to_owned(), "application/json".to_owned())];
        if body.is_some() {
            headers.insert(0, ("Content-Type".to_owned(), "application/json".to_owned()));
        }
        if let Some(key) = key {
            headers.push(("Authorization".to_owned(), format!("Bearer {key}")));
        }
        let request = HttpRequest {
            method: if body.is_some() { "POST" } else { "GET" },
            url: format!("{}{path}", self.base_url),
            headers,
            body: body.map(<[u8]>::to_vec),
            timeout: self.timeout,
            // A body of exactly 8 MiB is a body that fits.
            limit: MAX_BODY_BYTES,
        };
        let response = self.transport.send(request).map_err(|error| self.unanswered(error, key))?;
        let wait = retry_after(&response);
        let HttpResponse { status, body, .. } = response;
        match body.len() <= MAX_BODY_BYTES {
            true => Ok((status, body, wait)),
            // A refusal that long is still a refusal: its status is kept, and its words are not.
            false if status != 200 => Ok((status, Vec::new(), wait)),
            false => Err(Error::Evidence("The server's response is larger than 8 MiB.".into())),
        }
    }

    /// A request that got no response, as an error that does not name the key.
    fn unanswered(&self, error: TransportError, key: Option<&str>) -> Error {
        match error {
            TransportError::TimedOut => Error::Transport(format!("The decision model did not answer within {:?}.", self.timeout)),
            TransportError::Failed(why) => Error::Transport(scrub(format!("The request to the decision model failed: {why}."), key)),
        }
    }

    /// One attempt: the answers, or the error with the wait the server asked for.
    fn once(&self, body: &[u8], key: Option<&str>, request: &Request) -> Result<Response, (Error, Option<Duration>)> {
        let (status, bytes, wait) = self.exchange("/v1/systemone", Some(body), key).map_err(|error| (error, None))?;
        if status != 200 {
            return Err((refusal(status, &bytes, key), wait));
        }
        let response: Response = serde_json::from_slice(&bytes)
            .map_err(|error| {
                Error::Evidence(format!(
                    "The server's response is not a System One response ({}).",
                    category(&error)
                ))
            })
            .map_err(|error| (error, None))?;
        if let Some(fault) = mismatch(&request.questions, &response.answers) {
            return Err((fault.evidence(), None));
        }
        Ok(response)
    }
}

impl DecisionModel for Client {
    fn decide(&self, request: &Request) -> Result<Response, Error> {
        if !self.images && request.images() > 0 {
            let why = if typesafe(&self.base_url) {
                "TypeSafe's JSON endpoint does not read pictures in a state (it takes them as multipart assets, which this client does not send); OpenRouter's decisions read them"
            } else {
                "the server is not known to read them: say it does with `Client::images(true)`"
            };
            return Err(Error::Invalid(format!(
                "The request shows {} picture(s), and {} does not read pictures: {why}.",
                request.images(),
                self.base_url
            )));
        }
        let key = self.ready()?;
        let body = if request.model.is_empty() && !self.model.is_empty() {
            serde_json::to_vec(&request.clone().model(self.model.clone()))
        } else {
            serde_json::to_vec(request)
        }
        .map_err(|_| Error::Invalid("The request cannot be written as JSON.".into()))?;
        let mut attempt = 0;
        loop {
            let (error, wait) = match self.once(&body, key.as_deref(), request) {
                Ok(response) => return Ok(response),
                Err(failure) => failure,
            };
            match &self.retry {
                Some(retry) if attempt < retry.max_retries && error.is_retryable() => {
                    pause(retry.delay(attempt, wait));
                    attempt += 1;
                }
                _ => return Err(error),
            }
        }
    }

    /// The model it asks for, or the server it asks when it names none.
    fn name(&self) -> String {
        if self.model.is_empty() {
            self.base_url.clone()
        } else {
            self.model.clone()
        }
    }
}

/// What kind of thing was wrong with a body, without repeating any of it.
fn category(error: &serde_json::Error) -> &'static str {
    match error.classify() {
        serde_json::error::Category::Data => "it is JSON of another shape",
        serde_json::error::Category::Eof => "it is cut short",
        _ => "it is not JSON",
    }
}

/// Take a key out of a text that might, by some server's doing, hold it: as it is, and as
/// JSON spells it where that is another spelling. This is done to the whole of a text, before
/// any of it is cut off: a key cut in two is not found, and its first half is still a secret.
fn scrub(text: String, key: Option<&str>) -> String {
    let Some(key) = key else {
        return text;
    };
    let mut text = if text.contains(key) { text.replace(key, "[key]") } else { text };
    if key.contains(['"', '\\'])
        && let Ok(quoted) = serde_json::to_string(key)
    {
        let spelled = &quoted[1..quoted.len() - 1];
        if text.contains(spelled) {
            text = text.replace(spelled, "[key]");
        }
    }
    text
}

/// A base URL without the credentials somebody wrote into it (`user:password@`, before the
/// host), and whether there were any.
fn without_credentials(base_url: String) -> (String, bool) {
    let start = base_url.find("://").map_or(0, |at| at + 3);
    let end = base_url[start..].find(['/', '?', '#']).map_or(base_url.len(), |at| start + at);
    match base_url[start..end].rfind('@') {
        Some(at) => (format!("{}{}", &base_url[..start], &base_url[start + at + 1..]), true),
        None => (base_url, false),
    }
}

/// Whether a base URL is TypeSafe's own, whose JSON endpoint does not read pictures.
fn typesafe(base_url: &str) -> bool {
    let start = base_url.find("://").map_or(0, |at| at + 3);
    let end = base_url[start..].find(['/', '?', '#', ':']).map_or(base_url.len(), |at| start + at);
    let host = base_url[start..end].to_ascii_lowercase();
    host == "typesafe.ai" || host.ends_with(".typesafe.ai")
}

/// Wait before a request is asked again. A WebAssembly module with no system under it has no
/// clock to wait by, and waiting there is a panic: the request is asked again at once.
fn pause(wait: Duration) {
    if cfg!(not(all(target_family = "wasm", target_os = "unknown"))) {
        std::thread::sleep(wait);
    }
}

/// The wait a server asked for: `retry-after-ms` in milliseconds, else `retry-after` in seconds.
fn retry_after(response: &HttpResponse) -> Option<Duration> {
    let number = |name: &str| response.header(name)?.trim().parse::<f64>().ok();
    let seconds = number("retry-after-ms").map(|ms| ms / 1000.0).or_else(|| number("retry-after"))?;
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds.min(86_400.0)))
}

/// How much of a server's own words an error keeps.
const MESSAGE_CHARACTERS: usize = 500;

fn clip(text: &str) -> String {
    let text = text.trim();
    match text.char_indices().nth(MESSAGE_CHARACTERS) {
        Some((end, _)) => format!("{}...", &text[..end]),
        None => text.to_string(),
    }
}

/// A refusal, read from whichever of the bodies servers write:
///
/// - `{"detail": [{"loc", "msg", "type"}, ...]}`: a validation error from TypeSafe, and every
///   error of a one-system gateway;
/// - `{"detail": {"error_type", "message"}}`: hosted TypeSafe's 400, 401 and 403;
/// - `{"error": {"code", "message", "type"}}`: llama.cpp, and OpenRouter.
///
/// What a validation error echoes of the request (`input`) is left out.
fn refusal(status: u16, body: &[u8], key: Option<&str>) -> Error {
    let text = |value: Option<&Json>| value.and_then(Json::as_str).unwrap_or_default().to_string();
    let parsed: Option<Json> = serde_json::from_slice(body).ok();
    let (code, message) = match parsed.as_ref().map(|body| (body.get("detail"), body.get("error"))) {
        Some((Some(Json::Array(entries)), _)) if !entries.is_empty() => {
            let code = text(entries[0].get("type"));
            let parts: Vec<String> = entries
                .iter()
                .map(|entry| {
                    let message = text(entry.get("msg"));
                    let place: Vec<String> = entry
                        .get("loc")
                        .and_then(Json::as_array)
                        .map(|loc| {
                            loc.iter()
                                .map(|part| part.as_str().map_or_else(|| part.to_string(), str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    // `["body"]` alone says nothing about where.
                    if place.len() > 1 {
                        format!("{message} ({})", place.join("."))
                    } else {
                        message
                    }
                })
                .collect();
            (code, parts.join("; "))
        }
        Some((Some(Json::Object(detail)), _)) => (text(detail.get("error_type")), text(detail.get("message"))),
        Some((Some(Json::String(detail)), _)) => (String::new(), detail.clone()),
        Some((_, Some(Json::Object(error)))) => {
            let code = match (error.get("type"), error.get("code")) {
                (Some(Json::String(kind)), _) => kind.clone(),
                (_, Some(Json::String(code))) => code.clone(),
                _ => String::new(),
            };
            (code, text(error.get("message")))
        }
        Some((_, Some(Json::String(error)))) => (String::new(), error.clone()),
        // Anything else is some other server's page. Its start is kept, to say what answered.
        _ if (300..400).contains(&status) => (String::new(), "The server answered with a redirect, which is not followed.".into()),
        _ => (String::new(), String::from_utf8_lossy(body).into_owned()),
    };
    // The key is taken out first, and only then is anything cut off.
    let (code, message) = (scrub(code, key), scrub(message, key));
    let message = if message.trim().is_empty() {
        format!("The server answered with status {status}.")
    } else {
        clip(&message)
    };
    Error::Api {
        status,
        code: clip(&code),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_bodies() {
        let read = |status: u16, body: &str| match refusal(status, body.as_bytes(), None) {
            Error::Api { status, code, message } => (status, code, message),
            other => panic!("{other:?}"),
        };
        let validation = r#"{"detail":[{"type":"missing","loc":["body","questions","q","choice","criteria"],"msg":"Field required","input":{"type":"choice","instructions":"SECRET"}}]}"#;
        assert_eq!(
            read(422, validation),
            (422, "missing".into(), "Field required (body.questions.q.choice.criteria)".into())
        );
        let gateway = r#"{"detail":[{"loc":["body"],"msg":"Unknown model; see GET /v1/models","type":"unsupported_model"}]}"#;
        assert_eq!(
            read(422, gateway),
            (422, "unsupported_model".into(), "Unknown model; see GET /v1/models".into())
        );
        let hosted = r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server."}}"#;
        assert_eq!(
            read(401, hosted),
            (401, "authentication_error".into(), "Cannot authenticate with the server.".into())
        );
        let no_message = r#"{"detail":{"error_type":"max_tokens_exceeded"}}"#;
        assert_eq!(
            read(400, no_message),
            (400, "max_tokens_exceeded".into(), "The server answered with status 400.".into())
        );
        let llama =
            r#"{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error"}}"#;
        assert_eq!(
            read(400, llama),
            (
                400,
                "exceed_context_size_error".into(),
                "the request exceeds the available context size".into()
            )
        );
        let openrouter = r#"{"error":{"code":402,"message":"Insufficient credits"}}"#;
        assert_eq!(read(402, openrouter), (402, String::new(), "Insufficient credits".into()));
        assert_eq!(read(404, r#"{"detail":"Not Found"}"#), (404, String::new(), "Not Found".into()));
        assert_eq!(
            read(502, "<html>Bad Gateway</html>"),
            (502, String::new(), "<html>Bad Gateway</html>".into())
        );
        assert_eq!(read(503, ""), (503, String::new(), "The server answered with status 503.".into()));
        assert_eq!(
            read(307, ""),
            (
                307,
                String::new(),
                "The server answered with a redirect, which is not followed.".into()
            )
        );
    }

    #[test]
    fn a_refusal_never_repeats_the_key_or_the_request() {
        let echoed = r#"{"error":{"message":"bad key sk-secret-1","type":"sk-secret-1"}}"#;
        let shown = refusal(401, echoed.as_bytes(), Some("sk-secret-1")).to_string();
        assert!(!shown.contains("sk-secret-1") && shown.contains("[key]"));
        let validation = r#"{"detail":[{"type":"missing","loc":["body"],"msg":"Field required","input":{"instructions":"PRIVATE"}}]}"#;
        assert!(!refusal(422, validation.as_bytes(), None).to_string().contains("PRIVATE"));
        let long = "x".repeat(10_000);
        assert!(refusal(500, long.as_bytes(), None).to_string().len() < 600);
    }

    #[test]
    fn a_key_is_taken_out_wherever_the_cut_falls() {
        let key = "sk-live-0123456789abcdefghijklmnopqrstuvwxyz";
        // The key begins at each place from well before the cut to just after it.
        for before in (MESSAGE_CHARACTERS - key.len() - 5)..=(MESSAGE_CHARACTERS + 5) {
            let filler = "x".repeat(before);
            for body in [
                format!("{filler}{key} and more"),
                format!(r#"{{"error":{{"message":"{filler}{key}","type":"{filler}{key}"}}}}"#),
                format!(r#"{{"detail":{{"error_type":"{filler}{key}","message":"{filler}{key}"}}}}"#),
                format!(r#"{{"detail":[{{"type":"{filler}{key}","loc":["body","{filler}{key}"],"msg":"{filler}{key}"}}]}}"#),
            ] {
                let shown = format!("{:?}", refusal(401, body.as_bytes(), Some(key)));
                assert!(!shown.contains(&key[..8]), "{before}: {shown}");
            }
        }
        // A key with a quote in it, said back inside JSON this client does not know the shape of.
        let odd = r#"sk-"quoted"\key"#;
        let body = serde_json::json!({"problem": format!("no such key {odd}")}).to_string();
        let shown = refusal(401, body.as_bytes(), Some(odd)).to_string();
        assert!(!shown.contains("quoted") && shown.contains("[key]"), "{shown}");
    }

    #[test]
    fn credentials_are_taken_out_of_a_base_url() {
        let clean = |url: &str| without_credentials(url.to_string());
        assert_eq!(clean("https://user:pass@example.com/api"), ("https://example.com/api".into(), true));
        assert_eq!(clean("https://token@example.com"), ("https://example.com".into(), true));
        assert_eq!(clean("https://a@b@example.com:8443"), ("https://example.com:8443".into(), true));
        assert_eq!(clean("user:pass@example.com/api"), ("example.com/api".into(), true));
        for plain in [
            "https://example.com",
            "http://[::1]:8080/a@b",
            "https://example.com/?x=a@b",
            "",
            "example.com",
        ] {
            assert_eq!(clean(plain), (plain.to_string(), false));
        }
    }

    #[test]
    fn waits() {
        let retry = Retry {
            jitter: 0.0,
            ..Retry::default()
        };
        let millis = |attempt| retry.delay(attempt, None).as_millis();
        assert_eq!(
            [millis(0), millis(1), millis(2), millis(3), millis(4), millis(40)],
            [500, 1000, 2000, 4000, 5000, 5000]
        );
        assert_eq!(retry.delay(0, Some(Duration::from_secs(7))), Duration::from_secs(7));
        // A server that asks for longer than is honoured gets the ordinary wait.
        assert_eq!(retry.delay(0, Some(Duration::from_secs(61))), Duration::from_millis(500));
        let jittered = Retry::default();
        for _ in 0..50 {
            let wait = jittered.delay(0, None);
            assert!(wait <= Duration::from_millis(500) && wait >= Duration::from_millis(375));
        }

        let asked = |headers: &[(&str, &str)]| {
            let headers = headers.iter().map(|(name, value)| (name.to_string(), value.to_string()));
            retry_after(&HttpResponse {
                headers: headers.collect(),
                ..HttpResponse::default()
            })
        };
        assert_eq!(asked(&[]), None);
        assert_eq!(asked(&[("retry-after", "2")]), Some(Duration::from_secs(2)));
        assert_eq!(
            asked(&[("Retry-After", "2"), ("retry-after-ms", "250")]),
            Some(Duration::from_millis(250))
        );
        let dated = [("retry-after", "2"), ("retry-after-ms", "Wed, 21 Oct 2026 07:28:00 GMT")];
        assert_eq!(asked(&dated), Some(Duration::from_secs(2)));
    }

    #[test]
    fn keys_that_cannot_be_sent() {
        assert_eq!(Key::None.reveal(), Ok(None));
        assert_eq!(Key::Given("  sk-1\n".into()).reveal(), Ok(Some("sk-1".into())));
        for bad in ["", "   ", "sk 1", "sk-\u{7f}", "clé"] {
            let error = Key::Given(bad.into()).reveal().unwrap_err();
            assert!(matches!(&error, Error::Invalid(_)));
            assert!(bad.trim().is_empty() || !error.to_string().contains(bad.trim()));
        }
        let unset = Key::Env("ONE_SYSTEM_TEST_A_VARIABLE_NOBODY_SETS".into()).reveal().unwrap_err();
        assert!(unset.to_string().contains("ONE_SYSTEM_TEST_A_VARIABLE_NOBODY_SETS"));
        assert_eq!(format!("{:?}", Key::Given("sk-secret".into())), "[given]");
        assert_eq!(format!("{:?}", Key::Env("TYPESAFE_API_KEY".into())), "$TYPESAFE_API_KEY");
    }
}
