//! What a model answered, kept, and answered again.
//!
//! A decision model's answer is the one part of a run that cannot be worked out again: the
//! probabilities it gave. A decision is made from them by arithmetic ([`crate::decide`]), so
//! what is worth keeping is the request and the response, and the key they are kept under is
//! the request itself. [`key`] is that key: the same model asked the same questions about the
//! same state is the same key, in any process, on any machine.
//!
//! The key is of the request and of nothing else. A model that adds evidence of its own on the
//! way, a clip of sound it attaches say, is asked about more than its key says: two clips
//! asked the same questions are one key, and a replay would answer the second with the first's
//! answers. What a model is shown has to be in the request for a replay of it to be one.
//!
//! [`Replay`] is a model that answers from what was kept. A request that was never asked is
//! [`Error::Unrecorded`]: a replay that would have to ask something new has come to where the
//! run that was kept and this one part, and from there it is a new run, not a replay.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::wire::{Request, Response};
use crate::{DecisionModel, Error};

/// The key a request's answers are kept under: the SHA-256 of the request as it is sent, in
/// hex. A request that names no model is keyed as if it named `model`, the model it was put
/// to, so the same questions put to two models are two keys. A picture in the state is in
/// the request as its bytes, so two pictures asked the same questions are two keys.
///
/// ```
/// use systemone::{Question, Request, key};
///
/// let request = Request::new(serde_json::json!({"inputs": {"text": "I was charged twice."}}))
///     .question("refund", Question::noul("Is `inputs.text` asking for a refund?"));
/// assert_eq!(key(&request, "jev"), key(&request.clone().model("jev"), "another"));
/// assert_ne!(key(&request, "jev"), key(&request, "laya"));
/// ```
pub fn key(request: &Request, model: &str) -> String {
    let named;
    let request = if request.model.is_empty() {
        named = request.clone().model(model);
        &named
    } else {
        request
    };
    let mut hash = Sha256::new();
    if serde_json::to_writer(&mut hash, request).is_err() {
        // A request is JSON, and JSON can always be written.
        hash.update(b"unwritten");
    }
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A decision model that answers each request with the response it got when it was asked
/// before, and refuses one that was never asked.
///
/// ```
/// use systemone::{Answer, DecisionModel, Error, Question, Replay, Request, Scripted};
///
/// let live = Scripted::new(|_| [("refund", Answer::noul(0.93))]).named("jev");
/// let asked = Request::new(serde_json::json!({"inputs": {"text": "I was charged twice."}}))
///     .question("refund", Question::noul("Is `inputs.text` asking for a refund?"));
/// let answered = live.decide(&asked).unwrap();
///
/// let replay = Replay::new("jev").kept(&asked, answered.clone());
/// assert_eq!(replay.decide(&asked).unwrap(), answered);
///
/// let other = Request::new(serde_json::json!({"inputs": {"text": "Where is my parcel?"}}))
///     .question("refund", Question::noul("Is `inputs.text` asking for a refund?"));
/// assert!(matches!(replay.decide(&other), Err(Error::Unrecorded(_))));
/// ```
pub struct Replay {
    name: String,
    kept: HashMap<String, Result<Response, Error>>,
}

impl std::fmt::Debug for Replay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Replay")
            .field("name", &self.name)
            .field("kept", &self.kept.len())
            .finish()
    }
}

impl Replay {
    /// A replay of the model of this name, with nothing kept yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kept: HashMap::new(),
        }
    }

    /// The same replay, answering `request` with `response`. A request kept twice is answered
    /// with the later response.
    pub fn kept(mut self, request: &Request, response: Response) -> Self {
        self.keep(request, response);
        self
    }

    /// Answer `request` with `response` from now on.
    pub fn keep(&mut self, request: &Request, response: Response) {
        self.kept.insert(key(request, &self.name), Ok(response));
    }

    /// Fail `request` as it failed when it was asked: a failure that was kept is given again,
    /// and is not a request that was never asked.
    pub fn keep_failure(&mut self, request: &Request, error: Error) {
        self.kept.insert(key(request, &self.name), Err(error));
    }

    /// Answer, or fail, the request kept under `key` as it was answered then: for a log that
    /// holds the key and not the request.
    pub fn keep_key(&mut self, key: impl Into<String>, answered: Result<Response, Error>) {
        self.kept.insert(key.into(), answered);
    }

    /// How many requests it can answer.
    pub fn len(&self) -> usize {
        self.kept.len()
    }

    /// Whether it can answer none.
    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }
}

impl DecisionModel for Replay {
    fn decide(&self, request: &Request) -> Result<Response, Error> {
        let key = key(request, &self.name);
        let Some(answered) = self.kept.get(&key) else {
            let questions: Vec<&str> = request.questions.keys().map(String::as_str).collect();
            return Err(Error::Unrecorded(format!(
                "{} was never asked this request ({}), so there is no answer kept for it: key {key}.",
                self.name,
                questions.join(", ")
            )));
        };
        answered.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
