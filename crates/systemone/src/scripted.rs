//! A decision model that is a function.

use std::sync::Mutex;

use indexmap::IndexMap;

use crate::wire::{Answer, Request, Response, Usage};
use crate::{DecisionModel, Error};

type Answers = dyn Fn(&Request) -> Result<IndexMap<String, Answer>, Error> + Send + Sync;

/// A stand-in for a decision model: a function from a request to its answers.
///
/// It keeps every request it was asked, so a test can read what a model was shown. It
/// reports the tokens it is told to, so limits can be tested, and is otherwise free.
///
/// ```
/// use systemone::{Answer, DecisionModel, Question, Request, Scripted};
///
/// let model = Scripted::new(|request| {
///     let text = request.state["inputs"]["text"].as_str().unwrap_or_default();
///     [("look", Answer::noul(if text.contains("London") { 0.9 } else { 0.1 }))]
/// });
/// let request = Request::new(serde_json::json!({"inputs": {"text": "260 euros in London"}}))
///     .question("look", Question::noul("Is the answer stated in `inputs.text`?"));
/// assert_eq!(model.decide(&request).unwrap().answers["look"].noul, Some(0.9));
/// assert_eq!(model.requests().len(), 1);
/// ```
pub struct Scripted {
    answers: Box<Answers>,
    name: String,
    tokens: u64,
    asked: Mutex<Vec<Request>>,
}

impl std::fmt::Debug for Scripted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let asked = self.asked.try_lock().map(|asked| asked.len()).ok();
        f.debug_struct("Scripted")
            .field("name", &self.name)
            .field("tokens", &self.tokens)
            .field("asked", &asked)
            .finish_non_exhaustive()
    }
}

impl Scripted {
    /// A model that answers with what the function returns for each request.
    pub fn new<F, A, L>(answers: F) -> Self
    where
        F: Fn(&Request) -> A + Send + Sync + 'static,
        A: IntoIterator<Item = (L, Answer)>,
        L: Into<String>,
    {
        Self::fallible(move |request| Ok(answers(request).into_iter().map(|(name, answer)| (name.into(), answer)).collect()))
    }

    /// As [`Scripted::new`], for a function that can fail as a server can.
    pub fn fallible<F>(answers: F) -> Self
    where
        F: Fn(&Request) -> Result<IndexMap<String, Answer>, Error> + Send + Sync + 'static,
    {
        Self {
            answers: Box::new(answers),
            name: "scripted".into(),
            tokens: 0,
            asked: Mutex::new(Vec::new()),
        }
    }

    /// The name it answers under.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The input tokens it reports for every request.
    pub fn tokens(mut self, tokens: u64) -> Self {
        self.tokens = tokens;
        self
    }

    /// Every request it has been asked, in the order they arrived.
    pub fn requests(&self) -> Vec<Request> {
        self.asked.lock().expect("the record of requests").clone()
    }

    /// Forget the requests it has been asked.
    pub fn clear(&self) {
        self.asked.lock().expect("the record of requests").clear();
    }
}

impl DecisionModel for Scripted {
    fn decide(&self, request: &Request) -> Result<Response, Error> {
        self.asked.lock().expect("the record of requests").push(request.clone());
        let mut answers = (self.answers)(request)?;
        // A question the function did not answer is an error a real server would not make.
        if let Some(missing) = request.questions.keys().find(|name| !answers.contains_key(*name)) {
            return Err(Error::Evidence(format!("The scripted model gave no answer for '{missing}'.")));
        }
        answers.retain(|name, _| request.questions.contains_key(name));
        let usage = Usage {
            input_tokens: self.tokens,
            output_tokens: 0,
        };
        Ok(Response {
            model: self.name.clone(),
            answers,
            usage,
        })
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
