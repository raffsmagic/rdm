//! The System One protocol.
//!
//! A decision model is given a state and typed questions about it, and answers each question
//! with a decision and a probability. It does not write. There are three kinds of question:
//! a **noul** (yes or no), a **choice** (one of several options) and a **score** (a level on
//! a rubric).
//!
//! This crate is the protocol and nothing else: the request and the response as they are on
//! the wire ([`Request`], [`Response`]), the one thing every decision model does
//! ([`DecisionModel`]), how a decision is made from what a model answered ([`decide`]), and
//! a model that is a function ([`Scripted`]), for tests and for examples that need no network,
//! and a model that answers from what another answered before ([`Replay`], under [`key`]).
//!
//! ```
//! use systemone::{Answer, DecisionModel, Params, Question, Request, Scripted, decide};
//!
//! let model = Scripted::new(|_request| [("refund", Answer::noul(0.93))]);
//! let question = Question::noul("Is the customer in `inputs.message` asking for a refund?");
//! let request = Request::new(serde_json::json!({"inputs": {"message": "I was charged twice."}}))
//!     .question("refund", question.clone());
//!
//! let response = model.decide(&request).unwrap();
//! let refund = decide("refund", &question, &response.answers["refund"], &Params::default()).unwrap();
//! assert_eq!(refund.value(), true);
//! ```

#![warn(missing_docs)]

mod decide;
mod image;
mod replay;
mod scripted;
mod wire;

pub use decide::{Choice, Decision, Noul, Params, Score, choice_confidence, decide, likeliest, score_confidence};
pub use image::{CONTENT_TYPES, IMAGE_REF, IMAGE_URL, Image, image, images, is_image};
/// The ordered map the wire types hold their questions, answers and options in, at the version
/// they are built with.
pub use indexmap;
pub use replay::{Replay, key};
pub use scripted::Scripted;
pub use wire::{Answer, Json, Kind, Question, Request, Response, Usage};

use std::fmt;
use std::sync::Arc;

/// Something that answers typed questions about a state: a hosted model, a local one, a
/// gateway that picks between them, or a function.
pub trait DecisionModel: Send + Sync {
    /// Answer every question of a request, once.
    ///
    /// The answers are evidence, not decisions: probabilities, from which [`decide`] decides.
    /// A request whose `model` is empty asks for whichever model this is.
    fn decide(&self, request: &Request) -> Result<Response, Error>;

    /// The name of the model, as a record of who answered would give it.
    fn name(&self) -> String;
}

impl<M: DecisionModel + ?Sized> DecisionModel for Arc<M> {
    fn decide(&self, request: &Request) -> Result<Response, Error> {
        (**self).decide(request)
    }

    fn name(&self) -> String {
        (**self).name()
    }
}

impl<M: DecisionModel + ?Sized> DecisionModel for Box<M> {
    fn decide(&self, request: &Request) -> Result<Response, Error> {
        (**self).decide(request)
    }

    fn name(&self) -> String {
        (**self).name()
    }
}

/// What can go wrong between a request and a decision.
///
/// It is written as `{"kind": "transport", "message": "..."}`, so a failure can be kept with
/// the request it was the answer to, and given again by a [`Replay`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
pub enum Error {
    /// The request, or a number a decision is made with, is not one that can be used.
    Invalid(String),
    /// A model answered with something that is not evidence for the question asked.
    Evidence(String),
    /// A server refused the request.
    Api {
        /// The HTTP status it answered with.
        status: u16,
        /// The server's own word for what was wrong, as `unauthorized` or `schema_validation`. Empty where it gave none.
        code: String,
        /// What the server said, in a sentence.
        message: String,
    },
    /// No answer came back: the connection failed, or timed out.
    Transport(String),
    /// A model that answers from what was kept ([`Replay`]) was asked something it never was.
    Unrecorded(String),
}

impl Error {
    /// Whether asking again later could succeed: a timeout, a rate limit, a server's own failure.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Transport(_) => true,
            Error::Api { status, .. } => matches!(status, 408 | 429) || *status >= 500,
            Error::Invalid(_) | Error::Evidence(_) | Error::Unrecorded(_) => false,
        }
    }

    /// The HTTP status a server refused with, where a server refused.
    pub fn status(&self) -> Option<u16> {
        if let Error::Api { status, .. } = self {
            Some(*status)
        } else {
            None
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message) | Error::Evidence(message) | Error::Transport(message) | Error::Unrecorded(message) => {
                f.write_str(message)
            }
            Error::Api { status, code, message } if code.is_empty() => write!(f, "{status}: {message}"),
            Error::Api { status, code, message } => write!(f, "{status} {code}: {message}"),
        }
    }
}

impl std::error::Error for Error {}
