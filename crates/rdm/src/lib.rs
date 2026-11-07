//! Recursive Decision Models.
//!
//! An RLM lets a language model act on a state it cannot hold all at once, and call itself on
//! pieces of it. An RDM is the same idea for a model that returns decisions instead of text: a
//! System One model, which is given a state and typed questions and answers each with a
//! decision and a probability. A decision model cannot write, so it never makes a state. It
//! decides, and what it decides selects what makes the next state: code, a generative model,
//! another module.
//!
//! ```text
//! call(request):                         a request is a state and typed questions
//!     answers <- predict(request)        asked once, and never again
//!     for each tool the answers select:  the tools run at once
//!         observation <- tool(...)       inside a tool, calling the module is a sub-call, one level down
//!         if an output pointed at it:    the same question again, about the state as it is now
//!     return answers, observations
//! ```
//!
//! An output made with [`Signature::tool`] points: the module's tools are a map of names, each
//! with a description for the model and a thing for the code to do, and the map can change as
//! it runs.
//!
//! References: "Recursive Decision Models" (Weitekamp, 2026),
//! <https://raw.works/recursive-decision-models/>, and "Recursive Language Models" (Zhang,
//! Kraska, Khattab, 2025), <https://arxiv.org/abs/2512.24601>.
//!
//! The pieces:
//!
//! - a [`Signature`] says what a request shows and asks: inputs, and outputs that are a yes
//!   or no, a choice, a score, or a pointer at one of the module's tools;
//! - [`Inputs`] are the state: plain values, or live ones ([`Shared`]) that every request
//!   shows as they are then;
//! - a [`Tool`] is a name, a description and a function of the [`Call`] that selected it;
//! - [`Rdm`] is the module: it asks, lets the answers select tools, and keeps to its limits;
//! - an [`Arbiter`] settles the tools a request selected as they come back: which results it
//!   comes to, which tools are still wanted, which to start, so that what might be done at
//!   once becomes one thing done after another ([`arbitrate`]);
//! - a [`Prediction`] is what a call returns: its answers, what its tools came to, why it
//!   ended, and for the call that started the run the trajectory, an [`Entry`] per request;
//! - a [`record::Record`] is handed the run as it happens, an [`record::Event`] at a time, for
//!   a host to keep and to replay ([`record::replay`]).
//!
//! Recursion is a tool that asks the module again, through the call it is given:
//!
//! ```
//! use rdm::systemone::Answer;
//! use rdm::{Rdm, Scripted, Signature, Tool, inputs};
//!
//! let search = Signature::new("Search a text for the answer to a question.")
//!     .input("text")
//!     .input("question")
//!     .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");
//!
//! // Runs on a yes. Asks the same question about each half, one level down.
//! let look = Tool::new("look", "Look in each half of the text.", |call| {
//!     let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
//!     let lines: Vec<&str> = text.lines().collect();
//!     if lines.len() < 2 {
//!         return Ok(vec![text]); // one line, and the model said yes to it
//!     }
//!     let mut found = Vec::new();
//!     for half in lines.chunks(lines.len().div_ceil(2)) {
//!         let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
//!         found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
//!     }
//!     Ok(found)
//! });
//!
//! // A decision model that says yes to a text that has the answer in it.
//! let model = Scripted::new(|request| {
//!     let text = request.state["inputs"]["text"].as_str().unwrap_or_default();
//!     [("look", Answer::noul(if text.contains("nested") { 0.9 } else { 0.1 }))]
//! });
//!
//! let zen = "Beautiful is better than ugly.\nSimple is better than complex.\nFlat is better than nested.\nSparse is better than dense.";
//! let result = Rdm::new(search).tool(look).max_depth(6).model(model).call(inputs! { "text" => zen, "question" => "Is flat better than nested?" }).unwrap();
//! assert_eq!(result.observed::<Vec<String>>("look").unwrap_or_default(), ["Flat is better than nested."]);
//! assert_eq!(result.trajectory.len(), 5);
//! ```
//!
//! # Using it
//!
//! The crates are not published: a program depends on them by path, from wherever the
//! repository is.
//!
//! ```toml
//! [dependencies]
//! rdm = { path = "rdm-rust/crates/rdm" }                # with `default-features = false`, the core alone
//! one-system = { path = "rdm-rust/crates/one-system" }  # for a real decision model: `one_system::Client`
//! ```
//!
//! The program above needs the first line and no more. [`systemone`], the protocol, is this
//! crate's own dependency and is given again here as `rdm::systemone`, so a program names it
//! through `rdm` and not in its manifest; [`Scripted`], [`DecisionModel`] and [`Json`] are at
//! the root as well. `one-system` is another crate, which `rdm` does not depend on: it is how a
//! program reaches hosted Jev, OpenRouter, a server on the machine, or a gateway among them.

#![warn(missing_docs)]

pub mod arbitrate;
mod call;
mod error;
mod inputs;
mod module;
mod observation;
mod parallel;
mod prediction;
pub mod record;
mod signature;
mod tool;

#[cfg(feature = "filter")]
pub mod filter;
#[cfg(feature = "fit")]
pub mod fit;
#[cfg(feature = "lm")]
pub mod lm;
#[cfg(feature = "near")]
pub mod near;
#[cfg(feature = "reanchor")]
pub mod reanchor;
#[cfg(feature = "repl")]
pub mod repl;
#[cfg(feature = "rlm")]
pub mod rlm;
#[cfg(all(feature = "lm", target_os = "wasi", target_env = "p2"))]
mod wasi;

pub use arbitrate::{Arbiter, Arbitrate, Arbitration};
pub use call::{Ask, Call, Cancel};
pub use error::{Error, Result};
pub use inputs::{Input, Inputs, Shared};
pub use module::{DEEPEST, Fit, Rdm};
pub use observation::as_observation;
pub use prediction::{Entry, Prediction, Reason, Receipt};
pub use signature::{FINISH, FINISH_DESCRIPTION, IMAGE, InputField, NONE, OBSERVATION, Opt, Output, RESERVED, Signature, Space};
pub use tool::{Source, Tool, Tools};

/// The protocol a decision model speaks, and how a decision is made from what it answered.
pub use systemone;
pub use systemone::{Decision, DecisionModel, Json, Params, Scripted, image};

/// The example in the repository's README is run with the other examples.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadMe;
