//! A gateway for decision models.
//!
//! A decision model is given a state and typed questions about it, and answers each with a
//! probability: a noul (yes or no), a choice (one of several) or a score (a level on a
//! rubric). That is the System One protocol, and the crate [`systemone`] is it. This crate is
//! what stands between a program and the models that speak it:
//!
//! - [`Client`] is one System One server over HTTP: hosted TypeSafe, OpenRouter, a local
//!   llama.cpp serving Laya, or another gateway. What carries its requests is a [`Transport`]:
//!   this crate's own network, or, in a WebAssembly module, the host's.
//! - [`Gateway`] is several decision models behind one. It sends each request to one of its
//!   backends: the one the request names, or the one it works out is right, by what each
//!   backend can take, by what each would rather not, and, when more than one is left, by
//!   asking a model.
//! - [`Registry`] is a gateway written down as a JSON file, in the format of the Go gateway
//!   this is a variation of ([rawwerks/one-system](https://github.com/rawwerks/one-system)).
//! - `Server` (the `server` feature, on by default) is a gateway served over HTTP, and the
//!   binary `one-system` is that, configured from the environment. A WebAssembly module can
//!   listen on no port, and a build for one has neither.
//!
//! A client and a gateway are each a [`systemone::DecisionModel`], and a gateway's backends
//! are anything that is one, so they nest: a gateway can route among clients, other gateways,
//! and functions ([`systemone::Scripted`]).
//!
//! ```
//! use one_system::{Backend, Capabilities, Gateway, Limits, Picked};
//! use systemone::{Answer, DecisionModel, Question, Request, Scripted};
//!
//! // Two models that are functions. In use they would be `Client::local(...)` and `Client::typesafe()`.
//! let laya = Scripted::new(|request| {
//!     // Asked to pick a backend, it picks the hosted one for anything about sums.
//!     if request.questions.contains_key("backend") {
//!         let sums = request.state.as_str().is_some_and(|state| state.contains('+'));
//!         return vec![("backend", Answer::chose(if sums { "hosted" } else { "local" }, ["hosted", "local"]))];
//!     }
//!     vec![("correct", Answer::noul(0.6))]
//! })
//! .named("laya-english")
//! .tokens(30);
//! let jev = Scripted::new(|_| [("correct", Answer::noul(0.99))]).named("jev-1.13.0").tokens(300);
//!
//! let gateway = Gateway::builder("demo")
//!     .backend(
//!         Backend::new("local", laya)
//!             .description("Short, plain English.")
//!             .capabilities(Capabilities::all().max_questions(16))
//!             .limits(Limits::default().max_characters(1200)),
//!     )
//!     .backend(Backend::new("hosted", jev).description("Arithmetic, long documents, other languages."))
//!     .selector("local")
//!     .fallback("local")
//!     .build()?;
//!
//! let request = Request::new("17 + 25 = 42").question("correct", Question::noul("Is the arithmetic right?"));
//! let routed = gateway.route(&request);
//! let response = routed.result?;
//! assert_eq!(response.model, "jev-1.13.0");
//! assert_eq!(routed.route.picked, Some(Picked::Selected { confidence: 1.0 }));
//! // What the selector used and what the backend used, together.
//! assert_eq!(response.usage.input_tokens, 330);
//!
//! // The gateway is a decision model, and a backend can be named.
//! assert_eq!(gateway.decide(&request.clone().model("local"))?.model, "laya-english");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What a gateway does not do
//!
//! It asks a backend once. If the backend fails, the caller gets that failure as it is: the
//! gateway does not ask again and does not try another backend, because a request asked twice
//! is paid for twice and may be shown to two operators. Retrying is the caller's choice, and
//! [`Client::retry`] is there for a client used on its own.
//!
//! Limits are routing hints and not a privacy boundary. A request only one backend can take
//! goes to that backend, whoever runs it.

#![warn(missing_docs)]

mod client;
mod error;
mod gateway;
#[cfg(not(target_family = "wasm"))]
mod http;
mod letters;
#[cfg(not(target_family = "wasm"))]
mod proxy;
mod registry;
mod schema;
#[cfg(all(feature = "server", not(target_family = "wasm")))]
mod server;
mod shape;
mod transport;
#[cfg(all(target_os = "wasi", target_env = "p2"))]
mod wasi;

pub use client::{Client, Retry};
pub use error::ConfigError;
pub use gateway::{
    Backend, Gateway, GatewayBuilder, Picked, Route, Routed, Rule, SELECTOR_INSTRUCTIONS, SELECTOR_QUESTION, Selection, Stage,
};
pub use registry::{Entry, MAX_CONFIG_BYTES, Registry};
pub use schema::{MAX_BODY_BYTES, check_answers, parse_request, validate};
#[cfg(all(feature = "server", not(target_family = "wasm")))]
pub use server::{Running, Server};
pub use shape::{Capabilities, Limits, Summary};
pub use transport::{HttpRequest, HttpResponse, Transport, TransportError};

/// The examples in the README are run with the other examples.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadMe;
