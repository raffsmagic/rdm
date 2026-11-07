//! Language models through lm15: one that writes as a tool, and one that answers decision questions.
//!
//! A decision model cannot write, so a module that needs something written has a language
//! model on its map as a tool. There are three things here:
//!
//! - [`Lm`] is a handle on a language model and the name it is asked for by. lm15 is the only
//!   way this crate talks to one, and it reads the keys: nothing here holds or prints one.
//! - [`Generate`] is a signature completed by a language model: named inputs in, named and
//!   typed outputs back, as one JSON object. It is the small part of DSPy's `Predict` that a
//!   tool needs, and it is a tool when it is asked to be ([`Generate::tool`]).
//! - [`LmDecisionModel`] puts a decision request to an ordinary language model and has it
//!   answer with evidence, so a module can run with no decision model at all.
//!
//! ```
//! use rdm::lm::{Generate, Lm, ScriptedLm};
//! use rdm::{Rdm, Signature, inputs};
//! use systemone::{Answer, Scripted};
//!
//! // A language model that is a function, for a test or an example with no network.
//! let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"reply": "It will stay dry until the evening."}"#.to_owned()), "scripted");
//! let reply = Generate::parse("task -> reply").unwrap().lm(lm);
//!
//! // The decision model decides, and the language model writes when it is selected.
//! let route = Signature::new("Route a request.").input("task").noul("chat", "Is `inputs.task` only a question to answer in conversation?");
//! let rdm = Rdm::new(route).tool(reply.tool("chat", "Answer in conversation.")).model(Scripted::new(|_| [("chat", Answer::noul(0.9))]));
//! let result = rdm.call(inputs! { "task" => "Will it rain?" }).unwrap();
//! assert_eq!(result.observations["chat"]["reply"], "It will stay dry until the evening.");
//! ```

use std::fmt;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use lm15::{ErrorClass, FinishReason, LanguageModel, Lm15Error, Message, Part};
use serde::de::DeserializeOwned;
use serde_json::{Map, json};
use systemone::{Answer, DecisionModel, Json, Kind, Question, Usage};

use crate::call::Call;
use crate::error::{Error, Result, listed, sorted};
use crate::inputs::Inputs;
use crate::observation::as_observation;
use crate::signature::{InputField, fields, named};
use crate::tool::Tool;

/// The crate every language model is reached through: its requests, its router, its fakes.
pub use lm15;

/// A language model, and the name it is asked for by.
///
/// The name is a model string as lm15 routes one: `openrouter:anthropic/claude-sonnet-5.5`,
/// `anthropic:claude-...`, `openai-chat:gpt-...`, `ollama:qwen3:4b`. The key is lm15's to
/// find, in the environment (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`),
/// when the model is first asked. It is a handle: a clone asks the same model.
///
/// Asking waits for the answer, as everything in this crate does. It is lm15's `blocking`
/// that waits, on a runtime of lm15's own, and it panics where it is called from inside an
/// async runtime: call a module from a plain thread (`spawn_blocking`, under tokio).
///
/// The reply is streamed: asked for as server-sent events and read as it comes, so what has
/// been written is there to watch before the model is done ([`Lm::watch`]), and a branch that
/// arbitration stopped stops the request in the middle of the reply, not after it. What is
/// returned is the same reply, whole. [`Lm::streaming`] turns it off.
///
/// In WebAssembly lm15 is the wire alone, with no network of its own, and the host carries
/// each request. A module built for WASI 0.2 (`wasm32-wasip2`) asks through the host's
/// `wasi:http`, as [`Lm::new`] and [`Lm::local`] set it up. Any other module is given its
/// transport ([`Lm::with_config`], and lm15's `RouterConfig::transport`): one that answers
/// before it returns, as an import of the module that blocks or that the engine suspends
/// the module for. There is no runtime to wait on.
///
/// ```
/// use rdm::lm::Lm;
///
/// // Nothing is sent, and no key is read, until the model is asked.
/// let lm = Lm::new("openrouter:anthropic/claude-sonnet-5.5").unwrap().temperature(0.0).max_tokens(2000);
/// assert_eq!(lm.model(), "openrouter:anthropic/claude-sonnet-5.5");
/// // A string that routes nowhere is refused when the handle is made, not at the first request.
/// assert!(Lm::new("openrouter/anthropic/claude-sonnet-5.5").unwrap_err().is_usage());
/// ```
#[derive(Clone)]
pub struct Lm {
    model: Arc<dyn LanguageModel>,
    name: Arc<str>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    retries: u32,
    /// Which schemas go with a request for JSON: [`EVERY`], [`PLAIN`] or [`NO`]. A wire that
    /// refused one is not offered one like it again, by this handle or a clone of it.
    schemas: Arc<AtomicU8>,
    /// Whether the model is shown the pictures of a decision request ([`Lm::images`]).
    images: bool,
    /// Whether the reply is asked for as a stream ([`Lm::streaming`]).
    streams: bool,
    /// Whether the server answered a request for a stream with the reply whole, as a server that
    /// does not stream does. It is not asked for one again, by this handle or a clone of it.
    unstreamed: Arc<std::sync::atomic::AtomicBool>,
    /// What is shown the reply as it comes ([`Lm::watch`]).
    watch: Option<Arc<Watch>>,
    /// The branch the request is asked for, whose stop stops it ([`Lm::under`]).
    stop: Option<Arc<crate::arbitrate::Stop>>,
}

/// What watches a reply as it comes: shown what has been written so far, it says whether to go on.
type Watch = dyn Fn(&Partial) -> ControlFlow<()> + Send + Sync;

/// What a language model has written so far, while it is still writing: what [`Lm::watch`] is
/// shown each time more of the reply comes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Partial {
    /// The reply so far.
    pub text: String,
    /// What the model thought so far, where the provider shows it.
    pub thinking: String,
    /// How long after it was asked the first of the reply or of its thinking came.
    pub first: Option<Duration>,
}

impl Lm {
    /// A model by its string, with its key read from the environment by lm15.
    ///
    /// A string no provider answers to is a usage error here, before anything is asked.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::with_config(model, router())
    }

    /// A model by its string, on a router set up by hand: a key given outright, a provider at
    /// another address, a provider lm15 does not know.
    pub fn with_config(model: impl Into<String>, config: lm15::RouterConfig) -> Result<Self> {
        let model = model.into();
        let refused = |error: Lm15Error| Error::Usage(format!("No language model is reached as '{model}': {}", said(&error)));
        let router = lm15::LMRouter::with_config(config).map_err(refused)?;
        router.resolve(&model).map_err(refused)?;
        Ok(Self::shared(Arc::new(router), model))
    }

    /// A model on a server that speaks OpenAI's Chat Completions, at an address of its own:
    /// llama.cpp's `llama-server`, LM Studio, vLLM. The address is the root the server answers
    /// under, `/v1` included: `http://localhost:8080/v1`. No key is sent that means anything.
    ///
    /// ```
    /// use rdm::lm::Lm;
    ///
    /// let lm = Lm::local("http://localhost:8080/v1", "qwen3-4b").unwrap();
    /// assert_eq!(lm.model(), "local:qwen3-4b");
    /// ```
    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        let model = if model.starts_with("local:") {
            model
        } else {
            format!("local:{model}")
        };
        Self::with_config(model, local(&base_url.into())?)
    }

    /// Any language model lm15 can ask, under the name to ask it for: a router of one's own,
    /// one provider's adapter, a [`ScriptedLm`] or lm15's `FakeLM` in a test.
    pub fn of(model: impl LanguageModel + 'static, name: impl Into<String>) -> Self {
        Self::shared(Arc::new(model), name)
    }

    /// As [`Lm::of`], for a model that is already shared: a test keeps its own handle on a
    /// fake to read what it was asked.
    pub fn shared(model: Arc<dyn LanguageModel>, name: impl Into<String>) -> Self {
        Self {
            model,
            name: name.into().into(),
            temperature: None,
            max_tokens: None,
            retries: 2,
            schemas: Arc::new(AtomicU8::new(EVERY)),
            images: false,
            // A module in WebAssembly is handed the reply whole by its host, so it asks for it whole.
            streams: cfg!(not(target_family = "wasm")),
            unstreamed: Arc::default(),
            watch: None,
            stop: None,
        }
    }

    /// The same model at another temperature. None is set unless this is: the provider's own.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// The same model with a limit on the tokens of one reply. None is set unless this is.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// How many more times a request is sent when the failure is one that may pass: a rate
    /// limit, a server's own error, a connection that dropped. Two, unless this says otherwise;
    /// the wait doubles from half a second, or is what the server asked for. A WebAssembly
    /// module with no system under it has no clock to wait by, and sends again at once.
    pub fn retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    /// Whether a JSON Schema is sent with a request for JSON, for a wire to hold the reply to.
    /// It is, unless this says otherwise. The shape is described in the prompt either way, and
    /// the reply is read and checked here either way.
    ///
    /// Left on, the handle finds out for itself: a request a server refused with a schema is
    /// sent again without one, and a schema like the one refused is not offered again.
    pub fn json_schema(mut self, sent: bool) -> Self {
        self.schemas = Arc::new(AtomicU8::new(if sent { EVERY } else { NO }));
        self
    }

    /// Whether the model can be shown pictures. It is not, unless this says so: then a
    /// decision request with pictures in it ([`crate::Signature::image`]), or a [`Generate`]
    /// given one, is refused before anything is sent, as a client of a server that does not
    /// read pictures refuses them. [`LmDecisionModel::leave_out_images`] and
    /// [`Generate::leave_out_images`] answer without them instead, and say so to the model.
    /// Shown them, the model is given each as an image of lm15's, after the inputs, in the
    /// order the inputs are declared, and each input shows where its picture is: `<image: name>`.
    ///
    /// ```
    /// use rdm::lm::{Lm, ScriptedLm};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| String::new()), "scripted").images(true);
    /// assert!(lm.sees_images());
    /// ```
    pub fn images(mut self, shown: bool) -> Self {
        self.images = shown;
        self
    }

    /// Whether the model is shown pictures ([`Lm::images`]).
    pub fn sees_images(&self) -> bool {
        self.images
    }

    /// Whether the reply is asked for as a stream and read as it comes. It is, unless this says
    /// otherwise, except in WebAssembly, where the host hands the module the reply whole. Asked
    /// for whole, the reply is watched once, when it is all there.
    pub fn streaming(mut self, streams: bool) -> Self {
        self.streams = streams;
        self.unstreamed = Arc::default();
        self
    }

    /// The same model, with what is written shown to `watch` as it comes: the reply and the
    /// model's thinking so far, each time more of either comes. What it returns says whether to
    /// go on: `Break` stops the request where it is, and the call fails as stopped
    /// ([`Error::Stopped`]). It is the handle's, and its clones': one watch is one clone.
    ///
    /// ```
    /// use std::ops::ControlFlow;
    /// use std::sync::{Arc, Mutex};
    ///
    /// use rdm::lm::{Lm, ScriptedLm};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| "It will rain.".to_owned()), "scripted");
    /// let seen = Arc::new(Mutex::new(Vec::new()));
    /// let watching = seen.clone();
    /// let said = lm
    ///     .watch(move |partial| {
    ///         watching.lock().unwrap().push(partial.text.clone());
    ///         ControlFlow::Continue(())
    ///     })
    ///     .complete("", "Will it rain?")
    ///     .unwrap();
    /// assert_eq!(said.text, "It will rain.");
    /// assert_eq!(seen.lock().unwrap().last().unwrap(), "It will rain.");
    ///
    /// // Stopped after the first of the reply, the call is stopped, and not a failure of the model.
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| "It will rain.".to_owned()), "scripted");
    /// let stopped = lm.watch(|_| ControlFlow::Break(())).complete("", "Will it rain?").unwrap_err();
    /// assert!(stopped.is_stopped());
    /// ```
    pub fn watch(mut self, watch: impl Fn(&Partial) -> ControlFlow<()> + Send + Sync + 'static) -> Self {
        self.watch = Some(Arc::new(watch));
        self
    }

    /// The same model, asked for a branch: a stop of the branch stops a request in the middle of its reply.
    pub(crate) fn under(&self, call: &Call) -> Self {
        let mut lm = self.clone();
        lm.stop = call.active.stop.clone();
        lm
    }

    /// The name the model is asked for by.
    pub fn model(&self) -> &str {
        &self.name
    }

    /// This model as a decision model: [`LmDecisionModel::new`].
    pub fn decision_model(&self) -> LmDecisionModel {
        LmDecisionModel::new(self.clone())
    }

    /// Ask for text: a system prompt, which may be empty, and one message.
    ///
    /// ```
    /// use rdm::lm::{Lm, ScriptedLm};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, prompt| format!("You said: {prompt}")).tokens(7, 4), "scripted");
    /// let said = lm.complete("Repeat what you are told.", "hello").unwrap();
    /// assert_eq!((said.text.as_str(), said.tokens()), ("You said: hello", 11));
    /// ```
    pub fn complete(&self, system: &str, prompt: &str) -> Result<Completion> {
        self.completing(system, prompt).0
    }

    /// [`Lm::complete`], and the time the model was asked as a run's `wrote` event keeps it:
    /// none where the request could not be made.
    pub(crate) fn completing(&self, system: &str, prompt: &str) -> (Result<Completion>, Option<Wrote>) {
        let user = match message(Message::user(prompt)) {
            Ok(user) => user,
            Err(error) => return (Err(error), None),
        };
        let request = self.asking(&self.name, system, vec![user], None);
        let sent = self.send(&request);
        let wrote = Wrote {
            model: self.name.to_string(),
            response: sent.as_ref().ok().cloned(),
            error: sent.as_ref().err().map(ToString::to_string),
            request,
        };
        let completed = sent.map_err(|error| self.failed(&Failed::Lm(error))).map(|response| Completion {
            text: written(&response),
            usage: used(&response.usage),
            model: served(&response, &self.name),
        });
        (completed, Some(wrote))
    }

    /// Send a request as lm15 takes one, for what the rest of this does not cover: tools,
    /// images, several messages. A model, a temperature and a limit the request does not name
    /// are this handle's.
    pub fn request(&self, mut request: lm15::Request) -> Result<lm15::Response> {
        if request.model.is_empty() {
            request.model = self.name.to_string();
        }
        request.config.temperature = request.config.temperature.or(self.temperature);
        request.config.max_tokens = request.config.max_tokens.or(self.max_tokens);
        self.send(&request).map_err(|error| self.failed(&Failed::Lm(error)))
    }

    /// A request for this model: a system prompt, the messages so far, and the schema of the reply if one goes with it.
    fn asking(&self, model: &str, system: &str, messages: Vec<Message>, format: Option<&Format>) -> lm15::Request {
        let mut config = lm15::Config {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            ..Default::default()
        };
        let offered = self.schemas.load(Ordering::Relaxed);
        if let Some(format) = format.filter(|format| offered == EVERY || (offered == PLAIN && format.plain)) {
            let shape = json!({"type": "json_schema", "name": format.name, "schema": format.schema, "strict": format.strict});
            config.response_format = shape.as_object().cloned();
        }
        lm15::Request {
            model: model.to_owned(),
            system: (!system.is_empty()).then(|| system.into()),
            messages,
            config,
            ..Default::default()
        }
    }

    /// Send, and send again while the failure is one that may pass. A reply that failed after
    /// some of it was shown to the watch is not asked for again: what was shown would be shown twice.
    #[allow(clippy::result_large_err)]
    fn send(&self, request: &lm15::Request) -> std::result::Result<lm15::Response, Box<Lm15Error>> {
        let mut again = 0;
        loop {
            let began = clock();
            let mut partial = Partial::default();
            let streams = self.streams && !self.unstreamed.load(Ordering::Relaxed);
            let answered = if streams {
                waited(self.reading(request, began, &mut partial))
            } else {
                waited(self.model.complete(request)).map(|answered| self.whole(answered, began, &mut partial))
            };
            let Some(answered) = answered else {
                return Err(Box::new(unready()));
            };
            let shown = !partial.text.is_empty() || !partial.thinking.is_empty();
            match answered {
                Ok(response) => return Ok(response),
                // A server that does not stream: one that answered a request for a stream with the
                // reply whole (a stream that ended before anything of it came, and with no end), or
                // that refused it for the stream. It is asked for the reply whole, from now on.
                Err(error) if streams && !shown && unstreamed(&error) => {
                    self.unstreamed.store(true, Ordering::Relaxed);
                }
                Err(error)
                    if again < self.retries
                        && error.is_retryable()
                        && error.provider_code() != Some(STOPPED)
                        && !(shown && self.watch.is_some()) =>
                {
                    pause(wait(again, error.retry_after()));
                    again += 1;
                }
                Err(error) => return Err(Box::new(error)),
            }
        }
    }

    /// Read a reply as it streams, showing the watch each piece of it, and stopping where the
    /// watch or the branch says to. The reply, whole, once the stream has ended.
    // The error is lm15's, as its streams give it.
    #[allow(clippy::result_large_err)]
    async fn reading(
        &self,
        request: &lm15::Request,
        began: Option<Instant>,
        partial: &mut Partial,
    ) -> std::result::Result<lm15::Response, Lm15Error> {
        use futures_core::Stream;
        let mut stream = lm15::ResponseStream::new(self.model.stream(request), request);
        loop {
            if self.stop.as_ref().is_some_and(|stop| stop.is_set()) {
                return Err(stopped_by("the branch it was asked for was stopped"));
            }
            let Some(event) = std::future::poll_fn(|context| std::pin::Pin::new(&mut stream).poll_next(context)).await else {
                break;
            };
            let lm15::StreamEvent::Delta(lm15::StreamDeltaEvent { delta }) = event? else {
                continue;
            };
            let (more, grows) = match &delta {
                lm15::Delta::Text(text) => (text.text.as_str(), &mut partial.text),
                lm15::Delta::Thinking(thinking) => (thinking.text.as_str(), &mut partial.thinking),
                _ => continue,
            };
            if more.is_empty() {
                continue;
            }
            grows.push_str(more);
            partial.first = partial.first.or_else(|| began.map(|began| began.elapsed()));
            if self.watch.as_ref().is_some_and(|watch| watch(partial).is_break()) {
                return Err(stopped_by("its watch stopped it"));
            }
        }
        stream.response().await
    }

    /// A reply asked for whole, shown to the watch once, as all of it came at once.
    #[allow(clippy::result_large_err)]
    fn whole(
        &self,
        answered: std::result::Result<lm15::Response, Lm15Error>,
        began: Option<Instant>,
        partial: &mut Partial,
    ) -> std::result::Result<lm15::Response, Lm15Error> {
        let response = answered?;
        partial.text = written(&response);
        partial.first = began.map(|began| began.elapsed());
        if self.watch.as_ref().is_some_and(|watch| watch(partial).is_break()) {
            return Err(stopped_by("its watch stopped it"));
        }
        Ok(response)
    }

    /// Send a request for JSON. A wire that cannot hold a reply to a schema is asked without
    /// one: the prompt says the shape, and the reply is checked here.
    fn send_for_json(&self, request: &mut lm15::Request, format: &Format) -> std::result::Result<lm15::Response, Box<Lm15Error>> {
        match self.send(request) {
            Err(error) if request.config.response_format.is_some() && unconstrained(&error) => {
                request.config.response_format = None;
                let response = self.send(request)?;
                // It was the schema: asked without it, the model answered. A wire with no place
                // for one, or a server that refused a plain one, is not offered another. A server
                // that refused one with names of any kind in it is still offered the plain ones.
                let plain_too = format.plain || error.is_a(ErrorClass::CapabilityError);
                self.schemas.fetch_max(if plain_too { NO } else { PLAIN }, Ordering::Relaxed);
                Ok(response)
            }
            other => other,
        }
    }

    /// What went wrong, as the error a call returns. A model that could not be asked ends the
    /// run; one that wrote something else than it was asked for has failed as a tool fails.
    fn failed(&self, why: &Failed) -> Error {
        match why {
            Failed::Lm(error) if error.provider_code() == Some(STOPPED) => Error::Stopped(format!(
                "The language model '{}' was stopped while it wrote: {}",
                self.name,
                error.message()
            )),
            Failed::Lm(error) => Error::Model(model_error(&self.name, error)),
            Failed::Reply(mistake) => Error::Tool(format!(
                "The language model '{}' did not write what it was asked for: {mistake}",
                self.name
            )),
            Failed::Request(why) => Error::Usage(format!("The language model '{}' cannot be asked that: {why}", self.name)),
        }
    }
}

/// Every schema is offered.
const EVERY: u8 = 0;
/// Only a plain schema is offered: see [`Format::plain`].
const PLAIN: u8 = 1;
/// No schema is offered.
const NO: u8 = 2;

/// The model's name and what it is set to. Never a key: the handle has none.
impl fmt::Debug for Lm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lm")
            .field("model", &self.name)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

/// What a language model wrote, and what it took.
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    /// The text of the reply.
    pub text: String,
    /// The tokens the request used, as the provider counted them.
    pub usage: Usage,
    /// The model that answered, as the provider named it.
    pub model: String,
}

impl Completion {
    /// The tokens in and out together: what a tool reports with [`Call::spend`].
    pub fn tokens(&self) -> u64 {
        self.usage.total()
    }
}

/// What a scripted model writes: a reply, from the system prompt and the last message.
type Writes = dyn Fn(&str, &str) -> String + Send + Sync;

/// A language model that is a function, from the system prompt and the last message to the
/// text of a reply. For tests, and for examples that need no network and no key.
///
/// It keeps every request it was asked, as lm15 has them. Because it is a function and not a
/// list of replies, it answers the same whichever of two tools running at once asks first.
///
/// ```
/// use std::sync::Arc;
///
/// use rdm::lm::{Lm, ScriptedLm};
///
/// let scripted = Arc::new(ScriptedLm::new(|_system, prompt| prompt.to_uppercase()));
/// let lm = Lm::shared(scripted.clone(), "scripted");
/// assert_eq!(lm.complete("", "quiet").unwrap().text, "QUIET");
/// assert_eq!(scripted.requests()[0].model, "scripted");
/// ```
pub struct ScriptedLm {
    writes: Box<Writes>,
    tokens: (u64, u64),
    asked: Mutex<Vec<lm15::Request>>,
}

impl std::fmt::Debug for ScriptedLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let asked = self.asked.try_lock().map(|asked| asked.len()).ok();
        f.debug_struct("ScriptedLm")
            .field("tokens", &self.tokens)
            .field("asked", &asked)
            .finish_non_exhaustive()
    }
}

impl ScriptedLm {
    /// A model that replies with what the function returns, given the system prompt and the last message.
    pub fn new(writes: impl Fn(&str, &str) -> String + Send + Sync + 'static) -> Self {
        Self {
            writes: Box::new(writes),
            tokens: (0, 0),
            asked: Mutex::new(Vec::new()),
        }
    }

    /// The tokens it reports for every request: in, and out.
    pub fn tokens(mut self, input: u64, output: u64) -> Self {
        self.tokens = (input, output);
        self
    }

    /// Every request it has been asked, in the order they arrived.
    pub fn requests(&self) -> Vec<lm15::Request> {
        self.asked.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn answer(&self, request: &lm15::Request) -> lm15::Response {
        self.asked.lock().unwrap_or_else(PoisonError::into_inner).push(request.clone());
        let system = match &request.system {
            Some(lm15::SystemContent::Text(text)) => text.as_str(),
            _ => "",
        };
        let prompt = request.messages.last().and_then(Message::text).unwrap_or_default();
        let mut response = reply((self.writes)(system, &prompt), self.tokens.0, self.tokens.1);
        response.model = "scripted".into();
        response
    }
}

impl LanguageModel for ScriptedLm {
    fn complete<'a>(
        &'a self,
        request: &'a lm15::Request,
    ) -> std::pin::Pin<Box<dyn Future<Output = std::result::Result<lm15::Response, Lm15Error>> + Send + 'a>> {
        Box::pin(async move { Ok(self.answer(request)) })
    }

    fn stream(&self, request: &lm15::Request) -> lm15::testing::LanguageModelStream {
        replayed(Ok(self.answer(request)), request)
    }
}

/// A reply as a language model gives one, for lm15's `FakeLM`: its text, and the tokens in
/// and out. The model it says answered is `fake`.
///
/// ```
/// use rdm::lm::{Lm, lm15::testing::FakeLM, reply};
///
/// let lm = Lm::of(FakeLM::new([reply("An owl.", 9, 3)]), "any-name");
/// assert_eq!(lm.complete("", "What eats mice at night?").unwrap().tokens(), 12);
/// ```
pub fn reply(text: impl Into<String>, input_tokens: u64, output_tokens: u64) -> lm15::Response {
    let text = text.into();
    let parts = vec![Part::text(if text.is_empty() { " ".to_owned() } else { text })];
    lm15::Response {
        id: None,
        model: "fake".into(),
        message: Message {
            role: lm15::Role::Assistant,
            parts,
            continuation: Vec::new(),
        },
        finish_reason: FinishReason::Stop,
        usage: lm15::Usage {
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            ..Default::default()
        },
        logprobs: None,
        logprobs_complete: true,
        adaptations: Vec::new(),
        provider_data: None,
    }
}

/// What a message shows under an input's name: the text between `[[ ## name ## ]]` and the
/// next input, or the sentence that closes the message. For a [`ScriptedLm`] that answers
/// from what it was shown, which is all an example with no network has to go on.
///
/// It reads a message as [`Generate`] and [`LmDecisionModel`] write one, and looks no
/// further than that: an input whose own text holds a line like `[[ ## ... ## ]]` after a
/// blank one is read up to there.
///
/// ```
/// use rdm::inputs;
/// use rdm::lm::{Generate, Lm, ScriptedLm, input_of};
/// use serde_json::json;
///
/// // A language model that answers with the first line of the passage it is shown.
/// let lm = Lm::of(
///     ScriptedLm::new(|_system, prompt| {
///         let passage = input_of(prompt, "passage").unwrap_or_default();
///         json!({"answer": passage.lines().next().unwrap_or_default()}).to_string()
///     }),
///     "scripted",
/// );
/// let answer = Generate::parse("passage, question -> answer").unwrap().lm(lm);
/// let written = answer.call(inputs! { "passage" => "London: 260 euros.\nParis: 240 euros.", "question" => "What is the limit in London?" }).unwrap();
/// assert_eq!(written["answer"], "London: 260 euros.");
/// ```
pub fn input_of<'a>(prompt: &'a str, name: &str) -> Option<&'a str> {
    let header = format!("[[ ## {name} ## ]]\n");
    let from = match prompt.strip_prefix(header.as_str()) {
        Some(_) => header.len(),
        None => prompt.find(&format!("\n\n{header}"))? + 2 + header.len(),
    };
    let rest = &prompt[from..];
    let until = rest.find("\n\n[[ ## ").or_else(|| rest.rfind(CLOSING))?;
    Some(&rest[..until])
}

/// How a message ends, after its inputs: the fields to answer with follow it.
const CLOSING: &str = "\n\nRespond with a JSON object in the following order of fields: ";

/// The type of a generated output: what the model is told to write, and what its reply is checked against.
///
/// ```
/// use rdm::lm::Type;
///
/// let parts = Type::parse("list[str]").unwrap();
/// assert_eq!(parts, Type::List(Box::new(Type::Str)));
/// assert_eq!(parts.to_string(), "list[str]");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    /// Text: `str`.
    Str,
    /// A whole number: `int`.
    Int,
    /// A number: `float`.
    Float,
    /// True or false: `bool`.
    Bool,
    /// An object of any keys: `dict`.
    Dict,
    /// A list of one type: `list[str]`. A bare `list` is a list of anything.
    List(Box<Type>),
    /// Any JSON value: `Any`.
    Any,
}

impl Type {
    /// A type as a signature writes it: `str`, `int`, `float`, `bool`, `dict`, `list[...]`, `Any`.
    pub fn parse(text: &str) -> Option<Type> {
        let text = text.trim();
        Some(match text {
            "str" => Type::Str,
            "int" => Type::Int,
            "float" => Type::Float,
            "bool" => Type::Bool,
            "dict" => Type::Dict,
            "list" => Type::List(Box::new(Type::Any)),
            "Any" => Type::Any,
            _ if text.starts_with("dict[") && text.ends_with(']') => Type::Dict,
            _ => Type::List(Box::new(Type::parse(text.strip_prefix("list[")?.strip_suffix(']')?)?)),
        })
    }

    /// A list of a type: `Type::list(Type::Str)` is `list[str]`.
    pub fn list(of: Type) -> Type {
        Type::List(Box::new(of))
    }

    /// The JSON Schema of a value of the type.
    pub fn schema(&self) -> Json {
        match self {
            Type::Str => json!({"type": "string"}),
            Type::Int => json!({"type": "integer"}),
            Type::Float => json!({"type": "number"}),
            Type::Bool => json!({"type": "boolean"}),
            Type::Dict => json!({"type": "object"}),
            Type::List(of) => json!({"type": "array", "items": of.schema()}),
            Type::Any => json!({}),
        }
    }

    /// Whether a schema of the type is one a wire can hold a reply to exactly: an object of any keys is not.
    fn closed(&self) -> bool {
        match self {
            Type::Dict | Type::Any => false,
            Type::List(of) => of.closed(),
            _ => true,
        }
    }

    /// A value as this type, or what it is instead.
    ///
    /// What can be read as the type without guessing is: a number where text was asked for is
    /// that number written out, a whole number written as `3.0` or `"3"` is 3, and a list or
    /// an object that arrived as a string of JSON is what the string holds.
    pub(crate) fn fit(&self, value: Json) -> std::result::Result<Json, String> {
        let instead = |value: &Json| Err(format!("it is {}", what(value)));
        match (self, value) {
            (Type::Any, value) => Ok(value),
            (Type::Str, Json::String(text)) => Ok(Json::String(text)),
            (Type::Str, value @ (Json::Number(_) | Json::Bool(_))) => Ok(Json::String(value.to_string())),
            (Type::Int, Json::Number(number)) if number.is_i64() || number.is_u64() => Ok(Json::Number(number)),
            (Type::Int, Json::Number(number)) => match number.as_f64() {
                Some(whole) if whole.fract() == 0.0 && whole.abs() < 9e15 => Ok(json!(whole as i64)),
                _ => Err("it is a number that is not whole".to_owned()),
            },
            (Type::Int, Json::String(text)) => text.trim().parse::<i64>().map(Json::from).or_else(|_| instead(&Json::String(text))),
            (Type::Float, Json::Number(number)) => Ok(Json::Number(number)),
            (Type::Float, Json::String(text)) => match text.trim().parse::<f64>().ok().and_then(serde_json::Number::from_f64) {
                Some(number) => Ok(Json::Number(number)),
                None => instead(&Json::String(text)),
            },
            (Type::Bool, Json::Bool(value)) => Ok(Json::Bool(value)),
            (Type::Bool, Json::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
                "true" => Ok(Json::Bool(true)),
                "false" => Ok(Json::Bool(false)),
                _ => instead(&Json::String(text)),
            },
            (Type::Dict, Json::Object(object)) => Ok(Json::Object(object)),
            (Type::List(of), Json::Array(items)) => {
                let fitted = items.into_iter().enumerate().map(|(position, item)| {
                    let item = of.fit(item).map_err(|problem| format!("item {position}: {problem}"))?;
                    Ok(item)
                });
                Ok(Json::Array(fitted.collect::<std::result::Result<_, String>>()?))
            }
            (Type::Dict | Type::List(_), Json::String(text)) => match serde_json::from_str::<Json>(&text) {
                Ok(held @ (Json::Object(_) | Json::Array(_))) => self.fit(held),
                _ => instead(&Json::String(text)),
            },
            (_, other) => instead(&other),
        }
    }
}

/// The type as a signature writes it, and as a model is told it.
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Str => f.write_str("str"),
            Type::Int => f.write_str("int"),
            Type::Float => f.write_str("float"),
            Type::Bool => f.write_str("bool"),
            Type::Dict => f.write_str("dict"),
            // A list of anything is `list`, as Python names it.
            Type::List(of) if **of == Type::Any => f.write_str("list"),
            Type::List(of) => write!(f, "list[{of}]"),
            Type::Any => f.write_str("Any"),
        }
    }
}

/// One output of a [`Generate`]: its type, and what it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputField {
    /// What the model is to write, and what its reply is checked against.
    pub kind: Type,
    /// What the output is. May be empty.
    pub description: String,
}

impl OutputField {
    /// An output's type and what the output is.
    pub fn new(kind: Type, description: impl Into<String>) -> Self {
        Self {
            kind,
            description: description.into(),
        }
    }
}

/// A signature completed by a language model: named inputs in, named and typed outputs back.
///
/// It is what the examples use DSPy's `Predict` for, and no more: one request, one JSON
/// object holding the outputs, each checked against its type, and the tokens it took. The
/// model is told the shape in the prompt and, where the wire can hold a reply to a schema,
/// by the schema as well. A reply that is not the outputs is shown back to the model once,
/// with what is wrong with it.
///
/// Its fields are its own. A [`crate::Signature`] is questions for a decision model: every
/// output of one is a yes or no, a choice or a score, with a probability, and none of that
/// is true of a text. So the two are apart, and a string that is a signature of both
/// (`"task -> reply"`) means text here and a yes or no there.
///
/// ```
/// use rdm::inputs;
/// use rdm::lm::{Generate, Lm, ScriptedLm};
///
/// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"parts": ["Turn the TV on.", "Switch the lights off."]}"#.to_owned()).tokens(120, 18), "scripted");
/// let write = Generate::parse("task -> parts: list[str]").unwrap().lm(lm);
///
/// let written = write.call(inputs! { "task" => "Turn the TV on and switch the lights off." }).unwrap();
/// let parts: Vec<String> = written.read("parts").unwrap();
/// assert_eq!(parts, ["Turn the TV on.", "Switch the lights off."]);
/// assert_eq!(written.tokens(), 138);
/// ```
///
/// Built in code, with instructions and what each field is:
///
/// ```
/// use rdm::lm::{Generate, Type};
///
/// let answer = Generate::new("Answer the question from the passage, in one sentence.")
///     .input_described("passage", "A passage of the handbook.")
///     .input("question")
///     .output_described("answer", "The answer, or that the passage does not say.")
///     .output_typed("certain", Type::Bool);
/// assert_eq!(answer.outputs()["certain"].kind, Type::Bool);
/// ```
#[derive(Clone, Debug)]
pub struct Generate {
    lm: Option<Lm>,
    instructions: String,
    inputs: IndexMap<String, InputField>,
    outputs: IndexMap<String, OutputField>,
    retries: usize,
    leave_out_images: bool,
}

impl Generate {
    /// A signature with its instructions and no fields yet.
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            lm: None,
            instructions: instructions.into(),
            inputs: IndexMap::new(),
            outputs: IndexMap::new(),
            retries: 1,
            leave_out_images: false,
        }
    }

    /// Whether an input that is a picture ([`crate::image`], as a request with a picture input
    /// shows one) is left out for a model that is not shown pictures ([`Lm::images`]), with
    /// the input showing `<image: name>` and the model told how many were left out. It is not,
    /// unless this says so: such an input is a usage error, and nothing is sent. A model that
    /// is shown pictures is given them in any case.
    ///
    /// Which inputs are pictures is read from their values here, since a signature written as
    /// `"window -> reply"` takes any value: a value that is a picture is never written into
    /// the prompt as its base64.
    ///
    /// ```
    /// use rdm::lm::{Generate, Lm, ScriptedLm};
    /// use rdm::{image, inputs};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, prompt| format!(r#"{{"reply": {:?}}}"#, prompt.lines().next().unwrap_or_default())), "scripted");
    /// let describe = Generate::parse("window -> reply").unwrap().lm(lm);
    /// let window = || inputs! { "window" => image("image/png", b"png") };
    /// assert!(describe.call(window()).unwrap_err().is_usage());
    /// let said = describe.leave_out_images(true).call(window()).unwrap();
    /// assert!(said.text("reply").unwrap().starts_with("1 picture of the inputs left out"));
    /// ```
    pub fn leave_out_images(mut self, leave_out: bool) -> Self {
        self.leave_out_images = leave_out;
        self
    }

    /// The string form: `"passage, question -> answer"`, `"task -> parts: list[str]"`.
    ///
    /// A field's type is `str` unless it says otherwise. An output is `str`, `int`, `float`,
    /// `bool`, `dict`, or a `list[...]` of one of them. The instructions are the ones DSPy
    /// gives a signature written this way: "Given the fields `task`, produce the fields `parts`."
    pub fn parse(text: &str) -> Result<Self> {
        let malformed = || {
            Error::Usage(format!(
                "A signature is written `inputs -> outputs`, each a list of names: got {text:?}."
            ))
        };
        let (inputs, outputs) = text.split_once("->").ok_or_else(malformed)?;
        if outputs.contains("->") {
            return Err(malformed());
        }
        let mut generate = Generate::new("");
        for field in fields(inputs) {
            let (name, type_name) = named(&field).ok_or_else(malformed)?;
            generate = generate.input_typed(name, type_name.unwrap_or("str"));
        }
        for field in fields(outputs) {
            let (name, type_name) = named(&field).ok_or_else(malformed)?;
            let kind = Type::parse(type_name.unwrap_or("str")).ok_or_else(|| {
                Error::Usage(format!(
                    "Output `{name}` is typed `{}`; a generated output is `str`, `int`, `float`, `bool`, `dict`, or a `list[...]` of one of them.",
                    type_name.unwrap_or_default()
                ))
            })?;
            generate = generate.output_typed(name, kind);
        }
        if generate.outputs.is_empty() {
            return Err(malformed());
        }
        let ticked = |names: Vec<&String>| names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ");
        generate.instructions = format!(
            "Given the fields {}, produce the fields {}.",
            ticked(generate.inputs.keys().collect()),
            ticked(generate.outputs.keys().collect())
        );
        Ok(generate)
    }

    /// The language model that completes it.
    pub fn lm(mut self, lm: Lm) -> Self {
        self.lm = Some(lm);
        self
    }

    /// How many times a reply that is not the outputs is shown back to the model, with what
    /// is wrong with it, for another. Once, unless this says otherwise.
    pub fn retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }

    /// The same fields under other instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// An input of type `str`, with no description.
    pub fn input(self, name: impl Into<String>) -> Self {
        self.input_with(name, "str", "")
    }

    /// An input of another type, as the model is told it: `dict`, `list[str]`, `int`. An input is shown, never checked.
    pub fn input_typed(self, name: impl Into<String>, type_name: impl Into<String>) -> Self {
        self.input_with(name, type_name, "")
    }

    /// An input of type `str`, with what it is.
    pub fn input_described(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.input_with(name, "str", description)
    }

    /// An input with its type and what it is.
    pub fn input_with(mut self, name: impl Into<String>, type_name: impl Into<String>, description: impl Into<String>) -> Self {
        self.inputs.insert(name.into(), InputField::new(type_name, description));
        self
    }

    /// An output of type `str`, with no description.
    pub fn output(self, name: impl Into<String>) -> Self {
        self.output_with(name, Type::Str, "")
    }

    /// An output of another type.
    pub fn output_typed(self, name: impl Into<String>, kind: Type) -> Self {
        self.output_with(name, kind, "")
    }

    /// An output of type `str`, with what it is.
    pub fn output_described(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.output_with(name, Type::Str, description)
    }

    /// An output with its type and what it is.
    pub fn output_with(mut self, name: impl Into<String>, kind: Type, description: impl Into<String>) -> Self {
        self.outputs.insert(name.into(), OutputField::new(kind, description));
        self
    }

    /// What the model is asked to do.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// The inputs, in the order they were declared.
    pub fn inputs(&self) -> &IndexMap<String, InputField> {
        &self.inputs
    }

    /// The outputs, in the order they were declared.
    pub fn outputs(&self) -> &IndexMap<String, OutputField> {
        &self.outputs
    }

    /// The language model that completes it, if it has been given one.
    pub fn language_model(&self) -> Option<&Lm> {
        self.lm.as_ref()
    }

    /// The JSON Schema of a reply: one object holding every output.
    pub fn schema(&self) -> Json {
        let properties: Map<String, Json> = self
            .outputs
            .iter()
            .map(|(name, field)| {
                let mut schema = field.kind.schema();
                if !field.description.is_empty() {
                    schema["description"] = field.description.as_str().into();
                }
                (name.clone(), schema)
            })
            .collect();
        object_schema(properties)
    }

    /// Complete the signature for these inputs: one request to the language model, and one
    /// more if its reply was not the outputs.
    ///
    /// The inputs are the signature's, every one and no other: anything else is a usage
    /// error. A language model that cannot be asked, or that fails, is [`Error::Model`]. One
    /// that answered and still did not write the outputs is [`Error::Tool`]: what it wrote is
    /// never how the module was used, and inside a tool that is the tool's observation.
    pub fn call(&self, inputs: Inputs) -> Result<Generated> {
        let mut spent = Usage::default();
        self.complete(&inputs.snapshot()?, &mut spent, &mut Vec::new(), None)
    }

    /// As [`Generate::call`], from inside a tool: the tokens it took are reported through the
    /// call, so they count against the run's limit and are recorded on the request as that
    /// tool's `tool_tokens`. What a failed attempt took is reported too.
    ///
    /// ```
    /// use rdm::lm::{Generate, Lm, ScriptedLm};
    /// use rdm::{Rdm, Signature, Tool, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"answer": "260 euros a night."}"#.to_owned()).tokens(40, 8), "scripted");
    /// let answer = Generate::parse("passage, question -> answer").unwrap().lm(lm);
    ///
    /// // Runs on a yes: the language model answers from the text the decision model said holds the answer.
    /// let look = Tool::new("look", "Answer from the text.", move |call| {
    ///     let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
    ///     let written = answer.call_in(call, inputs! { "passage" => text, "question" => question })?;
    ///     written.read::<String>("answer")
    /// });
    /// let search = Signature::new("Search a text.").input("text").input("question").noul("look", "Is the answer to `inputs.question` in `inputs.text`?");
    /// let rdm = Rdm::new(search).tool(look).model(Scripted::new(|_| [("look", Answer::noul(0.9))]));
    /// let result = rdm.call(inputs! { "text" => "London: 260 euros a night.", "question" => "What is the hotel limit?" }).unwrap();
    /// assert_eq!(result.observations["look"], "260 euros a night.");
    /// assert_eq!(result.trajectory[0].tool_tokens.as_ref().unwrap()["look"], 48);
    /// ```
    pub fn call_in(&self, call: &Call, inputs: Inputs) -> Result<Generated> {
        self.spending(call, &inputs.snapshot()?)
    }

    /// The signature as a tool. Selected, its inputs are taken by name from the request's
    /// answers, then from its inputs as the request showed them, as a module's are
    /// ([`Tool::module`]); a name that is neither is a usage error. Its observation is its
    /// outputs, an object by name, and its tokens are the request's `tool_tokens`.
    ///
    /// ```
    /// use rdm::lm::{Generate, Lm, ScriptedLm};
    /// use rdm::{Rdm, Signature, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"reply": "No rain is expected."}"#.to_owned()).tokens(30, 6), "scripted");
    /// let chat = Generate::parse("task -> reply").unwrap().lm(lm).tool("chat", "Answer in conversation.");
    ///
    /// let route = Signature::new("Route a request.").input("task").noul("chat", "Is `inputs.task` only a question?");
    /// let rdm = Rdm::new(route).tool(chat).model(Scripted::new(|_| [("chat", Answer::noul(0.9))]));
    /// let result = rdm.call(inputs! { "task" => "Will it rain?" }).unwrap();
    /// assert_eq!(result.observations["chat"], serde_json::json!({"reply": "No rain is expected."}));
    /// assert_eq!(result.trajectory[0].tool_tokens.as_ref().unwrap()["chat"], 36);
    /// ```
    pub fn tool(&self, name: impl Into<String>, description: impl Into<String>) -> Tool {
        let generate = self.clone();
        Tool::new(name, description, move |call| Ok(Json::Object(generate.call_from(call)?.outputs)))
    }

    /// As [`Generate::call_in`], with the inputs taken from the request the tool was selected
    /// by: each by name from the request's answers, then from its inputs as the request showed
    /// them. It is what [`Generate::tool`] does, for a tool that does something with the
    /// outputs before it returns: a name that is neither is a usage error, and the tokens are
    /// reported through the call.
    ///
    /// ```
    /// use rdm::lm::{Generate, Lm, ScriptedLm};
    /// use rdm::{Rdm, Signature, Tool, inputs};
    /// use serde::Deserialize;
    /// use systemone::{Answer, Scripted};
    ///
    /// #[derive(Deserialize)]
    /// struct Draft {
    ///     name: String,
    ///     code: String,
    /// }
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"name": "cool", "code": "def cool():\n    return cooler()"}"#.to_owned()), "scripted");
    /// let writer = Generate::new("Write one Python function that does what the task asks.").input("task").output("name").output("code").lm(lm);
    ///
    /// // `task` is an input of the request that selected the tool, so it is the writer's too.
    /// let write = Tool::new("write", "Have a new tool written.", move |call| {
    ///     let draft: Draft = writer.call_from(call)?.read_all()?;
    ///     Ok(format!("{}: {} lines", draft.name, draft.code.lines().count()))
    /// });
    /// let operate = Signature::new("Operate a smart home.").input("task").noul("write", "Does `inputs.task` need a tool that is not there?");
    /// let rdm = Rdm::new(operate).tool(write).model(Scripted::new(|_| [("write", Answer::noul(0.9))]));
    /// let result = rdm.call(inputs! { "task" => "Bring the heat down." }).unwrap();
    /// assert_eq!(result.observations["write"], "cool: 2 lines");
    /// ```
    pub fn call_from(&self, call: &Call) -> Result<Generated> {
        let (mut shown, mut missing) = (Map::new(), Vec::new());
        for field in self.inputs.keys() {
            match call.answer_value(field).or_else(|| call.shown().get(field).cloned()) {
                Some(value) => drop(shown.insert(field.clone(), value)),
                None => missing.push(field.clone()),
            }
        }
        if !missing.is_empty() {
            let who = call.tool().map_or("The fit".to_owned(), |name| format!("Tool '{name}'"));
            return Err(Error::Usage(format!(
                "{who} takes {}, which name no answer or input of the request. An input of a signature is taken by name from {}.",
                listed(&missing),
                sorted(call.names())
            )));
        }
        self.spending(call, &shown)
    }

    /// Complete, and report what it took through the call, whether or not it came to anything.
    /// Each time the model was asked is the run's `wrote` event.
    fn spending(&self, call: &Call, shown: &Map<String, Json>) -> Result<Generated> {
        // Nothing is written for a branch arbitration has stopped.
        call.go_on()?;
        let (mut spent, mut wrote) = (Usage::default(), Vec::new());
        let generated = self.complete(shown, &mut spent, &mut wrote, Some(call));
        call.spend(spent.total());
        for wrote in wrote {
            call.wrote(wrote)?;
        }
        generated
    }

    /// Complete for these inputs. Asked for a branch (`call`), a stop of the branch stops the
    /// request in the middle of its reply.
    fn complete(&self, shown: &Map<String, Json>, spent: &mut Usage, wrote: &mut Vec<Wrote>, call: Option<&Call>) -> Result<Generated> {
        self.check()?;
        let lm = self
            .lm
            .as_ref()
            .ok_or_else(|| Error::Usage("The signature has no language model to complete it: give it one with `lm(...)`.".to_owned()))?;
        let lm = &match call {
            Some(call) => lm.under(call),
            None => lm.clone(),
        };
        if shown.len() != self.inputs.len() || shown.keys().any(|name| !self.inputs.contains_key(name)) {
            return Err(Error::Usage(format!(
                "The call has inputs {}; its signature takes {}.",
                sorted(shown.keys()),
                sorted(self.inputs.keys())
            )));
        }
        // A picture is never written into the prompt as text: shown as itself, left out, or refused.
        let pictured: Vec<&String> = self
            .inputs
            .keys()
            .filter(|name| systemone::Image::of(&shown[*name]).is_some())
            .collect();
        let mut images = Vec::new();
        let mut referred;
        let shown = if pictured.is_empty() {
            shown
        } else {
            if !lm.images && !self.leave_out_images {
                return Err(Error::Usage(format!(
                    "{} {} a picture, and the language model '{}' is not shown pictures: show it them with `Lm::images(true)`, \
                     or leave them out with `Generate::leave_out_images(true)`.",
                    pictured.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", "),
                    if pictured.len() == 1 { "is" } else { "are each" },
                    lm.model()
                )));
            }
            referred = shown.clone();
            for name in &pictured {
                images.push(
                    referred
                        .insert((*name).clone(), Json::String(format!("<image: {name}>")))
                        .unwrap_or_default(),
                );
            }
            &referred
        };
        let schema = self.schema();
        let strict = self.outputs.values().all(|field| field.kind.closed());
        let format = Format {
            name: "outputs",
            strict,
            plain: strict && self.outputs.keys().all(|name| plain(name)),
            schema: schema.clone(),
        };
        let prompt = Prompt {
            system: system(
                &numbered(
                    self.inputs
                        .iter()
                        .map(|(name, field)| (name, field.type_name.clone(), field.description.as_str())),
                ),
                &numbered(
                    self.outputs
                        .iter()
                        .map(|(name, field)| (name, field.kind.to_string(), field.description.as_str())),
                ),
                self.inputs.keys(),
                &schema,
                &self.instructions,
            ),
            // In the order the signature declares them, whatever order they were given in.
            user: user(self.inputs.keys().map(|name| (name, &shown[name])), self.outputs.keys()),
            images: Vec::new(),
        };
        let mut prompt = prompt;
        if !images.is_empty() {
            if lm.images {
                prompt.images = pictures(&images).map_err(|why| Error::Usage(format!("A language model cannot be asked that: {why}")))?;
            } else {
                prompt.user = left_out(images.len(), &prompt.user);
            }
        }
        let mut model = String::new();
        let outputs = ask(lm, lm.model(), &prompt, &format, self.retries, spent, &mut model, wrote, |reply| {
            self.read(reply)
        })
        .map_err(|why| lm.failed(&why))?;
        Ok(Generated {
            outputs,
            usage: *spent,
            model,
        })
    }

    /// Whether the signature is one a model can be asked to complete.
    fn check(&self) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(Error::Usage("A signature to complete has at least one output.".to_owned()));
        }
        if let Some(name) = self.outputs.keys().find(|name| self.inputs.contains_key(*name)) {
            return Err(Error::Usage(format!("'{name}' is both an input and an output of the signature.")));
        }
        Ok(())
    }

    /// The outputs a reply holds, each as its type, or what is wrong with the reply.
    fn read(&self, reply: &Reply) -> std::result::Result<Map<String, Json>, String> {
        let names: Vec<&String> = self.outputs.keys().collect();
        let object = match reply.object(&names) {
            Some(object) => object.clone(),
            // One text was asked for and text came back, with no object about it: that is the
            // text, and what is inside the quotes if it was written as a string of JSON.
            None if !reply.text.trim().is_empty() && !reply.text.contains('{') && self.only_text() => {
                let text = reply.text.trim();
                let text = serde_json::from_str::<String>(text).unwrap_or_else(|_| text.to_owned());
                Map::from_iter([(names[0].clone(), Json::String(text))])
            }
            None => return Err(NOT_AN_OBJECT.to_owned()),
        };
        let (mut outputs, mut mistakes) = (Map::new(), Vec::new());
        for (name, field) in &self.outputs {
            match object.get(name).cloned().map(|value| field.kind.fit(value)) {
                Some(Ok(value)) => drop(outputs.insert(name.clone(), value)),
                Some(Err(problem)) => mistakes.push(format!("`{name}` is declared `{}`, and {problem}.", field.kind)),
                None => mistakes.push(format!("`{name}` is missing.")),
            }
        }
        if mistakes.is_empty() {
            Ok(outputs)
        } else {
            Err(mistakes.join(" "))
        }
    }

    /// Whether the one output is a text.
    fn only_text(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.values().all(|field| field.kind == Type::Str)
    }
}

/// What a [`Generate`] came to: the outputs by name, each of its declared type, and what it took.
///
/// As a tool's return value it is its outputs: an object by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    /// The outputs, by name, in the order the signature declares them.
    pub outputs: Map<String, Json>,
    /// The tokens every request for it used, a retry's with the first's.
    pub usage: Usage,
    /// The model that answered, as the provider named it.
    pub model: String,
}

impl Generated {
    /// An output, if the signature has one of that name.
    pub fn get(&self, name: &str) -> Option<&Json> {
        self.outputs.get(name)
    }

    /// An output that is text.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.outputs.get(name).and_then(Json::as_str)
    }

    /// An output as a type of the code's own: `read::<Vec<String>>("parts")`. The outputs
    /// were checked against the signature, so a name it does not have, or a type the output
    /// was not declared as, is the code's mistake: a usage error.
    pub fn read<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        let value = self.outputs.get(name).ok_or_else(|| {
            Error::Usage(format!(
                "No output is named '{name}': the signature has {}.",
                listed(self.outputs.keys())
            ))
        })?;
        serde_json::from_value(value.clone())
            .map_err(|error| Error::Usage(format!("The output '{name}' is read as a type it is not: {error}.")))
    }

    /// Every output together, as a type of the code's own: a struct with a field for each,
    /// say. As with [`Generated::read`], a type the outputs are not is the code's mistake.
    ///
    /// ```
    /// use rdm::inputs;
    /// use rdm::lm::{Generate, Lm, ScriptedLm};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct Answer {
    ///     answer: String,
    ///     certain: bool,
    /// }
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"answer": "260 euros a night.", "certain": true}"#.to_owned()), "scripted");
    /// let answer = Generate::parse("passage, question -> answer, certain: bool").unwrap().lm(lm);
    /// let written: Answer = answer
    ///     .call(inputs! { "passage" => "London: 260 euros a night.", "question" => "What is the hotel limit?" })
    ///     .unwrap()
    ///     .read_all()
    ///     .unwrap();
    /// assert_eq!((written.answer.as_str(), written.certain), ("260 euros a night.", true));
    /// ```
    pub fn read_all<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(Json::Object(self.outputs.clone())).map_err(|error| {
            Error::Usage(format!(
                "The outputs {} are read as a type they are not: {error}.",
                listed(self.outputs.keys())
            ))
        })
    }

    /// The tokens in and out together.
    pub fn tokens(&self) -> u64 {
        self.usage.total()
    }
}

/// An output by name; null for a name the signature does not have.
impl std::ops::Index<&str> for Generated {
    type Output = Json;

    fn index(&self, name: &str) -> &Json {
        static NULL: Json = Json::Null;
        self.outputs.get(name).unwrap_or(&NULL)
    }
}

/// The outputs, an object by name: what a tool that returns it is observed to have returned.
impl serde::Serialize for Generated {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.outputs.serialize(serializer)
    }
}

/// A decision model that is an ordinary language model.
///
/// A decision request is put to it as a prompt: each question as the JSON a decision model
/// is sent (`{"instructions", "type", "criteria"}`), the state as the inputs, and for each
/// question the schema of the evidence to answer with. A yes or no is answered with the
/// probability of yes; a choice and a score with a probability for each option or level and
/// a confidence. Those are what a System One model returns, so what is decided from them is
/// decided as it always is, by the module's thresholds, cuts and weights, which the language
/// model never sees.
///
/// The evidence is checked before it is returned: every question answered, labels exactly
/// the ones declared, every probability from 0 to 1. Evidence that fails is shown back to the
/// model once, with what is wrong with it. If it fails again the request has failed, as it
/// has when a decision model answers with something that is not evidence.
///
/// The probabilities are what a language model says they are, not what it measured: use a
/// decision model where the numbers matter, and this where there is none.
///
/// ```
/// use rdm::lm::{Lm, LmDecisionModel, ScriptedLm};
/// use rdm::{Rdm, Signature, inputs};
///
/// let lm = Lm::of(
///     ScriptedLm::new(|_system, _prompt| r#"{"refund": {"noul": 0.93}, "team": {"probabilities": {"billing": 0.8, "sales": 0.2}, "confidence": 0.7}}"#.to_owned()).tokens(210, 30),
///     "scripted",
/// );
/// let triage = Signature::new("Triage a message from a customer.")
///     .input("message")
///     .noul("refund", "Is the customer in `inputs.message` asking for a refund?")
///     .choice("team", "Which team should handle `inputs.message`?", ["billing", "sales"]);
///
/// let result = Rdm::new(triage).model(LmDecisionModel::new(lm)).call(inputs! { "message" => "I was charged twice." }).unwrap();
/// assert!(result.yes("refund"));
/// assert_eq!(result.chosen("team"), Some("billing"));
/// assert_eq!(result.tokens(), 240);
/// ```
#[derive(Clone, Debug)]
pub struct LmDecisionModel {
    lm: Lm,
    retries: usize,
    leave_out_images: bool,
}

impl LmDecisionModel {
    /// A decision model that asks this language model.
    pub fn new(lm: Lm) -> Self {
        Self {
            lm,
            retries: 1,
            leave_out_images: false,
        }
    }

    /// Whether a request with pictures in it ([`crate::Signature::image`]) is answered without
    /// them by a model that is not shown pictures ([`Lm::images`]): each input that is one
    /// shows `<image: name>`, and the model is told how many were left out. It is not, unless
    /// this says so: such a request is refused with [`systemone::Error::Invalid`] before
    /// anything is sent, as a client of a server that does not read pictures refuses it. A
    /// model that is shown pictures is given them in any case.
    ///
    /// ```
    /// use rdm::lm::{Lm, LmDecisionModel, ScriptedLm};
    /// use rdm::{Signature, image};
    /// use systemone::{DecisionModel, Error};
    ///
    /// let look = Signature::new("Look.").image("window", "").noul("editor", "Is `inputs.window` an editor?");
    /// let request = look.request(serde_json::json!({"window": image("image/png", b"png")}).as_object().unwrap()).unwrap();
    /// let lm = Lm::of(ScriptedLm::new(|_system, _prompt| r#"{"editor": {"noul": 0.5}}"#.to_owned()), "scripted");
    /// assert!(matches!(LmDecisionModel::new(lm.clone()).decide(&request), Err(Error::Invalid(_))));
    /// assert!(LmDecisionModel::new(lm).leave_out_images(true).decide(&request).is_ok());
    /// ```
    pub fn leave_out_images(mut self, leave_out: bool) -> Self {
        self.leave_out_images = leave_out;
        self
    }

    /// How many times evidence that is not evidence for the questions is shown back to the
    /// model, with what is wrong with it, for another. Once, unless this says otherwise.
    pub fn retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }

    /// The language model it asks.
    pub fn lm(&self) -> &Lm {
        &self.lm
    }
}

impl From<Lm> for LmDecisionModel {
    fn from(lm: Lm) -> Self {
        LmDecisionModel::new(lm)
    }
}

impl DecisionModel for LmDecisionModel {
    /// Answer every question of a request with evidence. A request that names a model is put
    /// to that model, through the same handle; one that names none is put to the handle's own.
    fn decide(&self, request: &systemone::Request) -> std::result::Result<systemone::Response, systemone::Error> {
        let model = if request.model.is_empty() {
            self.lm.model()
        } else {
            request.model.as_str()
        };
        if request.questions.is_empty() {
            return Ok(systemone::Response {
                model: model.to_owned(),
                ..Default::default()
            });
        }
        for (name, question) in &request.questions {
            if question.kind != Kind::Noul && question.labels().is_empty() {
                return Err(systemone::Error::Invalid(format!(
                    "Question '{name}' is a {} with nothing to choose from.",
                    question.kind.as_str()
                )));
            }
        }
        let state = State::of(&request.state, request.questions.keys());
        let listed = request.questions.iter().enumerate().map(|(position, (name, question))| {
            let asked = serde_json::to_string_pretty(question).unwrap_or_default();
            format!("{}. `{name}` ({}):\n{asked}", position + 1, evidence_name(question.kind))
        });
        let mut prompt = Prompt {
            system: system(
                &state.input_fields,
                &listed.collect::<Vec<_>>().join("\n"),
                state.inputs.keys(),
                &evidence_schema(&request.questions, true),
                &state.instructions,
            ),
            user: user(state.inputs.iter(), request.questions.keys()),
            images: Vec::new(),
        };
        if !state.images.is_empty() {
            if self.lm.images {
                prompt.images = pictures(&state.images)
                    .map_err(|why| systemone::Error::Invalid(format!("The language model '{model}' cannot be asked that: {why}")))?;
            } else if self.leave_out_images {
                prompt.user = left_out(state.images.len(), &prompt.user);
            } else {
                return Err(systemone::Error::Invalid(format!(
                    "The request shows {} picture(s), and the language model '{model}' is not shown pictures: show it them with \
                     `Lm::images(true)`, or answer without them with `LmDecisionModel::leave_out_images(true)`.",
                    state.images.len()
                )));
            }
        }
        let format = Format {
            name: "evidence",
            strict: true,
            plain: request
                .questions
                .iter()
                .all(|(name, question)| plain(name) && question.labels().iter().all(|label| plain(label))),
            schema: evidence_schema(&request.questions, false),
        };
        let (mut usage, mut served) = (Usage::default(), String::new());
        let read = |reply: &Reply| evidence(&request.questions, reply);
        // What it wrote is evidence, and the request's `asked` event keeps it: it is not kept twice.
        let answers = ask(
            &self.lm,
            model,
            &prompt,
            &format,
            self.retries,
            &mut usage,
            &mut served,
            &mut Vec::new(),
            read,
        )
        .map_err(|why| match why {
            Failed::Lm(error) => model_error(model, &error),
            Failed::Reply(mistake) => {
                systemone::Error::Evidence(format!("The language model '{model}' did not answer with evidence: {mistake}"))
            }
            Failed::Request(why) => systemone::Error::Invalid(format!("The language model '{model}' cannot be asked that: {why}")),
        })?;
        Ok(systemone::Response {
            model: served,
            answers,
            usage,
        })
    }

    fn name(&self) -> String {
        self.lm.model().to_owned()
    }
}

/// What the model is told, and what it is asked.
struct Prompt {
    system: String,
    user: String,
    /// The pictures shown after the text, as lm15's parts. None for most.
    images: Vec<Part>,
}

impl Prompt {
    /// The message that asks: the text, and after it the pictures, if there are any.
    fn asking(&self, text: String) -> std::result::Result<Message, Failed> {
        if self.images.is_empty() {
            return said_by(Message::user(text));
        }
        let mut parts = vec![
            Part::text(text),
            Part::text("The pictures, in the order of the inputs that stand for them:"),
        ];
        parts.extend(self.images.iter().cloned());
        said_by(Message::new(lm15::Role::User, parts))
    }
}

/// The prompt, for a model told that pictures were left out: never left out without a word,
/// so it reads the rest knowing it.
fn left_out(count: usize, user: &str) -> String {
    let pictures = if count == 1 {
        "1 picture".to_owned()
    } else {
        format!("{count} pictures")
    };
    format!(
        "{pictures} of the inputs left out: this model is not shown pictures. An input that shows `<image: ...>` stands for one.\n\n{user}"
    )
}

/// The pictures of a state as lm15's image parts: the bytes of each, in base64.
fn pictures(images: &[Json]) -> std::result::Result<Vec<Part>, String> {
    let each = images.iter().map(|item| {
        let made = match systemone::Image::of(item) {
            Some(picture) => lm15::ImagePart::from_data(picture.content_type, picture.data),
            None => match item.pointer("/image_url/url").and_then(Json::as_str) {
                Some(url) => lm15::ImagePart::from_url(url),
                None => return Err("a picture of the state has no address".to_owned()),
            },
        };
        made.map(Part::Image).map_err(|error| error.message)
    });
    each.collect()
}

/// The schema a reply is held to, where a wire can hold it to one.
struct Format {
    name: &'static str,
    schema: Json,
    /// Whether the schema says everything: no object in it may have keys it does not name.
    strict: bool,
    /// Whether the schema is one every wire that takes schemas takes: it says everything, and
    /// every name in it is a plain one ([`plain`]). An option of a choice is a name in the
    /// schema of its evidence, and an option may be any text.
    plain: bool,
}

/// Why asking came to nothing.
enum Failed {
    /// The model could not be asked, or failed.
    Lm(Box<Lm15Error>),
    /// The model answered, and not with what it was asked for.
    Reply(String),
    /// The request is not one lm15 takes.
    Request(String),
}

/// What a model replied: its text, and every JSON object found in it.
struct Reply {
    text: String,
    objects: Vec<Map<String, Json>>,
    /// Whether the reply stopped at the limit of tokens.
    cut: bool,
}

impl Reply {
    fn of(response: &lm15::Response) -> Self {
        let mut text = written(response);
        // A wire that holds a reply to a schema may hand the object over already read.
        let read = response.message.parts.iter().filter_map(|part| match part {
            Part::Data(data) => data.value.as_object().cloned(),
            _ => None,
        });
        let mut objects: Vec<Map<String, Json>> = read.collect();
        if objects.is_empty() {
            objects = objects_in(&text);
        } else if text.trim().is_empty() {
            // The object took the place of the text: it is what the model wrote, and what it is shown to have written.
            let each: Vec<String> = objects.iter().map(|object| Json::Object(object.clone()).to_string()).collect();
            text = each.join("\n");
        }
        Self {
            text,
            objects,
            cut: response.finish_reason == FinishReason::Length,
        }
    }

    /// The object that is the reply: the first that has every name asked for, or else the first there is.
    fn object(&self, names: &[&String]) -> Option<&Map<String, Json>> {
        let whole = self
            .objects
            .iter()
            .find(|object| names.iter().all(|name| object.contains_key(*name)));
        whole.or_else(|| self.objects.first())
    }
}

/// What is wrong with a reply that has no JSON object in it.
const NOT_AN_OBJECT: &str = "it is not a JSON object.";

/// The text of a reply: what the model wrote, and what it said in place of writing if it would not.
fn written(response: &lm15::Response) -> String {
    let texts = response.message.parts.iter().filter_map(|part| match part {
        Part::Text(text) => Some(text.text.as_str()),
        Part::Refusal(refusal) => Some(refusal.text.as_str()),
        _ => None,
    });
    texts.collect::<Vec<_>>().join("\n")
}

/// Every JSON object written in a text, in order: the text itself if it is one, or else each
/// that starts at a `{` and reads to its end, whatever is round it. A code fence, a sentence
/// before or after, a thought written first: none of them is in the way.
fn objects_in(text: &str) -> Vec<Map<String, Json>> {
    if let Ok(Json::Object(object)) = serde_json::from_str(text.trim()) {
        return vec![object];
    }
    let (mut found, mut from) = (Vec::new(), 0);
    while let Some(at) = text[from..].find('{') {
        let start = from + at;
        let mut values = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Json>();
        match values.next() {
            Some(Ok(Json::Object(object))) => {
                found.push(object);
                from = start + values.byte_offset();
            }
            _ => from = start + 1,
        }
    }
    found
}

/// Ask for one JSON object and read it. A reply that cannot be read is shown back to the
/// model with what is wrong with it, as many times as `retries` says.
#[allow(clippy::too_many_arguments)]
fn ask<T>(
    lm: &Lm,
    model: &str,
    prompt: &Prompt,
    format: &Format,
    retries: usize,
    spent: &mut Usage,
    served_by: &mut String,
    wrote: &mut Vec<Wrote>,
    read: impl Fn(&Reply) -> std::result::Result<T, String>,
) -> std::result::Result<T, Failed> {
    let mut messages = vec![prompt.asking(prompt.user.clone())?];
    let mut attempt = 0;
    loop {
        let mut request = lm.asking(model, &prompt.system, messages.clone(), Some(format));
        let sent = lm.send_for_json(&mut request, format);
        wrote.push(Wrote {
            model: model.to_owned(),
            response: sent.as_ref().ok().cloned(),
            error: sent.as_ref().err().map(ToString::to_string),
            request,
        });
        let response = sent.map_err(Failed::Lm)?;
        let usage = used(&response.usage);
        spent.input_tokens += usage.input_tokens;
        spent.output_tokens += usage.output_tokens;
        *served_by = served(&response, model);
        let reply = Reply::of(&response);
        let mistake = match read(&reply) {
            Ok(value) => return Ok(value),
            // A reply that stopped at the limit of tokens is wrong for that reason, whatever is missing from it.
            Err(mistake) if reply.cut => format!("{mistake} It was cut short at the limit of tokens."),
            Err(mistake) => mistake,
        };
        if attempt == retries {
            return Err(Failed::Reply(mistake));
        }
        attempt += 1;
        let correction = format!(
            "That reply cannot be used: {mistake}\nReply again with one JSON object and nothing else, matching the JSON Schema you were given."
        );
        if reply.text.trim().is_empty() {
            // Nothing was written to show back: the request again, with what was wrong.
            messages = vec![prompt.asking(format!("{}\n\n{correction}", prompt.user))?];
        } else {
            messages.push(said_by(Message::assistant(reply.text.as_str()))?);
            messages.push(said_by(Message::user(correction))?);
        }
    }
}

/// The system prompt: the fields, how a request is laid out, the schema of the reply, and what it is all for.
fn system<'a>(
    input_fields: &str,
    output_fields: &str,
    inputs: impl Iterator<Item = &'a String>,
    schema: &Json,
    instructions: &str,
) -> String {
    // No line of the lists ends in a space, whoever wrote it: a field with no description is its name and type.
    let tidy = |fields: &str| fields.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    let mut text = format!(
        "Your input fields are:\n{}\nYour output fields are:\n{}\n",
        tidy(input_fields),
        tidy(output_fields)
    );
    text.push_str("All interactions will be structured in the following way, with the appropriate values filled in.\n\n");
    let blocks: Vec<String> = inputs.map(|name| format!("[[ ## {name} ## ]]\n{{{name}}}\n\n")).collect();
    if !blocks.is_empty() {
        text.push_str("Inputs will have the following structure:\n\n");
        text.push_str(&blocks.concat());
    }
    text.push_str("Outputs will be one JSON object and nothing else, matching this JSON Schema:\n\n");
    text.push_str(&schema.to_string());
    text.push_str("\n\nIn adhering to this structure, your objective is:\n");
    text.push_str(instructions);
    text
}

/// A value as a prompt shows it, which is as DSPy's adapters show it: a text as it is, an
/// object or a list as JSON, and anything else as Python writes it (`True`, `None`, `1.0`).
fn shown(value: &Json) -> String {
    match value {
        Json::Null => "None".to_owned(),
        Json::Bool(true) => "True".to_owned(),
        Json::Bool(false) => "False".to_owned(),
        other => as_observation(other),
    }
}

/// The message: each input under its name, as [`shown`], and the fields to answer with.
fn user<'a>(inputs: impl Iterator<Item = (&'a String, &'a Json)>, outputs: impl Iterator<Item = &'a String>) -> String {
    let mut text: String = inputs
        .map(|(name, value)| format!("[[ ## {name} ## ]]\n{}\n\n", shown(value)))
        .collect();
    let outputs: Vec<String> = outputs.map(|name| format!("`{name}`")).collect();
    // Each input ends in a blank line, so the closing sentence follows one: the same text as `CLOSING`, less that.
    text.push_str(CLOSING.trim_start());
    text.push_str(&outputs.join(", then "));
    text.push('.');
    text
}

/// Fields as a model is told of them: one numbered line each.
fn numbered<'a>(fields: impl Iterator<Item = (&'a String, String, &'a str)>) -> String {
    let lines = fields
        .enumerate()
        .map(|(position, (name, type_name, description))| format!("{}. `{name}` ({type_name}): {description}", position + 1));
    lines.collect::<Vec<_>>().join("\n").trim().to_owned()
}

/// Whether a name is one any wire takes as a key of a schema: letters, digits, `_`, `.` and `-`, and no more than 64 of them.
fn plain(name: &str) -> bool {
    let allowed = |character: char| character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-');
    !name.is_empty() && name.len() <= 64 && name.chars().all(allowed)
}

/// An object that has exactly these properties, every one of them.
fn object_schema(properties: Map<String, Json>) -> Json {
    let required: Vec<&String> = properties.keys().collect();
    json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
}

/// The state of a decision request as a prompt shows it: what it is for, the inputs, and how they are described.
struct State {
    instructions: String,
    input_fields: String,
    inputs: Map<String, Json>,
    /// The pictures shown after the rest: the items of an array state after the first.
    images: Vec<Json>,
}

impl State {
    /// A state in the shape a module sends (`instructions`, `input_fields`, `inputs`) is read
    /// as that. Any other state, a string or an object of the caller's own, is one input named `state`.
    ///
    /// A state that is an array of an object and then pictures (`{"type": "image_url", ...}`),
    /// as a signature with pictures makes one, is read as that object, and the pictures.
    fn of<'a>(state: &Json, questions: impl Iterator<Item = &'a String>) -> Self {
        let (state, images) = match state.as_array().and_then(|items| items.split_first()) {
            Some((first, rest)) if first.is_object() && !rest.is_empty() && rest.iter().all(systemone::is_image) => (first, rest.to_vec()),
            _ => (state, Vec::new()),
        };
        let shaped = state.as_object().filter(|object| {
            let known = object.iter().all(|(key, value)| match key.as_str() {
                "instructions" | "input_fields" => value.is_string(),
                "inputs" => value.is_object(),
                "demos" => value.is_array(),
                _ => false,
            });
            known && object.contains_key("inputs")
        });
        let inputs = match shaped.and_then(|object| object["inputs"].as_object()) {
            Some(inputs) => inputs.clone(),
            None => Map::from_iter([("state".to_owned(), state.clone())]),
        };
        let text = |key: &str| shaped.and_then(|object| object.get(key)).and_then(Json::as_str).unwrap_or_default();
        let ticked = |names: Vec<&String>| names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ");
        let mut instructions = match text("instructions") {
            "" => format!(
                "Given the fields {}, produce the fields {}.",
                ticked(inputs.keys().collect()),
                ticked(questions.collect())
            ),
            given => given.to_owned(),
        };
        // Examples are labels, not distributions: they are shown as they are, beside the instructions.
        if let Some(demos) = shaped
            .and_then(|object| object.get("demos"))
            .filter(|demos| demos.as_array().is_some_and(|demos| !demos.is_empty()))
        {
            instructions.push_str("\n\nTask examples (labels or evidence):\n");
            instructions.push_str(&as_observation(demos));
        }
        let input_fields = match text("input_fields") {
            "" => numbered(inputs.iter().map(|(name, value)| (name, type_of(value).to_owned(), ""))),
            given => given.to_owned(),
        };
        Self {
            instructions,
            input_fields,
            inputs,
            images,
        }
    }
}

/// The type a value would be declared as.
fn type_of(value: &Json) -> &'static str {
    match value {
        Json::Null => "None",
        Json::Bool(_) => "bool",
        Json::Number(number) if number.is_f64() => "float",
        Json::Number(_) => "int",
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// What a value is, in a sentence about a reply.
fn what(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "true or false",
        Json::Number(_) => "a number",
        Json::String(_) => "a string",
        Json::Array(_) => "a list",
        Json::Object(_) => "an object",
    }
}

/// The name of the evidence a kind of question is answered with.
fn evidence_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Noul => "NoulEvidence",
        Kind::Choice => "ChoiceEvidence",
        Kind::Score => "ScoreEvidence",
    }
}

/// The schema of the evidence for every question: for a yes or no, the probability of yes;
/// for a choice or a score, a probability for each label and a confidence. Every object is
/// closed. The ranges are said where the model reads the schema, and left out of the one a
/// wire is given: not every wire that takes a schema takes a range in it.
fn evidence_schema(questions: &IndexMap<String, Question>, ranges: bool) -> Json {
    let probability = |description: Option<&str>| {
        let mut schema = Map::from_iter([("type".to_owned(), json!("number"))]);
        if let Some(description) = description {
            schema.insert("description".to_owned(), description.into());
        }
        if ranges {
            schema.insert("minimum".to_owned(), 0.into());
            schema.insert("maximum".to_owned(), 1.into());
        }
        Json::Object(schema)
    };
    let each = questions.iter().map(|(name, question)| {
        let evidence = match question.kind {
            Kind::Noul => Map::from_iter([("noul".to_owned(), probability(Some("Probability that the answer is true.")))]),
            Kind::Choice | Kind::Score => {
                let labels = question.labels().into_iter().map(|label| (label, probability(None))).collect();
                let mut probabilities = object_schema(labels);
                probabilities["description"] = "Probability of each option; sum to one.".into();
                Map::from_iter([
                    ("probabilities".to_owned(), probabilities),
                    (
                        "confidence".to_owned(),
                        probability(Some("Confidence in the decision, from 0 to 1.")),
                    ),
                ])
            }
        };
        (name.clone(), object_schema(evidence))
    });
    object_schema(each.collect())
}

/// The evidence a reply holds for every question, or what is wrong with it.
fn evidence(questions: &IndexMap<String, Question>, reply: &Reply) -> std::result::Result<IndexMap<String, Answer>, String> {
    let names: Vec<&String> = questions.keys().collect();
    let object = reply.object(&names).ok_or(NOT_AN_OBJECT)?;
    let (mut answers, mut mistakes) = (IndexMap::new(), Vec::new());
    for (name, question) in questions {
        let given = match object.get(name) {
            Some(given) => given.clone(),
            // One question, and its evidence written without its name round it.
            None if questions.len() == 1 && (object.contains_key("noul") || object.contains_key("probabilities")) => {
                Json::Object(object.clone())
            }
            None => {
                mistakes.push(format!("`{name}` is missing."));
                continue;
            }
        };
        match evidence_for(question, &given) {
            Ok(answer) => drop(answers.insert(name.clone(), answer)),
            Err(mistake) => mistakes.push(format!("`{name}`: {mistake}")),
        }
    }
    if mistakes.is_empty() {
        Ok(answers)
    } else {
        Err(mistakes.join("\n"))
    }
}

/// The evidence for one question, checked: a probability from 0 to 1, for exactly the labels declared.
fn evidence_for(question: &Question, given: &Json) -> std::result::Result<Answer, String> {
    // A number written as a string is that number, as it is to the Python module, which reads evidence through pydantic.
    let number = |value: &Json| match value {
        Json::String(text) => text.trim().parse::<f64>().ok().filter(|number| number.is_finite()),
        other => other.as_f64(),
    };
    let unit = |value: &Json, what: &str| match number(value) {
        Some(number) if (0.0..=1.0).contains(&number) => Ok(number),
        Some(number) => Err(format!("{what} is {number}, which is not from 0 to 1.")),
        None => Err(format!("{what} is {}, and it should be a number from 0 to 1.", self::what(value))),
    };
    if question.kind == Kind::Noul {
        let probability = match given {
            Json::Object(object) => object
                .get("noul")
                .ok_or(r#"it should be {"noul": the probability that the answer is true}."#)?,
            bare => bare,
        };
        return Ok(Answer::noul(unit(probability, "the probability")?));
    }
    let labels = question.labels();
    let object = given
        .as_object()
        .ok_or_else(|| format!("it is {}, and it should be an object.", what(given)))?;
    let probabilities = match object.get("probabilities") {
        Some(Json::Object(probabilities)) => probabilities,
        Some(other) => return Err(format!("`probabilities` is {}, and it should be an object.", what(other))),
        // The distribution written without `probabilities` round it.
        None if !object.is_empty() && object.keys().all(|key| labels.contains(key)) => object,
        None => return Err("`probabilities` is missing.".to_owned()),
    };
    let missing: Vec<&String> = labels.iter().filter(|label| !probabilities.contains_key(*label)).collect();
    let unknown: Vec<&String> = probabilities.keys().filter(|key| !labels.contains(key)).collect();
    if !missing.is_empty() || !unknown.is_empty() {
        // The labels are in the question: only the ones that are wrong are said again.
        let mut mistake = match question.kind {
            Kind::Score => format!("the probabilities must be for exactly the levels '0' to '{}'.", labels.len() - 1),
            _ => "the probabilities must be for exactly the options declared.".to_owned(),
        };
        if !missing.is_empty() {
            mistake.push_str(&format!(" Missing: {}.", listed(&missing)));
        }
        if !unknown.is_empty() {
            mistake.push_str(&format!(" Not declared: {}.", listed(&unknown)));
        }
        return Err(mistake);
    }
    let mut ordered = Vec::new();
    for label in &labels {
        ordered.push((
            label.clone(),
            unit(&probabilities[label], &format!("the probability of '{label}'"))?,
        ));
    }
    if ordered.iter().map(|(_, probability)| probability).sum::<f64>() <= 0.0 {
        return Err("every probability is zero.".to_owned());
    }
    let mut answer = match question.kind {
        Kind::Score => Answer::score(ordered.into_iter().map(|(_, probability)| probability)),
        _ => Answer::choice(ordered),
    };
    // A confidence the model left out is worked out from the distribution, as a decision model publishes one.
    if let Some(confidence) = object.get("confidence").filter(|confidence| !confidence.is_null()) {
        answer.confidence = Some(unit(confidence, "the confidence")?);
    }
    Ok(answer)
}

/// The tokens a request used: in and out. A provider that gave only a total has it counted as in.
fn used(usage: &lm15::Usage) -> Usage {
    match (usage.input_tokens, usage.output_tokens) {
        (None, None) => Usage {
            input_tokens: usage.total_tokens.unwrap_or(0),
            output_tokens: 0,
        },
        (input, output) => Usage {
            input_tokens: input.unwrap_or(0),
            output_tokens: output.unwrap_or(0),
        },
    }
}

/// The model that answered, as the provider named it, or else as it was asked for.
fn served(response: &lm15::Response, asked: &str) -> String {
    if response.model.is_empty() {
        asked.to_owned()
    } else {
        response.model.clone()
    }
}

/// How long to wait before a request is sent again: what the server asked for, up to a
/// minute, or else half a second doubling to five.
fn wait(again: u32, asked: Option<f64>) -> Duration {
    match asked {
        Some(seconds) if seconds.is_finite() && seconds >= 0.0 => Duration::from_secs_f64(seconds.min(60.0)),
        _ => Duration::from_millis(500u64.saturating_mul(1 << again.min(4)).min(5000)),
    }
}

/// A router as it is before anything is said of it: with lm15's own network on a machine,
/// the host's HTTP in a module built for WASI 0.2, and no network in any other module.
fn router() -> lm15::RouterConfig {
    let config = lm15::RouterConfig::new();
    #[cfg(all(target_os = "wasi", target_env = "p2"))]
    let config = config.transport(crate::wasi::WasiHttp);
    config
}

/// Wait for what a language model answers, on lm15's own runtime.
#[cfg(not(target_family = "wasm"))]
fn waited<F: Future>(asked: F) -> Option<F::Output> {
    Some(lm15::blocking::block_on(asked))
}

/// Wait for what a language model answers, in WebAssembly: there is no runtime to wait on and
/// no second thread to wake this one, so the answer is asked for here until it is there. A
/// host that answers before it returns has it ready the first time. One that does not is not
/// waited for without end: nothing in the module would ever make its answer ready.
#[cfg(target_family = "wasm")]
fn waited<F: Future>(asked: F) -> Option<F::Output> {
    use std::task::{Context, Poll, Waker};
    let mut asked = std::pin::pin!(asked);
    let mut context = Context::from_waker(Waker::noop());
    (0..1024).find_map(|_| match asked.as_mut().poll(&mut context) {
        Poll::Ready(answered) => Some(answered),
        Poll::Pending => None,
    })
}

/// An answer that was not there when it was waited for, which only a module can meet.
/// One time a language model was asked, as a run's `wrote` event keeps it.
pub(crate) struct Wrote {
    /// The model asked for.
    pub(crate) model: String,
    /// The request as it was sent.
    pub(crate) request: lm15::Request,
    /// What came back.
    pub(crate) response: Option<lm15::Response>,
    /// What it failed with.
    pub(crate) error: Option<String>,
}

/// A reply as a `wrote` event keeps it: in lm15's JSON, as it reads back. lm15 makes a reply
/// whole when it reads one (a total of tokens where the parts are given), so a reply is kept as
/// it is after a reading, and the reply a replay gives is written down the same.
pub(crate) fn as_kept(response: &lm15::Response) -> Option<Json> {
    let written = serde_json::to_value(response).ok()?;
    let read: lm15::Response = serde_json::from_value(written.clone()).ok()?;
    serde_json::to_value(read).ok().or(Some(written))
}

/// The key a language model's reply is kept under: the SHA-256 of the request in lm15's own
/// JSON, in hex. The same request is the same key in any process.
pub fn key(request: &lm15::Request) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    if serde_json::to_writer(&mut hash, request).is_err() {
        // A request is JSON, and JSON can always be written.
        hash.update(b"unwritten");
    }
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// What a [`ReplayLm`] says it failed with where it was never asked the request: a replay that
/// has left what was recorded, which ends the run as a decision model's [`systemone::Replay`] does.
const UNRECORDED: &str = "unrecorded";

/// The code of a request stopped in the middle of its reply: by its watch, or by the stop of
/// the branch it was asked for. No provider gave it.
const STOPPED: &str = "stopped";

/// A request stopped in the middle of its reply, and why.
fn stopped_by(why: &str) -> Lm15Error {
    let mut meta = lm15::ErrorMeta::new(why);
    meta.provider_code = Some(STOPPED.to_owned());
    Lm15Error::ProviderError(meta)
}

/// Now, where there is a clock to read: a WebAssembly module with no system under it has none,
/// and reading one there is a panic.
fn clock() -> Option<Instant> {
    cfg!(not(all(target_family = "wasm", target_os = "unknown"))).then(Instant::now)
}

/// A language model that answers each request with the reply it got when it was asked
/// before, from a run's `wrote` events, and refuses one that was never asked.
///
/// A refusal ends the run, as a decision model's replay does ([`systemone::Error::Unrecorded`]):
/// it is a run that has gone another way than the one recorded.
///
/// ```
/// use std::sync::{Arc, Mutex};
///
/// use rdm::lm::{Generate, Lm, ReplayLm, ScriptedLm};
/// use rdm::record::Event;
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let events = Arc::new(Mutex::new(Vec::new()));
/// let kept = events.clone();
/// let run = |lm: Lm| {
///     let answer = Generate::parse("question -> answer").unwrap().lm(lm);
///     let reply = Tool::new("reply", "Answer.", move |call| answer.call_in(call, inputs! { "question" => call.arg::<String>("question")? })?.read::<String>("answer"));
///     let kept = kept.clone();
///     Rdm::new(Signature::new("Chat.").input("question").noul("reply", "Is `inputs.question` a question?"))
///         .tool(reply)
///         .model(Scripted::new(|_| [("reply", Answer::noul(0.9))]))
///         .record(move |event: &Event| Ok(kept.lock().unwrap().push(event.clone())))
///         .call(inputs! { "question" => "What eats mice at night?" })
///         .unwrap()
/// };
/// let live = run(Lm::of(ScriptedLm::new(|_, _| r#"{"answer": "An owl."}"#.to_owned()), "scripted"));
/// let written = events.lock().unwrap().clone();
/// let again = run(Lm::of(ReplayLm::new(&written), "scripted"));
/// assert_eq!(again.observations["reply"], live.observations["reply"]);
/// ```
pub struct ReplayLm {
    kept: std::collections::HashMap<String, std::result::Result<lm15::Response, String>>,
}

impl std::fmt::Debug for ReplayLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayLm").field("kept", &self.kept.len()).finish()
    }
}

impl ReplayLm {
    /// A language model that answers what the `wrote` events among these say was asked.
    pub fn new<'a>(events: impl IntoIterator<Item = &'a crate::record::Event>) -> Self {
        let mut kept = std::collections::HashMap::new();
        for event in events {
            if let crate::record::Event::Wrote { key, response, error, .. } = event {
                let response = response.as_ref().and_then(|response| serde_json::from_value(response.clone()).ok());
                let answered = match (response, error) {
                    (Some(response), _) => Ok(response),
                    (None, Some(error)) => Err(error.clone()),
                    (None, None) => continue,
                };
                kept.insert(key.clone(), answered);
            }
        }
        Self { kept }
    }

    // The error is lm15's, as its `LanguageModel` returns it.
    #[allow(clippy::result_large_err)]
    fn answer(&self, request: &lm15::Request) -> std::result::Result<lm15::Response, Lm15Error> {
        let key = key(request);
        match self.kept.get(&key) {
            Some(Ok(response)) => Ok(response.clone()),
            // A failure that was kept is given again, as one that asking again would not mend.
            Some(Err(error)) => Err(Lm15Error::ProviderError(lm15::ErrorMeta::new(error.clone()))),
            None => {
                let mut meta = lm15::ErrorMeta::new(format!(
                    "the language model was never asked this request, so there is no reply kept for it: key {key}"
                ));
                meta.provider_code = Some(UNRECORDED.to_owned());
                Err(Lm15Error::NotConfiguredError(meta))
            }
        }
    }
}

impl LanguageModel for ReplayLm {
    fn complete<'a>(
        &'a self,
        request: &'a lm15::Request,
    ) -> std::pin::Pin<Box<dyn Future<Output = std::result::Result<lm15::Response, Lm15Error>> + Send + 'a>> {
        Box::pin(async move { self.answer(request) })
    }

    fn stream(&self, request: &lm15::Request) -> lm15::testing::LanguageModelStream {
        replayed(self.answer(request), request)
    }
}

/// A reply, given again as the events of a stream by lm15's own fake. An object lm15 read out of
/// the text of a reply is given as that text, as it came: a stream has no event for an object,
/// and reading the stream reads it out again.
fn replayed(answered: std::result::Result<lm15::Response, Lm15Error>, request: &lm15::Request) -> lm15::testing::LanguageModelStream {
    let answered = answered.map(|mut response| {
        for part in &mut response.message.parts {
            if let Part::Data(data) = part {
                *part = Part::text(data.value.to_string());
            }
        }
        response
    });
    let fake = lm15::testing::FakeLM::default();
    fake.push(answered);
    fake.stream(request)
}

fn unready() -> Lm15Error {
    Lm15Error::TransportError(lm15::ErrorMeta::new(
        "the language model's answer was not ready when the host's transport returned: \
         in WebAssembly a transport answers before it returns",
    ))
}

/// Wait before a request is sent again. A WebAssembly module with no system under it has no
/// clock to wait by, and waiting there is a panic: the request is sent again at once.
fn pause(wait: Duration) {
    if cfg!(not(all(target_family = "wasm", target_os = "unknown"))) {
        std::thread::sleep(wait);
    }
}

/// Whether a failure is a server's that does not stream: a stream with no end, or a request
/// refused for its stream (as a 400 that says "streaming is not supported yet").
fn unstreamed(error: &Lm15Error) -> bool {
    let refused = matches!(
        error.class(),
        ErrorClass::InvalidRequestError | ErrorClass::UnsupportedFeatureError | ErrorClass::CapabilityError
    ) && (error.feature().is_some_and(|feature| feature.contains("stream"))
        || error.message().to_lowercase().contains("stream"));
    refused || error.is_a(ErrorClass::StreamAssemblyError)
}

/// Whether a failure is the schema's: a wire with no place for one, or a server that refused the request it came with.
fn unconstrained(error: &Lm15Error) -> bool {
    let capability = error.is_a(ErrorClass::CapabilityError) && error.feature().is_none_or(|feature| feature == "config.response_format");
    capability || error.class() == ErrorClass::InvalidRequestError
}

/// What lm15 said went wrong, in one line. lm15 never puts a key in it.
fn said(error: &Lm15Error) -> String {
    format!("{}: {}", error.class_name(), error.message())
}

/// A language model's failure as a decision model's: a status where the provider gave one,
/// a connection that failed where the request may pass later, and otherwise a request that cannot be made.
fn model_error(model: &str, error: &Lm15Error) -> systemone::Error {
    if error.provider_code() == Some(UNRECORDED) {
        return systemone::Error::Unrecorded(format!("The language model '{model}' failed: {}", error.message()));
    }
    match error.status() {
        Some(status) => systemone::Error::Api {
            status,
            code: error.provider_code().unwrap_or(error.class_name()).to_owned(),
            message: format!("The language model '{model}' failed: {}", error.message()),
        },
        None if error.is_retryable() => systemone::Error::Transport(format!("The language model '{model}' failed: {}", said(error))),
        None => systemone::Error::Invalid(format!("The language model '{model}' failed: {}", said(error))),
    }
}

/// A message lm15 made, or why it would not.
fn said_by(made: std::result::Result<Message, lm15::ValidationError>) -> std::result::Result<Message, Failed> {
    made.map_err(|error| Failed::Request(error.message))
}

/// As [`said_by`], where the failure is the caller's.
fn message(made: std::result::Result<Message, lm15::ValidationError>) -> Result<Message> {
    made.map_err(|error| Error::Usage(format!("A language model cannot be asked that: {}", error.message)))
}

/// A router for a server that speaks Chat Completions at an address of its own, as the provider `local`.
fn local(base_url: &str) -> Result<lm15::RouterConfig> {
    let compat = lm15::OpenAIChatCompat::preset("lmstudio")
        .cloned()
        .ok_or_else(|| Error::Usage("lm15 has no preset for a local Chat Completions server.".to_owned()))?;
    let provider = lm15::DeclaredProvider::chat("local", base_url, compat).placeholder_key("sk-no-key-required");
    Ok(router().providers([provider]))
}

#[cfg(test)]
mod tests {
    use lm15::testing::{FakeResponse, FakeTransport};

    use super::*;

    #[test]
    fn objects_are_found_whatever_is_round_them() {
        let found = |text: &str| objects_in(text).into_iter().map(Json::Object).collect::<Vec<_>>();
        assert_eq!(found(r#"  {"a": 1}  "#), [json!({"a": 1})]);
        assert_eq!(found("```json\n{\"a\": 1}\n```"), [json!({"a": 1})]);
        assert_eq!(
            found("Here it is: {\"a\": {\"b\": \"}\"}} and no more."),
            [json!({"a": {"b": "}"}})]
        );
        assert_eq!(
            found("<think>{maybe}</think> {\"a\": 1} then {\"b\": 2}"),
            [json!({"a": 1}), json!({"b": 2})]
        );
        assert_eq!(found("no object, [1, 2] and {\"cut\": "), Vec::<Json>::new());
    }

    #[test]
    fn the_wait_doubles_and_a_server_is_obeyed_up_to_a_minute() {
        assert_eq!(wait(0, None), Duration::from_millis(500));
        assert_eq!(wait(1, None), Duration::from_millis(1000));
        assert_eq!(wait(9, None), Duration::from_millis(5000));
        assert_eq!(wait(0, Some(2.0)), Duration::from_secs(2));
        assert_eq!(wait(0, Some(900.0)), Duration::from_secs(60));
        assert_eq!(wait(0, Some(f64::NAN)), Duration::from_millis(500));
    }

    #[test]
    fn a_local_server_is_asked_at_its_own_address_with_no_key_of_anyones() {
        // In WebAssembly a reply is asked for whole, and so it is given.
        let reply = if cfg!(target_family = "wasm") {
            whole()
        } else {
            streamed(&[("content", "An "), ("content", "owl.")])
        };
        let transport = FakeTransport::new([reply]);
        let lm = Lm::with_config("local:qwen3-4b", at(&transport)).unwrap();
        assert_eq!(lm.complete("", "What eats mice at night?").unwrap().text, "An owl.");
        let sent = transport.requests();
        assert_eq!(sent[0].url, "http://localhost:8080/v1/chat/completions");
        assert_eq!(sent[0].header("authorization"), Some("Bearer sk-no-key-required"));
        assert_eq!(sent[0].body.as_ref().unwrap()["model"], "qwen3-4b");
    }

    /// A local server that is asked through `transport`.
    fn at(transport: &FakeTransport) -> lm15::RouterConfig {
        local("http://localhost:8080/v1")
            .unwrap()
            .env([] as [(&str, &str); 0])
            .transport(transport.clone())
    }

    /// A reply in Chat Completions' server-sent events: each piece a chunk of its own, of the
    /// text (`content`) or of the thinking (`reasoning`), then the usage, and the end.
    fn streamed(pieces: &[(&str, &str)]) -> FakeResponse {
        let event = |chunk: Json| format!("data: {chunk}\n\n");
        let mut events: Vec<_> = pieces
            .iter()
            .map(|(kind, text)| event(json!({"id": "c1", "model": "qwen3-4b", "choices": [{"index": 0, "delta": {*kind: text}}]})))
            .collect();
        events.push(event(
            json!({"id": "c1", "model": "qwen3-4b", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ));
        events.push(event(json!({"id": "c1", "model": "qwen3-4b", "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})));
        events.push("data: [DONE]\n\n".to_owned());
        // One chunk of the wire for each event.
        let chunks = events.into_iter().flat_map(|event| FakeResponse::new(200, event).chunks).collect();
        FakeResponse {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            chunks,
        }
    }

    /// What a watch was shown, each time: the reply so far, and the thinking so far.
    type Seen = Arc<Mutex<Vec<(String, String)>>>;

    fn watched(lm: Lm) -> (Lm, Seen) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let watching = seen.clone();
        let lm = lm.watch(move |partial| {
            watching.lock().unwrap().push((partial.text.clone(), partial.thinking.clone()));
            ControlFlow::Continue(())
        });
        (lm, seen)
    }

    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a module asks for a reply whole")]
    fn a_reply_is_asked_for_as_a_stream_and_watched_as_it_comes() {
        let transport = FakeTransport::new([streamed(&[
            ("reasoning", "Owls hunt "),
            ("reasoning", "at night."),
            ("content", "An "),
            ("content", "owl."),
        ])]);
        let (lm, seen) = watched(Lm::with_config("local:qwen3-4b", at(&transport)).unwrap());
        let said = lm.complete("", "What eats mice at night?").unwrap();
        assert_eq!((said.text.as_str(), said.tokens()), ("An owl.", 10));
        let body = transport.requests()[0].body.clone().unwrap();
        assert_eq!(
            (&body["stream"], &body["stream_options"]["include_usage"]),
            (&json!(true), &json!(true))
        );
        let seen = seen.lock().unwrap();
        let text: Vec<&str> = seen.iter().map(|(text, _)| text.as_str()).collect();
        // Shown each piece as it came: the thinking first, where the server shows it, then the reply.
        assert_eq!(text.last(), Some(&"An owl."));
        assert!(text.contains(&"An "), "{seen:?}");
        assert_eq!(seen.last().unwrap().1, "Owls hunt at night.");
    }

    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a module asks for a reply whole")]
    fn a_watch_that_says_stop_stops_the_request_where_it_is() {
        let transport = FakeTransport::new([streamed(&[("content", "An "), ("content", "owl.")])]);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let watching = seen.clone();
        let lm = Lm::with_config("local:qwen3-4b", at(&transport)).unwrap().watch(move |partial| {
            watching.lock().unwrap().push(partial.text.clone());
            ControlFlow::Break(())
        });
        let error = lm.complete("", "What eats mice at night?").unwrap_err();
        assert!(error.is_stopped(), "{error}");
        // Stopped at the first piece, and not asked again.
        assert_eq!(*seen.lock().unwrap(), ["An "]);
        assert_eq!(transport.requests().len(), 1);
    }

    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a module asks for a reply whole")]
    fn a_stopped_branch_stops_its_request_in_the_middle_of_the_reply() {
        let transport = FakeTransport::new([streamed(&[("content", "An "), ("content", "owl "), ("content", "does.")])]);
        let stop = crate::arbitrate::Stop::under(None);
        let mut lm = Lm::with_config("local:qwen3-4b", at(&transport)).unwrap();
        lm.stop = Some(stop.clone());
        // The branch is stopped once the first of the reply has come.
        let lm = lm.watch(move |_| {
            stop.set();
            ControlFlow::Continue(())
        });
        let (said, wrote) = lm.completing("", "What eats mice at night?");
        assert!(said.unwrap_err().is_stopped());
        assert!(wrote.unwrap().response.is_none());
    }

    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a module asks for a reply whole")]
    fn a_server_that_answers_a_stream_with_the_reply_whole_is_asked_for_it_whole() {
        let transport = FakeTransport::new([whole(), whole(), whole()]);
        let lm = Lm::with_config("local:qwen3-4b", at(&transport)).unwrap();
        assert_eq!(lm.complete("", "What eats mice at night?").unwrap().text, "An owl.");
        // Asked for a stream once, then whole, and whole from then on, by a clone too.
        assert_eq!(lm.clone().complete("", "And by day?").unwrap().text, "An owl.");
        let streamed: Vec<bool> = transport
            .requests()
            .iter()
            .map(|sent| sent.body.as_ref().unwrap()["stream"] == true)
            .collect();
        assert_eq!(streamed, [true, false, false]);
    }

    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "a module asks for a reply whole")]
    fn a_server_that_refuses_a_stream_is_asked_for_the_reply_whole() {
        // As a server answers that does not stream yet.
        let refused = FakeResponse::new(400, r#"{"error": "streaming is not supported yet"}"#).header("content-type", "application/json");
        let transport = FakeTransport::new([refused, whole(), whole()]);
        let lm = Lm::with_config("local:qwen3-4b", at(&transport)).unwrap();
        assert_eq!(lm.complete("", "What eats mice at night?").unwrap().text, "An owl.");
        assert_eq!(lm.complete("", "And by day?").unwrap().text, "An owl.");
        let streamed: Vec<bool> = transport
            .requests()
            .iter()
            .map(|sent| sent.body.as_ref().unwrap()["stream"] == true)
            .collect();
        assert_eq!(streamed, [true, false, false]);
    }

    /// A reply whole, as a server that does not stream gives it.
    fn whole() -> FakeResponse {
        FakeResponse::json(&json!({
            "id": "c1", "model": "qwen3-4b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "An owl."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        }))
    }

    #[test]
    fn a_reply_asked_for_whole_is_watched_once_when_it_is_all_there() {
        let transport = FakeTransport::new([whole()]);
        let (lm, seen) = watched(Lm::with_config("local:qwen3-4b", at(&transport)).unwrap().streaming(false));
        assert_eq!(lm.complete("", "What eats mice at night?").unwrap().text, "An owl.");
        assert!(
            transport.requests()[0]
                .body
                .as_ref()
                .unwrap()
                .get("stream")
                .is_none_or(|stream| stream == false)
        );
        assert_eq!(*seen.lock().unwrap(), [("An owl.".to_owned(), String::new())]);
    }
}
