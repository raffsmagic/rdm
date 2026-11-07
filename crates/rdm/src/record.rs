//! A run, written down as it happens.
//!
//! What a run comes to is a function of a few things that cannot be worked out again: what a
//! model answered, and what a tool returned. Everything else, the decisions and the
//! trajectory, follows from those by code. A module given a [`Record`] hands it an [`Event`]
//! for each of them as it happens, and for the places in the run they happened at, so that a
//! host can keep them and work the rest out again.
//!
//! The host keeps them: rdm has no clock and no files. What makes an event durable (when it
//! happened, on which device, its place in the host's log, the version of the host's own code)
//! is the host's to add around it.
//!
//! ```text
//! began     the module, and the tag the call was given
//! opened    a request: where it is in the run, and its inputs where no `asked` follows to show them
//! asked     the request as it was sent, under its key, and what the model answered or failed with
//! started   a tool the answers selected, before it runs: the intent
//! wrote     a language model asked by a tool or a fit, and what it replied (`lm::ReplayLm` replays it)
//! returned  what it returned: the outcome
//! changed   a change a tool made to a map
//! ran       a piece of code a REPL ran for a request: what it printed, and how it ended
//! answered  what a function of the host answered that code: the world, as the code saw it
//! drew      what the code drew at random
//! clocked   the time the code read
//! ended     why the run ended, or what it failed with
//! ```
//!
//! Code that a REPL runs is a function of what it is answered: the same code, given the same
//! answers, the same time and the same randomness, prints the same. So what a REPL hands the
//! record is what comes from outside it (`answered`, `drew`, `clocked`) and, to set beside a replay, how each piece of
//! code ended (`ran`). What a language model wrote for the code (`llm_query`) is a `wrote`, as
//! any other language model's is. A function that asks the module again (`decide`, `choose`)
//! is a sub-call, with events of its own. `Repl::replaying` answers from these events.
//!
//! A decision is not an event: it is made from the evidence an `asked` holds and the numbers
//! it names (`params`), by [`systemone::decide`]. Changing a threshold decides again from the
//! same events, and asks nothing.
//!
//! A host need not keep all of an event. What a request showed (`opened.inputs`,
//! `asked.request`, `wrote.request`) and what a tool returned (`returned.observation`) can be
//! left out where they should not be kept, and the event still reads back: the key, the model
//! and the answer are what [`replay`] needs, for the same model asked the same thing. Only a
//! new model, asked what the old one was, needs the request. So too for a REPL: the code that
//! ran and what it printed (`ran.code`, `ran.printed`), and what a function of the host was
//! given and answered (`answered.args`, `answered.value`). An answer kept without what it was
//! given is replayed to the function's calls in order; one not kept at all is not replayed.
//!
//! A record is called on the thread the event happened on, before the run goes on. An error
//! from it ends the run with that error: a host that cannot keep a `started` can refuse to let
//! the tool run, and a host that wants a run to stop can stop it at its next event.
//!
//! ```
//! use std::sync::{Arc, Mutex};
//!
//! use rdm::record::{Event, replay};
//! use rdm::{Rdm, Signature, Tool, inputs};
//! use systemone::{Answer, Scripted};
//!
//! let events = Arc::new(Mutex::new(Vec::new()));
//! let kept = events.clone();
//! let triage = Signature::new("Triage a ticket.").input("ticket").noul("refund", "Is `inputs.ticket` asking for a refund?");
//! let refund = Tool::new("refund", "Refund the order.", |_| Ok("refunded"));
//! let model = Scripted::new(|_| [("refund", Answer::noul(0.93))]).named("jev");
//! let rdm = Rdm::new(triage).tool(refund).model(model).record(move |event: &Event| {
//!     kept.lock().unwrap().push(event.clone());
//!     Ok(())
//! });
//!
//! let live = rdm.call(inputs! { "ticket" => "I was charged twice." }).unwrap();
//! let names: Vec<&str> = events.lock().unwrap().iter().map(Event::name).collect();
//! assert_eq!(names, ["began", "opened", "asked", "started", "returned", "ended"]);
//!
//! // The same run again, with every answer taken from what was kept and none asked.
//! rdm.set_model(replay("jev", events.lock().unwrap().iter()));
//! let again = rdm.call(inputs! { "ticket" => "I was charged twice." }).unwrap();
//! assert_eq!(again.trajectory, live.trajectory);
//! ```

use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use systemone::{IMAGE_URL, Image, Json, Params, Replay, Request, Response};

use crate::error::Result;
use crate::prediction::Reason;

/// The version of the shape of an [`Event`]. A host writes it beside the events it keeps, and
/// reads a log of another version as that version's.
///
/// 2: `ran`, `answered`, `drew` and `clocked`, what a REPL hands the record. The events of 1
/// are as they were.
pub const SCHEMA: u32 = 2;

/// One thing that happened in a run.
///
/// Every event names its run, and every event but `began` and `ended` the request it happened
/// at, by `index`: its place in the run's trajectory. `parent` is an index of the same run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "an event is made once and handed over by reference: the size of the largest costs nothing"
)]
pub enum Event {
    /// A run began.
    Began {
        /// The run's number: unique in the process, so runs at once (`Rdm::batch`) are told apart.
        /// A host that keeps it past the process maps it to a name of its own.
        run: u64,
        /// What the call was given to be known by ([`crate::Ask::tag`]): the host's own context. Null if none.
        tag: Json,
        /// The module that started it, as it was set.
        module: Module,
    },
    /// A request of the run, at the moment it is asked, or is known not to be.
    Opened {
        /// The run.
        run: u64,
        /// Its place in the trajectory.
        index: usize,
        /// The request whose tool made it, or the turn before it.
        parent: Option<usize>,
        /// How many calls are above it.
        depth: usize,
        /// Which turn of a pointer's loop it is.
        iteration: usize,
        /// The tool that led to it.
        tool: Option<String>,
        /// What it showed, after any fit. Absent when an `asked` follows, which shows it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inputs: Option<Json>,
        /// Whether it had been asked on the way to it, and was answered from then.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        repeat: bool,
        /// The limit that kept it from being asked.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skipped: Option<Reason>,
    },
    /// A request was put to the decision model.
    Asked {
        /// The run.
        run: u64,
        /// The request it was.
        index: usize,
        /// What it is kept under: [`systemone::key`] of the request, for the model asked.
        key: String,
        /// The model it was put to, by its name.
        model: String,
        /// The request as it was sent. Absent where the host kept the evidence and not what it
        /// was about: the key is enough to replay it to the same model, and nothing can be put
        /// to another without it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request: Option<Request>,
        /// What the model answered: the evidence. Absent when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response: Option<Response>,
        /// What it failed with, when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<systemone::Error>,
        /// The numbers each output was decided with, where they are not the defaults.
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        params: IndexMap<String, Params>,
    },
    /// A tool the answers selected is about to run.
    Started {
        /// The run.
        run: u64,
        /// The request whose answers selected it.
        index: usize,
        /// The output that selected it.
        output: String,
        /// The tool.
        tool: String,
    },
    /// A tool ran.
    Returned {
        /// The run.
        run: u64,
        /// The request whose answers selected it.
        index: usize,
        /// The output that selected it.
        output: String,
        /// The tool.
        tool: String,
        /// What it returned, or the failure it came to, as the next request is shown it. Null
        /// where the host did not keep it, as for a tool that returned nothing.
        #[serde(default)]
        observation: Json,
    },
    /// A tool still running said what it has so far ([`crate::Call::partial`]): what a ruling
    /// may use before it comes back. `version` counts what it said, from 1.
    Partial {
        /// The run.
        run: u64,
        /// The request whose answers selected it.
        index: usize,
        /// The output that selects it.
        output: String,
        /// The tool.
        tool: String,
        /// Which time it said so.
        version: usize,
        /// What it has so far.
        partial: Json,
    },
    /// Something is about to be done outside ([`crate::Call::acting`]): the intent, kept before
    /// it is done. A record that refuses it keeps it from being done.
    Acting {
        /// The run.
        run: u64,
        /// The request whose tool is to do it.
        index: usize,
        /// The tool. None for a fit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
    },
    /// Something was done outside, and this is what it came to ([`crate::Call::acting`]): kept
    /// the moment it was done, whatever becomes of the branch after.
    Acted {
        /// The run.
        run: u64,
        /// The request whose tool did it.
        index: usize,
        /// The tool. None for a fit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        /// What it came to.
        receipt: Json,
    },
    /// Arbitration ruled on a request's tools as they were: after one or more came back, or,
    /// at round 0, before any started ([`crate::arbitrate`]). The decision request a ruling
    /// asked, where it asked one, is a request of its own, under this one.
    Arbitrated {
        /// The run.
        run: u64,
        /// The request whose tools it ruled on.
        index: usize,
        /// How many rulings on them came before it.
        round: usize,
        /// What it ruled.
        ruling: crate::arbitrate::Ruling,
        /// What it was shown of each branch that had something to show: `[output, version]`,
        /// the version of what it had so far (as its `partial` events number them), or 0 for
        /// what it came back with. A `partial` event is recorded after the ledger has it, so
        /// the order of events alone may not tell which version a ruling saw; this does.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        saw: Vec<(String, usize)>,
        /// What the arbiter failed with, when it failed: the ruling is then that of no arbiter,
        /// unless one made before stands in its place (a `revived` event follows).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A round's ruling could not be made, and the ruling made in round `from`, set aside then,
    /// stands in its place: the round's `arbitrated` event, just before, holds that ruling and
    /// what it saw then, not what the branches have come to since ([`crate::Arbiter::revived`]).
    Revived {
        /// The run.
        run: u64,
        /// The request whose tools it ruled on.
        index: usize,
        /// The round whose ruling could not be made.
        round: usize,
        /// The round the ruling that stands was made in.
        from: usize,
    },
    /// A tool arbitration no longer wanted was stopped: nothing more is begun for it. One that
    /// had started may still come back, and its `returned` is recorded when it does; what it
    /// did before it was stopped stands.
    Stopped {
        /// The run.
        run: u64,
        /// The request whose answers selected it, or whose arbitration started it.
        index: usize,
        /// The output that selects it.
        output: String,
        /// The tool.
        tool: String,
        /// Whether it had started. One that had not is never run.
        started: bool,
    },
    /// A language model wrote, for a tool or a fit of a request: one event for each time it was
    /// asked, a reply that could not be used and the one asked for after it alike.
    Wrote {
        /// The run.
        run: u64,
        /// The request whose tool or fit asked it.
        index: usize,
        /// The tool that asked it. None for a fit.
        tool: Option<String>,
        /// What it is kept under: `lm::key` of the request.
        key: String,
        /// The model asked for.
        model: String,
        /// The request as it was sent, in lm15's own JSON. Absent where the host did not keep it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request: Option<Json>,
        /// What the model replied, in lm15's own JSON. Absent when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response: Option<Json>,
        /// What it failed with, when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A REPL ran a piece of code for a request: the code of a step, or a written function the
    /// request pointed at, as it was run.
    Ran {
        /// The run.
        run: u64,
        /// The request it was run for.
        index: usize,
        /// The REPL, by its name.
        repl: String,
        /// The code. Absent where the host did not keep it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        /// What it printed, until it ended. Absent where the host did not keep it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        printed: Option<String>,
        /// How it ended.
        #[serde(flatten)]
        ending: Ending,
    },
    /// A function of the host answered code a REPL was running for a request: a function given
    /// as a closure over its arguments, or a tool. What it returned is the world as the code saw
    /// it, and cannot be worked out again.
    Answered {
        /// The run.
        run: u64,
        /// The request the code was run for.
        index: usize,
        /// The REPL, by its name.
        repl: String,
        /// The function, by its name.
        function: String,
        /// What it was given, by the names it declared: `*` holds what was given beyond them.
        /// Absent where the host did not keep it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<Json>,
        /// What it returned: null for a function that returned nothing. Absent when it failed,
        /// and where the host did not keep it.
        #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
        value: Option<Json>,
        /// What it failed with, as the code was told, when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Code a REPL was running for a request drew at random: `random` seeding itself, or
    /// `os.urandom`. What the REPL's source gave it, in hex.
    Drew {
        /// The run.
        run: u64,
        /// The request the code was run for.
        index: usize,
        /// The REPL, by its name.
        repl: String,
        /// The bytes, two hex digits each.
        bytes: String,
    },
    /// Code a REPL was running for a request read the clock: `time.time()`, a clock `time`
    /// reckons from it, `datetime.now()` or `date.today()`, each read once.
    Clocked {
        /// The run.
        run: u64,
        /// The request the code was run for.
        index: usize,
        /// The REPL, by its name.
        repl: String,
        /// The instant the REPL's clock gave, in seconds since the Unix epoch.
        seconds: f64,
    },
    /// A tool changed a map ([`crate::Call::changed`]).
    Changed {
        /// The run.
        run: u64,
        /// The request whose tool changed it.
        index: usize,
        /// The change, as the tool described it.
        change: Json,
    },
    /// The run ended.
    Ended {
        /// The run.
        run: u64,
        /// Why, when it came to a prediction.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<Reason>,
        /// What it failed with, when it failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

impl Event {
    /// What kind of event it is, as it is written: `began`, `opened`, `asked` and so on.
    pub fn name(&self) -> &'static str {
        match self {
            Event::Began { .. } => "began",
            Event::Opened { .. } => "opened",
            Event::Asked { .. } => "asked",
            Event::Started { .. } => "started",
            Event::Returned { .. } => "returned",
            Event::Wrote { .. } => "wrote",
            Event::Ran { .. } => "ran",
            Event::Answered { .. } => "answered",
            Event::Drew { .. } => "drew",
            Event::Clocked { .. } => "clocked",
            Event::Changed { .. } => "changed",
            Event::Arbitrated { .. } => "arbitrated",
            Event::Revived { .. } => "revived",
            Event::Acting { .. } => "acting",
            Event::Acted { .. } => "acted",
            Event::Partial { .. } => "partial",
            Event::Stopped { .. } => "stopped",
            Event::Ended { .. } => "ended",
        }
    }

    /// The same event with each picture it holds in place of its bytes: `{"type": "image_ref",
    /// "sha256": ..., "content_type": ...}` ([`systemone::Image::reference`]), for a host that
    /// keeps the record and not the pictures. A module hands its record every picture whole;
    /// this is for the host to call, so that every host strips them the same way.
    ///
    /// A picture is any value, at any depth, that is one as [`crate::image`] makes it, in what
    /// a request showed (`asked.request.state`, `opened.inputs`) and in what a tool returned
    /// (`returned.observation`): a decision model may read a picture wherever it is, so one
    /// in an input that is not a picture, or deep in an object, is stripped too. So are the
    /// image parts a language model was sent (lm15's, in `wrote.request`). The key stays, so a
    /// run kept this way replays by key, as one kept without what it showed does.
    ///
    /// ```
    /// use rdm::record::Event;
    /// use serde_json::json;
    ///
    /// let request = rdm::Signature::new("Look.")
    ///     .image("window", "")
    ///     .noul("editor", "Is `inputs.window` an editor?")
    ///     .request(json!({"window": rdm::image("image/png", b"")}).as_object().unwrap())
    ///     .unwrap();
    /// let asked = Event::Asked { run: 1, index: 0, key: "k".into(), model: "m".into(), request: Some(request), response: None, error: None, params: Default::default() };
    /// let Event::Asked { request: Some(kept), key, .. } = asked.without_images() else { unreachable!() };
    /// assert_eq!(kept.state[1]["type"], "image_ref");
    /// assert_eq!(kept.state[1]["sha256"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    /// assert_eq!(key, "k");
    /// ```
    pub fn without_images(&self) -> Event {
        let mut event = self.clone();
        match &mut event {
            Event::Opened { inputs: Some(inputs), .. } => referred(inputs),
            Event::Asked {
                request: Some(request), ..
            } => referred(&mut request.state),
            Event::Returned { observation, .. } => referred(observation),
            Event::Acted { receipt, .. } => referred(receipt),
            Event::Partial { partial, .. } => referred(partial),
            Event::Wrote {
                request: Some(request), ..
            } => written(request),
            _ => {}
        }
        event
    }

    /// The run it is of.
    pub fn run(&self) -> u64 {
        match self {
            Event::Began { run, .. }
            | Event::Opened { run, .. }
            | Event::Asked { run, .. }
            | Event::Started { run, .. }
            | Event::Returned { run, .. }
            | Event::Wrote { run, .. }
            | Event::Ran { run, .. }
            | Event::Answered { run, .. }
            | Event::Drew { run, .. }
            | Event::Clocked { run, .. }
            | Event::Changed { run, .. }
            | Event::Arbitrated { run, .. }
            | Event::Revived { run, .. }
            | Event::Acting { run, .. }
            | Event::Acted { run, .. }
            | Event::Partial { run, .. }
            | Event::Stopped { run, .. }
            | Event::Ended { run, .. } => *run,
        }
    }
}

/// How code that was run came to its end, as data: what a replay of a run sets beside what
/// the run recorded, so that a step that ended another way is seen to have.
///
/// Every ending but one is a matter of the code and of what it was answered: the same code,
/// given the same answers, ends the same way. [`Ending::TimedOut`] is the exception. The limit
/// on time is the one clock a REPL has, and code that ran out of time on one machine may not
/// on another.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "ended", rename_all = "snake_case")]
pub enum Ending {
    /// It ran to its end.
    Ran,
    /// A function that ends the code was called (`Repl::ending`).
    Finished,
    /// It raised, and nothing caught it: the error, in the line it is shown in.
    Raised {
        /// What it raised, and where.
        error: String,
    },
    /// It ran out of the time it is given. The interpreter is made again.
    TimedOut,
    /// It held or printed more than it may. The interpreter is made again.
    OutOfMemory {
        /// What it was refused.
        error: String,
    },
    /// The run it was part of is over: a function of the host said so.
    Ended {
        /// What the host said.
        error: String,
    },
}

/// Every picture in a value, at any depth, as what stands for it. Anything else is left as it is.
fn referred(value: &mut Json) {
    if let Some(reference) = Image::of(value).map(|picture| picture.reference()) {
        *value = reference;
        return;
    }
    match value {
        Json::Array(items) => items.iter_mut().for_each(referred),
        Json::Object(object) => object.values_mut().for_each(referred),
        _ => {}
    }
}

/// Every image part of a request in lm15's JSON (`{"type": "image", "media_type", "data"}`), as what stands for it.
fn written(value: &mut Json) {
    match value {
        Json::Array(items) => items.iter_mut().for_each(written),
        Json::Object(object) => {
            let image = object.get("type").and_then(Json::as_str) == Some("image");
            let content_type = object.get("media_type").and_then(Json::as_str).unwrap_or_default().to_owned();
            match object.get("data").and_then(Json::as_str) {
                Some(data) if image => {
                    // lm15 takes the bytes in base64, or as a data URL.
                    let url = if data.starts_with("data:") {
                        data.to_owned()
                    } else {
                        format!("data:{content_type};base64,{data}")
                    };
                    let shaped = serde_json::json!({"type": IMAGE_URL, IMAGE_URL: {"url": url}});
                    if let Some(reference) = Image::of(&shaped).map(|picture| picture.reference()) {
                        *value = reference;
                    }
                }
                _ => object.values_mut().for_each(written),
            }
        }
        _ => {}
    }
}

/// A value that is there, null or not: absent is the field left out, and not a null.
fn present<'de, D: serde::Deserializer<'de>>(written: D) -> std::result::Result<Option<Json>, D::Error> {
    Json::deserialize(written).map(Some)
}

/// A module as it was set when a run began: enough to tell a changed module from a changed model.
///
/// What a tool or a fit does is code, and is not here: a host that changes it says so with a
/// version of its own.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Module {
    /// The signature's instructions.
    pub instructions: String,
    /// The tools on its map, each as `[name, description]`, before any source was read.
    pub tools: Vec<(String, String)>,
    /// The names of its sources.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
    /// How many fits it has.
    #[serde(default)]
    pub fits: usize,
    /// Whether `finish` is offered.
    pub finish: bool,
    /// The names always offered with `finish`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub always: Vec<String>,
    /// Its `max_iters`.
    pub max_iters: usize,
    /// Its `max_depth`.
    pub max_depth: usize,
    /// Its `max_predictor_calls`.
    pub max_predictor_calls: usize,
    /// Its `max_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

/// What a run's events are handed to: the host's.
pub trait Record: Send + Sync {
    /// Keep one event. An error ends the run with it.
    fn record(&self, event: &Event) -> Result<()>;
}

impl<F> Record for F
where
    F: Fn(&Event) -> Result<()> + Send + Sync,
{
    fn record(&self, event: &Event) -> Result<()> {
        self(event)
    }
}

/// A record shared by the modules that hold it.
pub(crate) type Shared = Arc<dyn Record>;

/// A decision model that answers every request the events say `model` was asked, as it was
/// answered then, and refuses any other ([`systemone::Error::Unrecorded`]).
pub fn replay<'a>(model: &str, events: impl IntoIterator<Item = &'a Event>) -> Replay {
    let mut replay = Replay::new(model);
    for event in events {
        if let Event::Asked {
            model: asked,
            key,
            response,
            error,
            ..
        } = event
            && asked == model
        {
            let answered = match (response, error) {
                (Some(response), _) => Ok(response.clone()),
                (None, Some(error)) => Err(error.clone()),
                (None, None) => continue,
            };
            replay.keep_key(key.clone(), answered);
        }
    }
    replay
}
