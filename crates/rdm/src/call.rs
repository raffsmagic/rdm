//! The request a tool is running for, and what a tool can do through it.

use std::fmt;
use std::sync::Arc;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Map;
use systemone::{Decision, Json};

use crate::error::{Error, Result, listed, sorted};
use crate::inputs::{Input, Inputs};
use crate::module::{Active, Rdm};
use crate::prediction::{Entry, Prediction};
use crate::record::Event;
use crate::signature::Signature;
use crate::tool::Tools;

/// A call as it is asked for: the inputs, and what may be said beside them.
///
/// ```
/// use rdm::{Ask, Signature, inputs};
///
/// let judge = Signature::new("Judge one record against a question.")
///     .input("record")
///     .input("question")
///     .noul("answers", "Does `inputs.record` answer `inputs.question`?");
/// // Other questions than the module's own, and the answers only: no tool is run for it.
/// let ask = Ask::new(inputs! { "record" => "The hangar opens at dawn.", "question" => "When?" }).signature(judge).max_depth(0);
/// # let _ = ask;
/// ```
#[derive(Clone, Debug, Default)]
pub struct Ask {
    pub(crate) inputs: Inputs,
    pub(crate) signature: Option<Signature>,
    pub(crate) max_depth: Option<usize>,
    pub(crate) state: Option<Json>,
    pub(crate) tag: Option<Json>,
    pub(crate) stop: Option<Arc<crate::arbitrate::Stop>>,
}

impl Ask {
    /// A call with these inputs, asked the module's own questions.
    pub fn new(inputs: Inputs) -> Self {
        Self { inputs, ..Self::default() }
    }

    /// Questions to put in place of the module's own, for this call. The module's tools
    /// still apply, by name.
    pub fn signature(mut self, signature: Signature) -> Self {
        self.signature = Some(signature);
        self
    }

    /// How many levels below this call may still run tools, when that is fewer than the run
    /// allows. 0 asks for the answers only: no tool is run for it, and that is not a limit reached.
    ///
    /// It is in the place of the module's own `max_depth` for this call, and for every turn of
    /// the call's loop alike. The run a call is in still bounds it: the call that starts a run
    /// cannot go past the module's own limit, nor a sub-call past the limit of the call above.
    /// Another module asked from inside a tool ([`Call::on`]) counts its depth from there, so
    /// for it this may be more than its own as well as fewer. (The Python module keeps what the
    /// call asked for at the first turn of such a loop, and goes back to the module's own for
    /// the turns after it. Here a call has one limit.)
    pub fn max_depth(mut self, levels: usize) -> Self {
        self.max_depth = Some(levels);
        self
    }

    /// The answers only: no tool is run for it.
    pub fn answers_only(self) -> Self {
        self.max_depth(0)
    }

    /// What the tools of this call, and of the calls under it, can read and no request shows.
    pub fn state(mut self, state: impl Into<Json>) -> Self {
        self.state = Some(state.into());
        self
    }

    /// What a run this call starts is known by to the host that records it: the `tag` of its
    /// `began` event ([`crate::record`]). Nothing reads it but the host. A sub-call's is not used.
    pub fn tag(mut self, tag: impl Into<Json>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// A run that `cancel` can stop from outside it ([`Cancel`]). Only the call that starts a
    /// run reads it: a sub-call is stopped with the run it is in.
    pub fn stoppable(mut self, cancel: &Cancel) -> Self {
        self.stop = Some(cancel.0.clone());
        self
    }
}

/// What stops a run from outside it, once it is stopped ([`Ask::stoppable`]): the run begins no
/// more requests, its tools see their branch stopped ([`Call::stopped`]), nothing more outside
/// is begun for it ([`Call::acting`]), and its sub-calls and the modules its tools call fail
/// with [`Error::Stopped`]. What was begun before stands, and something outside in the middle
/// of being done is let finish: a stop and its start take it in turn.
///
/// ```
/// use rdm::{Ask, Cancel, Rdm, Signature, inputs};
/// use systemone::{Answer, Scripted};
///
/// let cancel = Cancel::new();
/// cancel.stop();
/// let judge = Rdm::new(Signature::new("Judge.").input("record").noul("holds", "Does it hold?"))
///     .model(Scripted::new(|_| [("holds", Answer::noul(0.9))]));
/// let came = judge.call_with(Ask::new(inputs! { "record" => "It holds." }).stoppable(&cancel));
/// assert!(came.is_err_and(|error| error.is_stopped()));
/// assert!(cancel.is_stopped());
/// ```
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<crate::arbitrate::Stop>);

impl Cancel {
    /// A run not yet stopped.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop the run, once anything outside it is in the middle of doing is done.
    pub fn stop(&self) {
        while !self.0.stop_unless_acting() {
            std::thread::yield_now();
        }
    }

    /// Whether it was stopped.
    pub fn is_stopped(&self) -> bool {
        self.0.is_set()
    }
}

impl From<Inputs> for Ask {
    fn from(inputs: Inputs) -> Self {
        Ask::new(inputs)
    }
}

/// What is running for a request: one of its tools, or a fit.
#[derive(Clone, Debug)]
pub(crate) enum Role {
    Tool(Arc<str>),
    Fit,
    /// The arbiter settling a request's tools ([`crate::arbitrate`]).
    Arbitrate,
}

/// The request that selected a tool: what it showed, what was answered, and the run it is part of.
///
/// A tool reads its arguments from it ([`Call::arg`]), makes sub-calls through it
/// ([`Call::sub`], [`Call::batch`]), and reaches the run through it: the tools on the map,
/// the state no request shows, the trajectory so far. A fit is given one too, for the request
/// it is fitting, with no answers yet.
///
/// The module is on the call, so a tool needs no reference of its own to recurse:
///
/// ```
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let count = Signature::new("Count down.").input_typed("n", "int").noul("again", "Is `inputs.n` more than zero?");
/// let again = Tool::new("again", "Asks about the number below.", |call| {
///     let n: i64 = call.arg("n")?;
///     let below = call.sub(inputs! { "n" => n - 1 })?; // one level down, in the same run
///     Ok(1 + below.observations.get("again").and_then(|seen| seen.as_i64()).unwrap_or(0))
/// });
/// let model = Scripted::new(|request| [("again", Answer::noul(if request.state["inputs"]["n"].as_i64() > Some(0) { 0.9 } else { 0.1 }))]);
///
/// let result = Rdm::new(count).tool(again).max_depth(5).model(model).call(inputs! { "n" => 2 }).unwrap();
/// assert_eq!(result.observations["again"], 2);
/// assert_eq!(result.trajectory.iter().map(|request| request.depth).collect::<Vec<_>>(), [0, 1, 2]);
/// ```
#[derive(Clone)]
pub struct Call {
    pub(crate) active: Active,
    /// The module sub-calls are put to: the one that is running, unless [`Call::on`] named another.
    pub(crate) rdm: Rdm,
    pub(crate) prediction: Option<Arc<Prediction>>,
    pub(crate) inputs: Inputs,
    pub(crate) shown: Arc<Map<String, Json>>,
    pub(crate) iteration: usize,
    pub(crate) role: Role,
}

impl Call {
    /// An argument, by name: the answer of that name as a plain value (a bool, a label, a
    /// number; null for choosing none), or else the input of that name as the request showed
    /// it. A name that is neither is a usage error, so a misspelt one never quietly receives
    /// something else.
    ///
    /// An output that offers [`crate::Opt::none`] reads as an `Option`: `None` when the model
    /// chose none of the others. What the model answered is never how the module was used, so
    /// an answer read as a type it is not is the tool's own failure ([`Error::Tool`]), which
    /// becomes its observation and leaves the run going. An input read as a type it is not is
    /// a usage error, as a misspelt name is: the inputs are what the code gave.
    ///
    /// ```
    /// use rdm::{Rdm, Signature, Tool, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let act = Signature::new("Operate a smart home.")
    ///     .input("task")
    ///     .noul("switch", "Does `inputs.task` ask for a device to be switched?")
    ///     .choice_or_none("device", "Which device does `inputs.task` mean?", ["tv", "lights"], "It names no device.");
    /// let switch = Tool::new("switch", "Switch a device.", |call| {
    ///     let device: Option<String> = call.arg("device")?;
    ///     Ok(device.map_or("nothing to switch".to_owned(), |device| format!("{device}: switched")))
    /// });
    /// let model = Scripted::new(|_| [("switch", Answer::noul(0.9)), ("device", Answer::chose("None", ["tv", "lights", "None"]))]);
    /// let result = Rdm::new(act).tool(switch).model(model).call(inputs! { "task" => "Switch it." }).unwrap();
    /// assert_eq!(result.observations["switch"], "nothing to switch");
    /// ```
    pub fn arg<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        if let Some(answer) = self.answer_value(name) {
            let none = if answer.is_null() {
                " It chose none of its options: read it as an `Option`."
            } else {
                ""
            };
            return serde_json::from_value(answer)
                .map_err(|error| Error::Tool(format!("The answer '{name}' is read as a type it is not: {error}.{none}")));
        }
        let value = self.shown.get(name).cloned().ok_or_else(|| self.takes(name))?;
        serde_json::from_value(value).map_err(|error| Error::Usage(format!("{} reads '{name}' as a type it is not: {error}.", self.who())))
    }

    /// An answer of the request as a plain value, if it has one of that name.
    pub(crate) fn answer_value(&self, name: &str) -> Option<Json> {
        let prediction = self.prediction.as_ref()?;
        prediction.answers.contains_key(name).then(|| prediction.value(name))
    }

    /// The names an argument can be read by: the request's answers and its inputs.
    pub fn names(&self) -> Vec<String> {
        let answers = self.prediction.iter().flat_map(|prediction| prediction.answers.keys().cloned());
        answers.chain(self.shown.keys().cloned()).collect()
    }

    fn who(&self) -> String {
        match &self.role {
            Role::Tool(name) => format!("Tool '{name}'"),
            Role::Fit => "The fit".to_owned(),
            Role::Arbitrate => "The arbiter".to_owned(),
        }
    }

    fn takes(&self, name: &str) -> Error {
        Error::Usage(format!(
            "{} takes {}, which name no answer or input of the request. An argument is read by name from {}.",
            self.who(),
            listed([name]),
            sorted(self.names())
        ))
    }

    /// An answer of the request, with its probabilities.
    pub fn answer(&self, name: &str) -> Option<&Decision> {
        self.prediction.as_ref().and_then(|prediction| prediction.get(name))
    }

    /// The answers that selected the tool, each with its probabilities. None for a fit: the request is not answered yet.
    pub fn prediction(&self) -> Option<&Prediction> {
        self.prediction.as_deref()
    }

    /// The inputs of the request, whole and as they were given: a live one is still live, so
    /// it can be passed on to a sub-call. A fit changes what the model is shown, not these.
    pub fn inputs(&self) -> &Inputs {
        &self.inputs
    }

    /// One input as it was given, to pass on.
    pub fn input(&self, name: &str) -> Result<Input> {
        self.inputs.get(name).cloned().ok_or_else(|| self.takes(name))
    }

    /// What the request showed, as it was when it was recorded.
    pub fn shown(&self) -> &Map<String, Json> {
        &self.shown
    }

    /// What the run was given as state: what tools read and no request shows. It is that of
    /// the nearest call above that gave one.
    pub fn state(&self) -> &Json {
        &self.active.state
    }

    /// The state read as a type of the tool's own, which may borrow from it: a struct with a
    /// field for each key it reads, say. The state is the caller's and not the model's, so one
    /// that is not of that type is a usage error.
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use rdm::{Ask, Rdm, Signature, Tool, inputs};
    /// use serde_json::json;
    /// use systemone::{Answer, Scripted};
    ///
    /// #[derive(serde::Deserialize)]
    /// struct Handbook<'a> {
    ///     #[serde(borrow)]
    ///     outline: HashMap<&'a str, Vec<&'a str>>,
    /// }
    ///
    /// let explore = Tool::new("explore", "List what is under an entry.", |call| {
    ///     let (entry, handbook): (String, Handbook) = (call.arg("entry")?, call.state_as()?);
    ///     Ok(handbook.outline.get(entry.as_str()).map_or(0, Vec::len))
    /// });
    /// let navigate = Signature::new("Search a handbook.").input("entry").noul("explore", "Does `inputs.entry` name the topic?");
    /// let rdm = Rdm::new(navigate).tool(explore).model(Scripted::new(|_| [("explore", Answer::noul(0.9))]));
    /// let ask = Ask::new(inputs! { "entry" => "Time off" }).state(json!({"outline": {"Time off": ["Annual leave", "Sick leave"]}}));
    /// assert_eq!(rdm.call_with(ask).unwrap().observations["explore"], 2);
    /// ```
    pub fn state_as<'a, T: Deserialize<'a>>(&'a self) -> Result<T> {
        T::deserialize(&*self.active.state)
            .map_err(|error| Error::Usage(format!("{} reads the state as a type it is not: {error}.", self.who())))
    }

    /// The module's tools in this run: a map of the run's own. A tool put here is on offer
    /// from the run's next request, and another run is not offered it.
    pub fn tools(&self) -> Tools {
        self.rdm.run_tools(&self.active.run)
    }

    /// Run a tool that is on the run's map, for this same request, and return what it returned.
    pub fn run(&self, name: &str) -> Result<Json> {
        // A stopped branch begins no tool, by its answers or by name.
        self.go_on()?;
        let tool = self
            .tools()
            .get(name)
            .ok_or_else(|| Error::Usage(format!("No tool named '{name}' is on the map.")))?;
        tool.call(self)
    }

    /// Every request of the run so far.
    pub fn trajectory(&self) -> Vec<Entry> {
        self.active.run.lock().trajectory.clone()
    }

    /// How many requests the run has recorded so far.
    pub fn requests(&self) -> usize {
        self.active.run.lock().trajectory.len()
    }

    /// Where in the trajectory this request is.
    pub fn index(&self) -> usize {
        self.active.index
    }

    /// How many calls are above this one. The first call of a run is at 0.
    pub fn depth(&self) -> usize {
        self.active.depth
    }

    /// Which turn of a pointer's loop this is.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// How many more requests may be asked, under every limit this request is under.
    pub fn requests_left(&self) -> usize {
        let state = self.active.run.lock();
        let left = self.active.budgets.iter().map(|budget| state.budgets[*budget].left());
        left.min().unwrap_or(0)
    }

    /// The tool that is running. None in a fit.
    pub fn tool(&self) -> Option<&str> {
        match &self.role {
            Role::Tool(name) => Some(name),
            Role::Fit | Role::Arbitrate => None,
        }
    }

    /// Whether the branch this tool is, or is under, was stopped by arbitration: what it was
    /// for is settled without it ([`crate::arbitrate`]). The module begins nothing more for a
    /// stopped branch, so a sub-call made now fails with [`Error::Stopped`]; a tool that does
    /// something long without asking the module asks this between its steps, and ends. What it
    /// did before stands. Never, outside a request that is arbitrated.
    pub fn stopped(&self) -> bool {
        self.active.stopped()
    }

    /// Nothing, unless the branch has been stopped ([`Call::stopped`]): then what a stopped
    /// branch ends with, for a tool to return with `?`.
    pub fn go_on(&self) -> Result<()> {
        self.active.go_on()
    }

    /// The other tools of the request this tool is running for, as they are now: each with
    /// where it is and, once it has come back, what it came back with. A tool that works for
    /// a while reads it to build on what the others found, and not find it again. Empty
    /// outside a request that is arbitrated ([`crate::arbitrate`]).
    pub fn siblings(&self) -> Vec<crate::arbitrate::Branch> {
        let Some(branches) = &self.active.branches else { return Vec::new() };
        let mine = self.tool();
        crate::arbitrate::lock(branches)
            .iter()
            .filter(|branch| Some(branch.tool.as_str()) != mine)
            .cloned()
            .collect()
    }

    /// Whether this tool was started by arbitration, after others came back, and not by the
    /// request's answers ([`crate::arbitrate`]). Its request's answers did not select it.
    pub fn arbitrated(&self) -> bool {
        self.active.late
    }

    /// Say what this tool has so far, before it comes back: its request's arbitration is asked
    /// about it, and may use it and stop the tool, or wait for more ([`crate::arbitrate`]).
    /// Each time it is said it replaces what was said before. Nothing, outside a request that
    /// is arbitrated, or once the tool has been stopped or let finish.
    pub fn partial(&self, so_far: impl Into<Json>) {
        let (Some(branches), Some(mine)) = (&self.active.branches, self.tool()) else {
            return;
        };
        let (output, version, partial) = {
            let mut ledger = crate::arbitrate::lock(branches);
            let Some(branch) = ledger
                .iter_mut()
                .find(|branch| branch.tool == mine && branch.state == crate::arbitrate::State::Running)
            else {
                return;
            };
            let partial: Json = so_far.into();
            branch.partial = Some(partial.clone());
            branch.version += 1;
            (branch.output.clone(), branch.version, partial)
        };
        // What it said is the record's, as a result is: which time, and what.
        let (index, tool) = (self.active.index, mine.to_owned());
        let _ = self.active.run.emit(|run| Event::Partial {
            run,
            index,
            output,
            tool,
            version,
            partial,
        });
        if let Some(wake) = &self.active.wake {
            wake();
        }
    }

    /// Do something outside, a command run or a message sent, that cannot be taken back once
    /// begun: while it is being done the branch is not stopped, and its request waits for what
    /// it comes to. What it comes to is the run's receipt the moment it is done, before anything
    /// else may stop the branch: it is in the record (`acted`) and in the prediction of the call
    /// that started the run ([`crate::Prediction::receipts`]), whatever becomes of the branch
    /// after, so that what was done is never left unsaid. It is not begun for a branch stopped
    /// already (then `Err`, what a stopped branch ends with).
    pub fn acting<T: serde::Serialize>(&self, does: impl FnOnce() -> T) -> Result<T> {
        let Some(stop) = &self.active.stop else {
            self.intent()?;
            let done = does();
            self.receipt(&done);
            return Ok(done);
        };
        /// The mark taken off however `does` ends.
        struct Done<'a>(&'a crate::arbitrate::Stop);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.end_acting();
            }
        }
        // A stop that came first wins: nothing is begun.
        if !stop.begin_acting() {
            return Err(crate::module::stopped());
        }
        let _done = Done(stop);
        self.intent()?;
        let done = does();
        // Kept while the branch is still held from being stopped.
        self.receipt(&done);
        Ok(done)
    }

    /// Keep that something is about to be done outside: a record that refuses it, or refused
    /// anything before it, keeps it from being done, as it keeps a tool from running.
    fn intent(&self) -> Result<()> {
        let (index, tool) = (self.active.index, self.tool().map(str::to_owned));
        self.active.run.emit(|run| Event::Acting { run, index, tool })
    }

    /// Keep what something done outside came to, as the run's receipt.
    fn receipt(&self, done: &impl serde::Serialize) {
        let receipt = crate::prediction::Receipt {
            index: self.active.index,
            tool: self.tool().map(str::to_owned),
            // Done all the same: one that cannot be written as JSON says so and why, never null.
            receipt: serde_json::to_value(done).unwrap_or_else(|error| serde_json::json!({ "unrecorded": error.to_string() })),
        };
        self.active
            .run
            .receipts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(receipt.clone());
        // A refusal is kept by the run, and is what its next event ends it with.
        let _ = self.active.run.emit(|run| Event::Acted {
            run,
            index: receipt.index,
            tool: receipt.tool,
            receipt: receipt.receipt,
        });
    }

    /// Whether the branch this tool is, or is under, was let finish unwaited by arbitration:
    /// what it comes to is recorded and held against what its request came to, and nothing it
    /// does is to act outside. The module cannot see what a tool does outside: a host whose
    /// tools act there refuses to, for a call that is shadowed. Never, outside a request that
    /// is arbitrated.
    pub fn shadowed(&self) -> bool {
        self.active.stop.as_ref().is_some_and(|stop| stop.is_shadowed())
    }

    /// The module: the one whose request this is, or the one [`Call::on`] named.
    pub fn rdm(&self) -> &Rdm {
        &self.rdm
    }

    /// The same request, with sub-calls put to another module. That module joins the run: its
    /// requests are in the same trajectory, one level down, under the run's limits and its
    /// own, to a depth counted from here.
    pub fn on(&self, module: &Rdm) -> Call {
        Call {
            rdm: module.clone(),
            ..self.clone()
        }
    }

    /// A sub-call: the module's own questions about other inputs, one level down, in the same
    /// run. Its tools run as any call's do, and what it returns comes back here.
    pub fn sub(&self, inputs: Inputs) -> Result<Prediction> {
        self.sub_with(Ask::new(inputs))
    }

    /// A sub-call that says more than its inputs: other questions, the answers only, a state of its own.
    pub fn sub_with(&self, ask: impl Into<Ask>) -> Result<Prediction> {
        let mut predictions = self.rdm.children(&self.active, vec![ask.into()])?;
        predictions
            .pop()
            .ok_or_else(|| Error::Usage("A sub-call returned nothing.".to_owned()))
    }

    /// Several sub-calls at once. They are admitted in the order given, so which ones a limit
    /// stops does not depend on timing, then asked concurrently. Their predictions come back
    /// in the same order; one a limit stopped has no answers and that limit as its reason.
    ///
    /// Each shows its inputs as they were when the batch was made: a live input is read once
    /// for every member, before any of them is asked. A member asked after another's tool has
    /// changed what the input shows, as the later ones of a batch wider than the module's
    /// `threads` are, is still asked about the state as it was then, and that is what its
    /// entry in the trajectory says it showed. A tool that wants each to see what the one
    /// before it did makes them one at a time, with [`Call::sub`].
    pub fn batch<A: Into<Ask>>(&self, asks: impl IntoIterator<Item = A>) -> Result<Vec<Prediction>> {
        self.rdm.children(&self.active, asks.into_iter().map(Into::into).collect())
    }

    /// Report tokens this tool, or this fit, used on a model of its own. They are counted
    /// against the run's limit, and recorded on the request apart from its own.
    pub fn spend(&self, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let mut state = self.active.run.lock();
        for budget in self.active.budgets.iter() {
            state.budgets[*budget].tokens += tokens;
        }
        let entry = &mut state.trajectory[self.active.index];
        match &self.role {
            Role::Tool(name) => *entry.tool_tokens.get_or_insert_default().entry(name.to_string()).or_default() += tokens,
            Role::Fit => *entry.fit_tokens.get_or_insert_default() += tokens,
            Role::Arbitrate => {
                let name = crate::arbitrate::ARBITRATE.to_owned();
                *entry.tool_tokens.get_or_insert_default().entry(name).or_default() += tokens
            }
        }
    }

    /// Hand the run's record a time a language model was asked for this tool or fit.
    #[cfg(feature = "lm")]
    pub(crate) fn wrote(&self, wrote: crate::lm::Wrote) -> Result<()> {
        let (index, tool) = (self.active.index, self.tool().map(str::to_owned));
        self.active.run.emit(|run| Event::Wrote {
            run,
            index,
            tool,
            key: crate::lm::key(&wrote.request),
            model: wrote.model,
            request: serde_json::to_value(&wrote.request).ok(),
            response: wrote.response.and_then(|response| crate::lm::as_kept(&response)),
            error: wrote.error,
        })
    }

    /// Hand the run's record an event of this request, made with the run's number and the
    /// request's place, if the run has a record. An error ends the run.
    #[cfg(feature = "repl")]
    pub(crate) fn emit(&self, event: impl FnOnce(u64, usize) -> Event) -> Result<()> {
        let index = self.active.index;
        self.active.run.emit(|run| event(run, index))
    }

    /// Record a change to a map in the run's trajectory, under `changes` on this request.
    ///
    /// A run that is recorded is handed it as a `changed` event. Should the record refuse it,
    /// the run ends with that error at its next event.
    pub fn changed(&self, what: impl Into<Json>) {
        let what = what.into();
        {
            let mut state = self.active.run.lock();
            state.trajectory[self.active.index]
                .changes
                .get_or_insert_default()
                .push(what.clone());
        }
        let index = self.active.index;
        // A refusal is kept by the run, and is what its next event ends it with.
        let _ = self.active.run.emit(|run| Event::Changed { run, index, change: what });
    }
}

impl fmt::Debug for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut call = f.debug_struct("Call");
        call.field("depth", &self.active.depth)
            .field("iteration", &self.iteration)
            .field("index", &self.active.index);
        call.field("role", &self.role).field("inputs", &self.shown).finish()
    }
}
