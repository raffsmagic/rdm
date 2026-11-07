//! The module and its loop.
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
//! With an arbiter ([`Rdm::arbiter`]), the tools the answers select are not waited for all
//! together: each time one comes back, what has come back is settled, and the call is over
//! once it is answered ([`crate::arbitrate`]).
//!
//! A run is a tree of such calls. Its shape is not written anywhere: it comes from what the
//! answers select and what the tools do. It stops where no answer selects a tool, where a
//! request comes round that was already asked on the way to it, or at a limit.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use indexmap::IndexMap;
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use systemone::{Decision, DecisionModel, Json, Params};

use crate::arbitrate::{ARBITRATE, Arbiter, Arbitration, Branch, Branches, Ledger, Ruling, State, Stop};
use crate::call::{Ask, Call, Role};
use crate::error::{Error, Result, listed, sorted};
use crate::inputs::Inputs;
use crate::observation::as_observation;
use crate::parallel::{Arrivals, Came, in_parallel};
use crate::prediction::{Entry, Prediction, Reason};
use crate::record::{self, Event, Module, Record};
use crate::signature::{FINISH, FINISH_DESCRIPTION, OBSERVATION, Opt, RESERVED, Signature, Space};
use crate::tool::{Source, Tool, Tools};

/// What makes two requests the same request.
type Key = [u8; 32];
/// The answers of one request, by output.
type Answers = IndexMap<String, Option<Decision>>;

/// Each module is given a number when it is built: a request it asks is a repeat only of its own.
static MODULES: AtomicU64 = AtomicU64::new(1);

/// Each run is given a number when it begins, so that the events of runs at once are told apart.
static RUNS: AtomicU64 = AtomicU64::new(1);

/// The depth no run goes below, whatever its modules allow: a call this deep is a leaf, as
/// one at `max_depth` is. It is answered, the tools it selected are not run, and the reason
/// is `max_depth`.
///
/// A module's own `max_depth` is counted from where it was called, so modules that call each
/// other never reach it. This is counted from the call that started the run, and they do.
pub const DEEPEST: usize = 1024;

/// A seam: what is given each request before it is asked, and returns the signature and
/// inputs to ask in their place.
///
/// A fit changes what the model is shown, not what the code acts on: tools still read the
/// call's inputs whole. It can make sub-calls through the call, as a tool can, and they are
/// sub-calls of the request it is fitting, which has already been counted, so they cannot
/// use up its place. It is given every request, its own included: a fit marks the questions
/// it asks ([`crate::Output::mark`]) and lets those through. It must keep every output it was
/// given, though it may cut an output's options. A choice it cuts down to one option is
/// answered with that option and not asked, as a choice declared with one option is.
///
/// ```
/// use rdm::{Rdm, Signature, inputs};
/// use systemone::{Answer, Scripted};
///
/// // Shows the first line of a long ticket, and leaves the questions as they are.
/// let first_line = |signature: Signature, inputs: rdm::Inputs, _: &rdm::Call| {
///     let ticket = inputs.value("ticket").and_then(|ticket| Some(ticket.as_str()?.lines().next()?.to_owned()));
///     Ok((signature, inputs.with("ticket", ticket)))
/// };
/// let model = std::sync::Arc::new(Scripted::new(|_| [("refund", Answer::noul(0.9))]));
/// let triage = Signature::new("Triage a ticket.").input("ticket").noul("refund", "Is the customer asking for a refund?");
/// let rdm = Rdm::new(triage).fit_fn(first_line).model(model.clone());
/// let result = rdm.call(inputs! { "ticket" => "I was charged twice.\nOrder 98423." }).unwrap();
/// assert_eq!(model.requests()[0].state["inputs"]["ticket"], "I was charged twice.");
/// assert_eq!(result.trajectory[0].inputs["ticket"], "I was charged twice.");
/// ```
pub trait Fit: Send + Sync {
    /// The signature and inputs to ask in place of these. The inputs are plain values: what
    /// the request shows.
    fn fit(&self, signature: Signature, inputs: Inputs, call: &Call) -> Result<(Signature, Inputs)>;

    /// A fit of its own that is set as this one is, for [`Rdm::copy`]. None, as it is unless
    /// a fit says otherwise, has the copy of the module use this same fit: right for one that
    /// keeps nothing between requests. A fit that does keep something returns a copy here, so
    /// that a copy of the module does not write into the original's.
    fn copy(&self) -> Option<Arc<dyn Fit>> {
        None
    }

    /// The fit as itself, for whatever knows what kind of thing it is. None, as it is unless
    /// a fit says otherwise. A fit that copies itself returns itself here: a tool that holds a
    /// handle to it then finds, among [`Rdm::fits`], the one the module that is running was
    /// given, which in a copy of the module is not the one it holds.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }
}

/// A fit that is a function.
struct FitFn<F>(F);

impl<F> Fit for FitFn<F>
where
    F: Fn(Signature, Inputs, &Call) -> Result<(Signature, Inputs)> + Send + Sync,
{
    fn fit(&self, signature: Signature, inputs: Inputs, call: &Call) -> Result<(Signature, Inputs)> {
        (self.0)(signature, inputs, call)
    }
}

/// What a module is set to. A request reads it as it is when the request is made.
#[derive(Clone)]
struct Config {
    sources: Vec<Arc<dyn Source>>,
    fits: Vec<Arc<dyn Fit>>,
    always: Vec<String>,
    finish: bool,
    max_iters: usize,
    max_depth: usize,
    max_predictor_calls: usize,
    max_tokens: Option<u64>,
    threads: usize,
    model: Option<Arc<dyn DecisionModel>>,
    params: IndexMap<String, Params>,
    /// What the runs it starts are written down to.
    record: Option<record::Shared>,
    /// A name two of the tools the module was built with had. Every call says so.
    duplicate: Option<String>,
    /// What settles the tools a request selected as they come back. None waits for all of them.
    arbiter: Option<Arc<dyn Arbiter>>,
}

struct Inner {
    id: u64,
    signature: Signature,
    tools: Tools,
    config: RwLock<Arc<Config>>,
}

/// A Recursive Decision Model.
///
/// It answers a signature's questions once, and lets the answers select tools. An output
/// named after a tool calls it on a yes or on a chosen option. An output typed as a tool
/// points at one of the module's tools: each request offers the tools as they are then, by
/// name and description, with `finish`, and after the tool pointed at has run the same
/// question is put again, showing the inputs as they are now and, in an input named
/// `observation`, what the tool returned.
///
/// A tool can be a function, another module, or anything that writes. Inside a tool, calling
/// the module through the [`Call`] is a sub-call: one level deeper, in the same run, with its
/// prediction returned to the tool. Nothing here generates text.
///
/// It is a handle, cheap to clone: a clone is the same module, and can be moved into a tool.
/// A tool reaches its own module through the call it is given, so it seldom needs one.
///
/// ```
/// use rdm::{Rdm, Shared, Signature, Tool, inputs};
/// use serde_json::json;
/// use systemone::{Answer, Scripted};
///
/// let house = Shared::new(json!({"tv": "off", "lights": "on"}));
/// let switch = |device: &'static str, to: &'static str| {
///     let house = house.clone();
///     Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
///         house.update(|house| house[device] = json!(to));
///         Ok(format!("{device}: now {to}"))
///     })
/// };
///
/// let operate = Signature::new("Operate a smart home, one step at a time.")
///     .input("task")
///     .input_typed("house", "dict")
///     .input_described("observation", "What the last step did.")
///     .tool("next", "What should happen next for `inputs.task`?");
///
/// // A decision model that points at what is not yet so. A real one is `one_system::Client`.
/// let model = Scripted::new(|request| {
///     let house = &request.state["inputs"]["house"];
///     let next = if house["tv"] == "off" { "tv on" } else if house["lights"] == "on" { "lights off" } else { "finish" };
///     [("next", Answer::chose(next, ["tv on", "lights off", "finish"]))]
/// });
///
/// let rdm = Rdm::new(operate).tool(switch("tv", "on")).tool(switch("lights", "off")).model(model);
/// let result = rdm.call(inputs! { "task" => "Movie time.", "house" => &house }).unwrap();
/// assert_eq!(result.observations["next"], json!([["tv on", "tv: now on"], ["lights off", "lights: now off"], ["finish", null]]));
/// assert_eq!(house.get(), json!({"tv": "on", "lights": "off"}));
/// ```
#[derive(Clone)]
pub struct Rdm {
    inner: Arc<Inner>,
}

impl Rdm {
    /// A module that asks a signature's questions. With no tools it is one request and its answers.
    ///
    /// The defaults: `max_iters` 20, `max_depth` 3, `max_predictor_calls` 256, no limit on
    /// tokens, and `finish` on offer. What is wrong with how a module is built (a field with
    /// a reserved name, two tools of one name, a limit below one) is returned by
    /// [`Rdm::check`] and by every call.
    pub fn new(signature: Signature) -> Self {
        let config = Config {
            sources: Vec::new(),
            fits: Vec::new(),
            always: Vec::new(),
            finish: true,
            max_iters: 20,
            max_depth: 3,
            max_predictor_calls: 256,
            max_tokens: None,
            threads: 8,
            model: None,
            params: IndexMap::new(),
            record: None,
            duplicate: None,
            arbiter: None,
        };
        let inner = Inner {
            id: MODULES.fetch_add(1, Ordering::Relaxed),
            signature,
            tools: Tools::default(),
            config: RwLock::new(Arc::new(config)),
        };
        Rdm { inner: Arc::new(inner) }
    }

    fn config(&self) -> Arc<Config> {
        self.inner.config.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn edit(&self, change: impl FnOnce(&mut Config)) {
        let mut config = self.inner.config.write().unwrap_or_else(PoisonError::into_inner);
        change(Arc::make_mut(&mut config));
    }

    fn set(self, change: impl FnOnce(&mut Config)) -> Self {
        self.edit(change);
        self
    }

    /// A tool the answers can select: by an output of its name, or by the output that points.
    pub fn tool(self, tool: Tool) -> Self {
        let mut tools = self.inner.tools.lock();
        if tools.contains_key(tool.name()) {
            let name = tool.name().to_owned();
            drop(tools);
            return self.set(|config| config.duplicate = config.duplicate.take().or(Some(name)));
        }
        tools.insert(tool.name().to_owned(), tool);
        drop(tools);
        self
    }

    /// Several tools, in order.
    pub fn tools_from(self, tools: impl IntoIterator<Item = Tool>) -> Self {
        tools.into_iter().fold(self, Rdm::tool)
    }

    /// Something that offers tools of its own: each request's map holds what it offers then.
    pub fn source(self, source: impl Source + 'static) -> Self {
        self.source_shared(Arc::new(source))
    }

    /// As [`Rdm::source`], for a source something else holds too.
    pub fn source_shared(self, source: Arc<dyn Source>) -> Self {
        self.set(|config| config.sources.push(source))
    }

    /// What is given each request before it is asked. Several are applied in the order given.
    pub fn fit(self, fit: impl Fit + 'static) -> Self {
        self.fit_shared(Arc::new(fit))
    }

    /// As [`Rdm::fit`], for a fit something else holds too.
    pub fn fit_shared(self, fit: Arc<dyn Fit>) -> Self {
        self.set(|config| config.fits.push(fit))
    }

    /// As [`Rdm::fit`], for a function.
    pub fn fit_fn<F>(self, fit: F) -> Self
    where
        F: Fn(Signature, Inputs, &Call) -> Result<(Signature, Inputs)> + Send + Sync + 'static,
    {
        self.fit(FitFn(fit))
    }

    /// Names of tools that stay on offer whatever a fit cuts from the map, as `finish` does.
    pub fn always<N: Into<String>>(self, names: impl IntoIterator<Item = N>) -> Self {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        self.set(|config| config.always = names)
    }

    /// Whether an output that points is offered `finish` beside the tools. Without it a loop
    /// ends at a limit, at a repeat, or at a tool named `finish` if there is one.
    pub fn finish(self, offered: bool) -> Self {
        self.set(|config| config.finish = offered)
    }

    /// Turns one pointer's loop may take.
    pub fn max_iters(self, turns: usize) -> Self {
        self.set(|config| config.max_iters = turns)
    }

    /// The depth at which a call's tools are not run. The first call is at depth 0, so 1 lets
    /// it use its tools and makes every sub-call a leaf, and 0 makes it a leaf itself. It is
    /// counted from where the module was called. A call can ask for another depth for itself
    /// ([`Ask::max_depth`]), and no run goes below [`DEEPEST`], whatever its modules allow.
    pub fn max_depth(self, depth: usize) -> Self {
        self.set(|config| config.max_depth = depth)
    }

    /// Requests answered in one run, sub-calls included.
    pub fn max_predictor_calls(self, requests: usize) -> Self {
        self.set(|config| config.max_predictor_calls = requests)
    }

    /// Tokens one run may use in total, as the models report them: those of its requests and
    /// those its tools report. None for no limit.
    pub fn max_tokens(self, tokens: impl Into<Option<u64>>) -> Self {
        let tokens = tokens.into();
        self.set(|config| config.max_tokens = tokens)
    }

    /// What settles the tools each request selects as they come back ([`crate::arbitrate`]).
    /// A module with none waits for every tool a request selected, and uses what each returned.
    pub fn arbiter(self, arbiter: impl Arbiter + 'static) -> Self {
        self.arbiter_shared(Arc::new(arbiter))
    }

    /// As [`Rdm::arbiter`], with one that is shared.
    pub fn arbiter_shared(self, arbiter: Arc<dyn Arbiter>) -> Self {
        self.set(|config| config.arbiter = Some(arbiter))
    }

    /// How many tools, or sub-calls of one batch, run at once. 8 unless set.
    pub fn threads(self, width: usize) -> Self {
        self.set(|config| config.threads = width.max(1))
    }

    /// What the runs this module starts are written down to, event by event: see [`crate::record`].
    ///
    /// A run has one record: that of the module that started it. Another module that joins the
    /// run, as a tool or from inside one, writes to the run's, whatever it was given itself.
    pub fn record(self, record: impl Record + 'static) -> Self {
        self.set_record(record);
        self
    }

    /// Write down the runs this module starts from now on: see [`crate::record`].
    pub fn set_record(&self, record: impl Record + 'static) {
        let record: record::Shared = Arc::new(record);
        self.edit(|config| config.record = Some(record));
    }

    /// Write down no run from now on.
    pub fn unset_record(&self) {
        self.edit(|config| config.record = None);
    }

    /// The module as a run's `began` event describes it.
    fn described(&self, config: &Config) -> Module {
        Module {
            instructions: self.inner.signature.instructions().to_owned(),
            tools: self
                .inner
                .tools
                .all()
                .iter()
                .map(|tool| (tool.name().to_owned(), tool.description().to_owned()))
                .collect(),
            sources: config.sources.iter().map(|source| source.name()).collect(),
            fits: config.fits.len(),
            finish: config.finish,
            always: config.always.clone(),
            max_iters: config.max_iters,
            max_depth: config.max_depth,
            max_predictor_calls: config.max_predictor_calls,
            max_tokens: config.max_tokens,
        }
    }

    /// The decision model that answers the module's requests.
    pub fn model(self, model: impl DecisionModel + 'static) -> Self {
        self.set_model(model);
        self
    }

    /// A threshold, cuts or weights for an output, by its name. See [`Rdm::set_params`].
    pub fn params(self, output: impl Into<String>, params: Params) -> Self {
        self.set_params(output, params);
        self
    }

    /// Whether the module is built in a way it can be used: no field with a name a call keeps
    /// for itself, no question that cannot be asked ([`Signature::check`]), no two tools of
    /// one name, and no limit below one. Every call checks.
    pub fn check(&self) -> Result<()> {
        let config = self.config();
        let reserved = self.inner.signature.reserved();
        if !reserved.is_empty() {
            return Err(Error::Usage(format!(
                "Field name(s) {} are reserved by RDM. Rename these fields.",
                sorted(reserved)
            )));
        }
        self.inner.signature.check()?;
        if let Some(name) = &config.duplicate {
            return Err(Error::Usage(format!("Duplicate tool name '{name}'")));
        }
        if config.max_iters < 1 || config.max_predictor_calls < 1 {
            return Err(Error::Usage(
                "max_iters and max_predictor_calls must be at least 1: the first request counts.".to_owned(),
            ));
        }
        Ok(())
    }

    /// Give the module its decision model, or another one. Requests asked from now on go to it.
    pub fn set_model(&self, model: impl DecisionModel + 'static) {
        let model: Arc<dyn DecisionModel> = Arc::new(model);
        self.edit(|config| config.model = Some(model));
    }

    /// The decision model, if the module has one.
    pub fn decision_model(&self) -> Option<Arc<dyn DecisionModel>> {
        self.config().model.clone()
    }

    /// Set the numbers an output is decided with: a noul's threshold, a score's cuts, a
    /// choice's weights. They are never sent to the model.
    ///
    /// They belong to the output's name, whichever request asks it: the module's own, or one
    /// whose questions a tool built. Parameters set for an output the module's signature
    /// declares apply only where the answers it can have are the ones declared, so weights
    /// fitted to one set of options are not applied to another.
    ///
    /// ```
    /// use rdm::{Rdm, Signature, inputs};
    /// use systemone::{Answer, Params, Scripted};
    ///
    /// let search = Rdm::new(Signature::new("Search a text.").input("text").noul("look", "Is the answer stated in `inputs.text`?"))
    ///     .model(Scripted::new(|_| [("look", Answer::noul(0.6))]));
    /// assert!(search.call(inputs! { "text" => "..." }).unwrap().yes("look"));
    /// search.set_params("look", Params::threshold(0.7));
    /// assert!(!search.call(inputs! { "text" => "..." }).unwrap().yes("look"));
    /// ```
    pub fn set_params(&self, output: impl Into<String>, params: Params) {
        let output = output.into();
        self.edit(|config| {
            config.params.insert(output, params);
        });
    }

    /// Forget the numbers set for an output: it is decided with the defaults again.
    pub fn unset_params(&self, output: &str) -> Option<Params> {
        let mut removed = None;
        self.edit(|config| removed = config.params.shift_remove(output));
        removed
    }

    /// The numbers set for an output, if any were.
    pub fn params_of(&self, output: &str) -> Option<Params> {
        self.config().params.get(output).cloned()
    }

    /// Every output that has numbers set, with them.
    pub fn all_params(&self) -> IndexMap<String, Params> {
        self.config().params.clone()
    }

    /// The questions the module asks when a call names no others.
    pub fn signature(&self) -> &Signature {
        &self.inner.signature
    }

    /// The module's own tools. A tool put here is on offer from the next run; a run in
    /// progress works on its own copy ([`Call::tools`]).
    pub fn tools(&self) -> &Tools {
        &self.inner.tools
    }

    /// The number the module was given when it was built. Clones share it; a [`Rdm::copy`] has its own.
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// What the module was given that offers tools of its own, in the order given. In a copy
    /// of a module these are the copy's own, where a source says how it is copied.
    pub fn sources(&self) -> Vec<Arc<dyn Source>> {
        self.config().sources.clone()
    }

    /// What each request is given before it is asked, in the order given. In a copy of a
    /// module these are the copy's own, where a fit says how it is copied.
    pub fn fits(&self) -> Vec<Arc<dyn Fit>> {
        self.config().fits.clone()
    }

    /// A module of its own that is set as this one is: the same questions, a copy of the
    /// tools and the parameters. Changing one from then on does not change the other.
    ///
    /// A source or a fit that keeps something of its own is copied too, where it says how
    /// ([`Source::copy`], [`Fit::copy`]): a run of the copy then changes what the copy's
    /// source offers, and not the original's. One that does not say is the same one in both.
    ///
    /// The tools are the same tools, and a tool is a closure: whatever one holds, the copy's
    /// holds too. A tool that holds a handle to a source or a fit of the module therefore
    /// holds the original's, and finds the copy's through the call it is given
    /// ([`Call::rdm`], then [`Rdm::sources`] or [`Rdm::fits`]). The helpers do so for what
    /// they are given a call for: a `Repl`'s writer, `define_in`, `remove_in` and `show_in`,
    /// and a `Near`'s `add_in` and `remove_in`, act on the copy's own in a run of the copy.
    /// A module that is a tool ([`Tool::module`]) is the same module in both.
    pub fn copy(&self) -> Rdm {
        let mut config = (*self.config()).clone();
        for source in &mut config.sources {
            *source = source.copy().unwrap_or_else(|| source.clone());
        }
        for fit in &mut config.fits {
            *fit = fit.copy().unwrap_or_else(|| fit.clone());
        }
        let inner = Inner {
            id: MODULES.fetch_add(1, Ordering::Relaxed),
            signature: self.inner.signature.clone(),
            tools: self.inner.tools.copy(),
            config: RwLock::new(Arc::new(config)),
        };
        Rdm { inner: Arc::new(inner) }
    }

    /// Answer the signature's questions about the inputs, and run the tools the answers select.
    ///
    /// This starts a run. Inside a tool, call the module through the tool's [`Call`]
    /// instead: that is a sub-call, which joins the run in progress.
    pub fn call(&self, inputs: Inputs) -> Result<Prediction> {
        self.call_with(Ask::new(inputs))
    }

    /// As [`Rdm::call`], for a call that says more than its inputs: other questions in place
    /// of the module's own, the answers only, or state for the tools.
    pub fn call_with(&self, ask: impl Into<Ask>) -> Result<Prediction> {
        let mut ask = ask.into();
        let config = self.config();
        self.check()?;
        let run = Arc::new(Run::new(config.record.clone()));
        let tag = ask.tag.take().unwrap_or(Json::Null);
        run.emit(|run| Event::Began {
            run,
            tag,
            module: self.described(&config),
        })?;
        // A run that does not come to its end, by an error or a panic, lets nothing finish
        // unwaited: what it would have been held against is not there.
        let mut unended = Unended {
            run: run.clone(),
            ended: false,
        };
        let ran = self.begin(&run, &config, ask);
        // A run that failed has ended too, and says with what. A record that refused has
        // nothing more to be handed.
        let ended = match &ran {
            Ok(prediction) => run.emit(|run| Event::Ended {
                run,
                reason: Some(prediction.termination_reason),
                error: None,
            }),
            Err(error) => {
                let _ = run.emit(|run| Event::Ended {
                    run,
                    reason: None,
                    error: Some(error.to_string()),
                });
                Ok(())
            }
        };
        let mut prediction = ran?;
        ended?;
        unended.ended = true;
        drop(unended);
        prediction.receipts = run.receipts.lock().unwrap_or_else(PoisonError::into_inner).clone();
        prediction.trajectory = match Arc::try_unwrap(run) {
            Ok(run) => run.state.into_inner().unwrap_or_else(PoisonError::into_inner).trajectory,
            Err(run) => run.lock().trajectory.clone(),
        };
        Ok(prediction)
    }

    /// The first request of a run, and everything it leads to.
    fn begin(&self, run: &Arc<Run>, config: &Config, ask: Ask) -> Result<Prediction> {
        run.lock().budgets.push(Budget::new(config));
        let budgets: Arc<[usize]> = Arc::from([0]);
        let stop = ask.stop.clone();
        let state = Arc::new(ask.state.unwrap_or_else(|| json!({})));
        let request = self.request(
            run,
            config,
            ask.signature.as_ref(),
            ask.inputs,
            0,
            Depth::Under(config.max_depth, ask.max_depth),
        )?;
        let opened = self.open(&mut run.lock(), &request, 0, None, None, 0, &None, &budgets);
        self.turns(run, opened, request, 0, None, budgets, state, stop)
    }

    /// Several runs at once, each on its own copy of the tools. Their results come back in the order given.
    pub fn batch<A: Into<Ask>>(&self, asks: impl IntoIterator<Item = A>) -> Vec<Result<Prediction>> {
        let runs: Vec<_> = asks
            .into_iter()
            .map(|ask| {
                let ask = ask.into();
                move || self.call_with(ask)
            })
            .collect();
        in_parallel(self.config().threads, runs)
    }

    /// Ask requests one level below the request whose tool is running, at once. Returns their predictions in order.
    pub(crate) fn children(&self, active: &Active, asks: Vec<Ask>) -> Result<Vec<Prediction>> {
        let config = self.config();
        self.check()?;
        active.go_on()?;
        let (run, depth) = (&active.run, active.depth + 1);
        // Another module called inside a tool joins the run, and behaves there as it does alone: its
        // own limits hold for what it does, its depth is counted from where it was called, and the
        // run's requests and tokens are counted for it too.
        let own = active.module.id() == self.id();
        let budgets = if own {
            active.budgets.clone()
        } else {
            let mut state = run.lock();
            state.budgets.push(Budget::new(&config));
            let mine = state.budgets.len() - 1;
            active.budgets.iter().copied().chain([mine]).collect()
        };
        let limit = if own { active.limit } else { usize::MAX };
        let mut requests = Vec::with_capacity(asks.len());
        for ask in asks {
            // State a sub-call is given is its own and that of the calls under it.
            let state = ask.state.map_or_else(|| active.state.clone(), Arc::new);
            let request = self.request(
                run,
                &config,
                ask.signature.as_ref(),
                ask.inputs,
                depth,
                Depth::Under(limit, ask.max_depth),
            )?;
            requests.push((request, state));
        }
        // They are admitted in the order given, so which ones a limit stops does not depend on timing.
        let opened: Vec<Opened> = {
            let mut state = run.lock();
            let open = |(request, _): &(Request, _)| {
                self.open(
                    &mut state,
                    request,
                    depth,
                    Some(active.index),
                    active.tool.clone(),
                    0,
                    &active.path,
                    &budgets,
                )
            };
            requests.iter().map(open).collect()
        };
        let calls = opened.into_iter().zip(requests).map(|(opened, (request, state))| {
            let (path, budgets, stop) = (active.path.clone(), budgets.clone(), active.stop.clone());
            move || self.turns(run, opened, request, depth, path, budgets, state, stop)
        });
        let predictions = in_parallel(config.threads, calls.collect())
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        // What stopped something in a call below, other than a decision, stopped it for the call above too.
        let cut = predictions
            .iter()
            .map(|prediction| prediction.termination_reason)
            .filter(|reason| *reason != Reason::Final);
        active.cuts.lock().unwrap_or_else(PoisonError::into_inner).extend(cut);
        Ok(predictions)
    }

    /// This module's tools for a run: a copy of its own, made when the run first needs it.
    pub(crate) fn run_tools(&self, run: &Run) -> Tools {
        self.mine(run).tools
    }

    /// What a run holds for this module: its tools, and the turn to read its sources.
    fn mine(&self, run: &Run) -> Mine {
        let mut state = run.lock();
        let mine = state.tools.entry(self.id()).or_insert_with(|| Mine {
            tools: self.inner.tools.copy(),
            reading: Arc::default(),
        });
        mine.clone()
    }

    /// The run's map as it is now: what each source offers is on it, a function written a moment ago, and not one removed.
    fn map(&self, run: &Run, config: &Config) -> Result<Tools> {
        let Mine { tools, reading } = self.mine(run);
        if config.sources.is_empty() {
            return Ok(tools);
        }
        // A source is read before the run is locked: it may take a while to say what it offers.
        // The run's requests read it one at a time, each putting on the map what it read before
        // the next reads: a reading that was slow to come back cannot take off the map a name
        // that a later reading put on it.
        let _turn = reading.lock().unwrap_or_else(PoisonError::into_inner);
        let offers: Vec<_> = config
            .sources
            .iter()
            .map(|source| (Arc::as_ptr(source).cast::<()>() as usize, source.name(), source.offers()))
            .collect();
        let mut state = run.lock();
        for (source, name, offered) in offers {
            let mine = state.offered.entry((self.id(), source)).or_default();
            let mut map = tools.lock();
            let now: HashSet<String> = offered.iter().map(|tool| tool.name().to_owned()).collect();
            for gone in mine.difference(&now) {
                map.shift_remove(gone);
            }
            let taken: Vec<&str> = offered
                .iter()
                .map(Tool::name)
                .filter(|name| map.contains_key(*name) && !mine.contains(*name))
                .collect();
            if !taken.is_empty() {
                return Err(Error::Usage(format!(
                    "{} are offered by `{name}` and are already names on the map.",
                    listed(taken)
                )));
            }
            for tool in offered {
                map.insert(tool.name().to_owned(), tool);
            }
            *mine = now;
        }
        Ok(tools)
    }

    /// The signature with its output that points offering the run's tools as they are now, and `finish`.
    fn offered(&self, run: &Run, config: &Config, mut signature: Signature) -> Result<Signature> {
        let pointers: Vec<String> = signature.pointers().into_iter().map(str::to_owned).collect();
        if pointers.len() > 1 {
            return Err(Error::Usage(format!(
                "Outputs {} point at a tool; a signature may have one output that points at a tool.",
                listed(&pointers)
            )));
        }
        for name in pointers {
            let tools = self.map(run, config)?;
            let map = tools.lock();
            let mut options: Vec<Opt> = map
                .values()
                .filter(|tool| tool.name() != FINISH)
                .map(|tool| Opt::new(tool.name(), tool.description()))
                .collect();
            match map.get(FINISH) {
                Some(finish) => options.push(Opt::new(FINISH, finish.description())),
                None if config.finish => options.push(Opt::new(FINISH, FINISH_DESCRIPTION)),
                None => {}
            }
            if options.is_empty() {
                return Err(Error::Usage(format!(
                    "Output '{name}' points at a tool, and there is none to point at: the map is empty and `finish` is not offered."
                )));
            }
            if let Some(output) = signature.output_mut(&name) {
                output.space = Space::Choice(options);
                output.points = true;
                output.always = std::iter::once(FINISH.to_owned()).chain(config.always.iter().cloned()).collect();
            }
        }
        Ok(signature)
    }

    /// A request as its signature, its inputs as they are now, and the depth at which its tools stop.
    fn request(
        &self,
        run: &Run,
        config: &Config,
        signature: Option<&Signature>,
        mut inputs: Inputs,
        depth: usize,
        limit: Depth,
    ) -> Result<Request> {
        let signature = self.offered(run, config, signature.unwrap_or(&self.inner.signature).clone())?;
        // A question that cannot be asked is refused here, before the request is recorded or counted.
        signature.check()?;
        if signature.pointer().is_some() && signature.inputs().contains_key(OBSERVATION) && !inputs.contains(OBSERVATION) {
            // Before the first tool has run there is nothing to observe.
            inputs.set(OBSERVATION, "");
        }
        let (given, taken): (HashSet<&str>, HashSet<&str>) = (
            inputs.names().into_iter().collect(),
            signature.inputs().keys().map(String::as_str).collect(),
        );
        if given != taken || !signature.reserved().is_empty() {
            return Err(Error::Usage(format!(
                "The call has inputs {}; its signature takes {}, and no field may be named any of {}.",
                sorted(inputs.names()),
                sorted(signature.inputs().keys()),
                sorted(RESERVED)
            )));
        }
        // What the request shows is read now: a live input is as its tools last left it.
        let shown = inputs.snapshot()?;
        // An input that is a picture holds one or nothing: anything else is refused before it is recorded or sent.
        signature.check_images(&shown)?;
        let (limit, answers_only) = match limit {
            // A call keeps to the limit of the run it is in, and to its own where that is lower. What
            // the call asked for itself is in the place of the module's own, and no run goes below
            // the deepest a run may.
            Depth::Under(limit, further) => (
                limit.min(depth.saturating_add(further.unwrap_or(config.max_depth))).min(DEEPEST),
                further == Some(0),
            ),
            // A later turn of a loop is the same call: it keeps the limit its first turn was given.
            Depth::Kept(limit, answers_only) => (limit, answers_only),
        };
        let key = key(self.id(), &signature, &shown);
        Ok(Request {
            signature,
            inputs,
            shown: Arc::new(shown),
            limit,
            answers_only,
            key,
        })
    }

    /// Record a request and count it against the limits. The caller holds the run's lock.
    #[allow(
        clippy::too_many_arguments,
        reason = "one private step of a run, called from this file with the run's parts"
    )]
    fn open(
        &self,
        state: &mut RunState,
        request: &Request,
        depth: usize,
        parent: Option<usize>,
        tool_name: Option<Arc<str>>,
        iteration: usize,
        path: &Path,
        budgets: &[usize],
    ) -> Opened {
        // What the request showed, as it showed it: a tool may change what an input shows later.
        let mut entry = Entry {
            depth,
            iteration,
            parent,
            tool_name: tool_name.map(|name| name.to_string()),
            inputs: Json::Object((*request.shown).clone()),
            ..Entry::default()
        };
        // A choice with one option leaves nothing to decide: it is answered with that option and
        // not asked. A request made only of such questions is not sent, and counts against no limit.
        let outputs = request.signature.outputs();
        let only: IndexMap<String, Decision> = outputs
            .iter()
            .filter_map(|(name, output)| Some((name.clone(), output.only()?)))
            .collect();
        let status = if let Some(answers) = asked_before(path, &request.key) {
            // The same questions about the same state, asked on the way here, would get the same
            // answers and select the same tools again. It is not asked: the run has come round.
            entry.repeat = true;
            Status::Repeat(answers)
        } else if only.len() < outputs.len() {
            // A request past a limit is recorded and not answered, so nothing is dropped silently.
            match budgets.iter().find_map(|budget| state.budgets[*budget].spent()) {
                Some(spent) => {
                    entry.skipped = Some(spent);
                    Status::Skipped(spent)
                }
                None => {
                    budgets.iter().for_each(|budget| state.budgets[*budget].predictor_calls += 1);
                    Status::Admitted
                }
            }
        } else {
            Status::Admitted
        };
        state.trajectory.push(entry);
        Opened {
            index: state.trajectory.len() - 1,
            only,
            status,
        }
    }

    /// Answer a request that has been recorded and, while its output that points points at a tool, put it again.
    ///
    /// Each turn after the first shows the inputs as they are now, so a tool that changed
    /// what a live input shows changed what the next turn shows, and an input named
    /// `observation` shows what the last tool returned. The turns are a loop at one level, so
    /// a loop may be as long as `max_iters` allows. Returns the first turn's answers, with
    /// every tool pointed at in turn, each with what it returned.
    #[allow(
        clippy::too_many_arguments,
        reason = "one private step of a run, called from this file with the run's parts"
    )]
    fn turns(
        &self,
        run: &Arc<Run>,
        mut opened: Opened,
        mut request: Request,
        depth: usize,
        mut path: Path,
        budgets: Arc<[usize]>,
        state: Arc<Json>,
        stop: Option<Arc<Stop>>,
    ) -> Result<Prediction> {
        let config = self.config();
        let pointer = request.signature.pointer().map(str::to_owned);
        // Each turn's options are made from the first turn's signature.
        let base = request.signature.clone();
        let (mut first, mut done, mut reasons, mut iteration) = (None, Vec::new(), Vec::new(), 0);
        loop {
            if stop.as_ref().is_some_and(|stop| stop.is_set()) {
                return Err(stopped());
            }
            let (turn, step) = self.answer(run, &config, &opened, &request, depth, &path, &budgets, &state, iteration, &stop)?;
            reasons.push(turn.reason);
            first.get_or_insert(turn);
            let Some(step) = step else { break };
            done.push(json!([step.name, step.result]));
            if step.name == FINISH {
                break;
            }
            path = step.path;
            let mut inputs = request.inputs.clone();
            if base.inputs().contains_key(OBSERVATION) {
                inputs.set(OBSERVATION, as_observation(&step.result));
            }
            request = self.request(
                run,
                &config,
                Some(&base),
                inputs,
                depth,
                Depth::Kept(request.limit, request.answers_only),
            )?;
            let parent = opened.index;
            iteration += 1;
            if iteration >= config.max_iters {
                let stopped = Entry {
                    depth,
                    iteration,
                    parent: Some(parent),
                    tool_name: Some(step.name),
                    inputs: Json::Object((*request.shown).clone()),
                    skipped: Some(Reason::MaxIters),
                    ..Entry::default()
                };
                let index = {
                    let mut state = run.lock();
                    state.trajectory.push(stopped);
                    state.trajectory.len() - 1
                };
                run.opened(index, false)?;
                reasons.push(Reason::MaxIters);
                break;
            }
            opened = self.open(
                &mut run.lock(),
                &request,
                depth,
                Some(parent),
                Some(step.name.into()),
                iteration,
                &path,
                &budgets,
            );
        }
        let first = first.unwrap_or_else(|| Turn::unanswered(&request.signature, Reason::Final));
        let mut prediction = Prediction::of(&base, first.answers);
        prediction.observations = first.observations;
        prediction.arbitration = first.arbitration;
        if let Some(pointer) = pointer {
            prediction.observations.insert(pointer, Json::Array(done));
        }
        prediction.termination_reason = Reason::of(reasons);
        Ok(prediction)
    }

    /// Ask a request that has been recorded, and run the tools its answers select.
    ///
    /// Returns the answers and, when the output that points pointed at a tool or at `finish`,
    /// that tool's name, what it returned, and the requests asked on the way.
    #[allow(
        clippy::too_many_arguments,
        reason = "one private step of a run, called from this file with the run's parts"
    )]
    fn answer(
        &self,
        run: &Arc<Run>,
        config: &Config,
        opened: &Opened,
        request: &Request,
        depth: usize,
        path: &Path,
        budgets: &Arc<[usize]>,
        state: &Arc<Json>,
        iteration: usize,
        stop: &Option<Arc<Stop>>,
    ) -> Result<(Turn, Option<Step>)> {
        let (index, signature) = (opened.index, &request.signature);
        match &opened.status {
            Status::Skipped(reason) => {
                run.opened(index, false)?;
                return Ok((Turn::unanswered(signature, *reason), None));
            }
            Status::Repeat(answers) => {
                run.lock().trajectory[index].answered(signature, decided(answers));
                run.opened(index, false)?;
                let turn = Turn {
                    answers: (**answers).clone(),
                    observations: IndexMap::new(),
                    reason: Reason::Repeat,
                    arbitration: None,
                };
                return Ok((turn, None));
            }
            Status::Admitted => {}
        }
        let mut active = Active {
            run: run.clone(),
            depth,
            index,
            limit: request.limit,
            path: path.clone(),
            module: self.clone(),
            budgets: budgets.clone(),
            state: state.clone(),
            tool: None,
            cuts: Arc::default(),
            stop: stop.clone(),
            branches: None,
            late: false,
            wake: None,
        };
        // Only the output is taken out: an input of the same name is still shown.
        let asked = opened.only.keys().fold(signature.clone(), |asked, name| asked.without_output(name));
        let mut answered = IndexMap::new();
        // What a fit left one option: answered as the one-option choices of the signature are.
        let mut left_one: IndexMap<String, Decision> = IndexMap::new();
        if asked.outputs().is_empty() {
            run.opened(index, false)?;
        } else {
            let (mut asked, mut shown) = (asked, request.shown.clone());
            if !config.fits.is_empty() {
                // A request that is more than one request can hold is cut down first, by whatever
                // a fit asks. What it asks are sub-calls of this one.
                let call = Call {
                    active: active.clone(),
                    rdm: self.clone(),
                    prediction: None,
                    inputs: request.inputs.clone(),
                    shown: request.shown.clone(),
                    iteration,
                    role: Role::Fit,
                };
                let mut inputs = Inputs::from((*request.shown).clone());
                for fit in &config.fits {
                    (asked, inputs) = fit.fit(asked, inputs, &call)?;
                }
                shown = Arc::new(inputs.snapshot()?);
                run.lock().trajectory[index].inputs = Json::Object((*shown).clone());
                // A choice a fit cut down to one option leaves nothing to decide either: it is
                // answered with that option, and only the rest is asked.
                let outputs = asked.outputs();
                left_one = outputs
                    .iter()
                    .filter_map(|(name, output)| Some((name.clone(), output.only()?)))
                    .collect();
                asked = left_one.keys().fold(asked.clone(), |asked, name| asked.without_output(name));
            }
            if asked.outputs().is_empty() {
                // Nothing is sent, so the request gives back the place it was admitted to.
                {
                    let mut state = run.lock();
                    budgets.iter().for_each(|budget| state.budgets[*budget].predictor_calls -= 1);
                }
                run.opened(index, false)?;
            } else {
                match self.ask(run, config, index, &asked, &shown, budgets)? {
                    Some(decisions) => answered = decisions,
                    None => return Ok((Turn::unanswered(signature, Reason::MaxTokens), None)),
                }
            }
        }
        let mut answers = Answers::new();
        for name in signature.outputs().keys() {
            let unasked = || left_one.get(name).or_else(|| opened.only.get(name)).cloned();
            let answer = answered.shift_remove(name).or_else(unasked);
            let answer = answer.ok_or_else(|| {
                Error::Usage(format!(
                    "A fit returned a signature without the output '{name}' of the request it was given."
                ))
            })?;
            answers.insert(name.clone(), Some(answer));
        }
        {
            let mut state = run.lock();
            let entry = &mut state.trajectory[index];
            if !opened.only.is_empty() || !left_one.is_empty() {
                entry.tokens.get_or_insert(0);
                entry.answered(signature, decided(&answers));
                let mut unasked: Vec<String> = opened.only.keys().chain(left_one.keys()).cloned().collect();
                unasked.sort();
                entry.unasked = Some(unasked);
            }
            entry.observations = Some(IndexMap::new());
        }
        // What is asked from here on, below this request or after it, has this request on its way.
        let answers = Arc::new(answers);
        active.path = Some(Arc::new(Asked {
            key: request.key,
            answers: answers.clone(),
            before: path.clone(),
        }));

        // An output named after a tool calls it: a yes calls it, and so does a chosen option. An
        // output that points calls the tool it points at. Each is (selecting output, tool).
        let tools = self.map(run, config)?;
        let pointer = signature.pointer();
        let decision = |name: &str| answers.get(name).and_then(Option::as_ref);
        let mut selected: Vec<(String, Tool)> = Vec::new();
        for (name, output) in signature.outputs() {
            if Some(name.as_str()) != pointer && decision(name).is_some_and(|answer| output.selects(answer)) {
                selected.extend(tools.get(name).map(|tool| (name.clone(), tool)));
            }
        }
        let pointed = pointer.and_then(|name| Some((name, decision(name)?.as_choice()?.value.as_str())));
        let mut finished = false;
        if let Some((name, at)) = pointed {
            match tools.get(at) {
                Some(tool) => selected.push((name.to_owned(), tool)),
                None => finished = at == FINISH,
            }
        }
        if depth >= request.limit {
            // The leaf: at max_depth a call is answered, and what it selected is not run. When the
            // caller asked for the answers only, nothing was kept from being done.
            if !selected.is_empty() && !request.answers_only {
                active.cuts.lock().unwrap_or_else(PoisonError::into_inner).push(Reason::MaxDepth);
            }
            selected.clear();
            finished = false;
        }
        let prediction = Arc::new(Prediction::of(signature, (*answers).clone()));
        // Every node runs what its answers selected the same way, with no arbiter as with one.
        if !selected.is_empty() {
            let settled = Settling {
                run,
                config,
                active: &active,
                request,
                prediction: &prediction,
                iteration,
                index,
            };
            return settled.settle(config.arbiter.clone(), &tools, selected, finished);
        }
        let calls = selected.iter().map(|(selector, tool)| {
            let name: Arc<str> = tool.name().into();
            let call = Call {
                active: Active {
                    tool: Some(name.clone()),
                    ..active.clone()
                },
                rdm: self.clone(),
                prediction: Some(prediction.clone()),
                inputs: request.inputs.clone(),
                shown: request.shown.clone(),
                iteration,
                role: Role::Tool(name),
            };
            move || -> Result<Json> { running(run, index, selector.clone(), tool, &call) }
        });
        let results = in_parallel(config.threads, calls.collect())
            .into_iter()
            .collect::<Result<Vec<Json>>>()?;
        let mut step = finished.then(|| Step {
            name: FINISH.to_owned(),
            result: Json::Null,
            path: active.path.clone(),
        });
        let mut observations = IndexMap::new();
        for ((selector, tool), result) in selected.into_iter().zip(results) {
            if Some(selector.as_str()) == pointer {
                step = Some(Step {
                    name: tool.name().to_owned(),
                    result: result.clone(),
                    path: active.path.clone(),
                });
            }
            observations.insert(selector, result);
        }
        run.lock().trajectory[index].observations = Some(observations.clone());
        // A limit that stopped something below this call stopped it for this call.
        let reason = Reason::of(active.cuts.lock().unwrap_or_else(PoisonError::into_inner).iter().copied());
        let turn = Turn {
            answers: (*answers).clone(),
            observations,
            reason,
            arbitration: None,
        };
        Ok((turn, step))
    }

    /// Put one admitted request to the model and record what it answered. Returns None for one that was not asked.
    fn ask(
        &self,
        run: &Run,
        config: &Config,
        index: usize,
        signature: &Signature,
        shown: &Map<String, Json>,
        budgets: &[usize],
    ) -> Result<Option<IndexMap<String, Decision>>> {
        {
            let mut state = run.lock();
            if budgets.iter().any(|budget| state.budgets[*budget].out_of_tokens()) {
                // What was asked before it has used the tokens since it was admitted.
                state.trajectory[index].skipped = Some(Reason::MaxTokens);
                budgets.iter().for_each(|budget| state.budgets[*budget].predictor_calls -= 1);
                drop(state);
                run.opened(index, false)?;
                return Ok(None);
            }
        }
        let model = config.model.as_ref().ok_or_else(|| {
            Error::Usage("The module has no decision model: give it one with `model(...)` or `set_model(...)`.".to_owned())
        })?;
        let request = signature.request(shown)?;
        run.opened(index, true)?;
        let answered = model.decide(&request);
        run.emit(|run| {
            let name = model.name();
            let params = request
                .questions
                .keys()
                .map(|output| (output.clone(), self.params_for(config, output, signature)))
                .filter(|(_, params)| *params != Params::default())
                .collect();
            let (response, error) = match &answered {
                Ok(response) => (Some(response.clone()), None),
                Err(error) => (None, Some(error.clone())),
            };
            Event::Asked {
                run,
                index,
                key: systemone::key(&request, &name),
                model: name,
                request: Some(request.clone()),
                response,
                error,
                params,
            }
        })?;
        let response = answered?;
        let tokens = response.usage.total();
        let mut decisions = IndexMap::new();
        for (name, question) in &request.questions {
            let answer = response
                .answers
                .get(name)
                .ok_or_else(|| systemone::Error::Evidence(format!("The model gave no answer for '{name}'.")))?;
            let params = self.params_for(config, name, signature);
            let decision = systemone::decide(name, question, answer, &params).map_err(|error| match error {
                // The numbers a decision is made with are the module's: a wrong one is a wrong use of it.
                systemone::Error::Invalid(message) => Error::Usage(message),
                other => Error::Model(other),
            })?;
            decisions.insert(name.clone(), decision);
        }
        let mut state = run.lock();
        budgets.iter().for_each(|budget| state.budgets[*budget].tokens += tokens);
        let entry = &mut state.trajectory[index];
        entry.tokens = Some(tokens);
        // Which model answered: a run may put its requests to more than one.
        entry.model = Some(if response.model.is_empty() { model.name() } else { response.model });
        entry.answered(signature, decisions.clone());
        Ok(Some(decisions))
    }

    /// The numbers an output of a request is decided with.
    ///
    /// Parameters apply by output name. Those fitted to one answer space, a choice's weights
    /// say, are not another's: a request that asks an output the module declares over other
    /// options is answered without them.
    fn params_for(&self, config: &Config, name: &str, asked: &Signature) -> Params {
        let Some(params) = config.params.get(name) else {
            return Params::default();
        };
        let same = |declared: &crate::Output| asked.output(name).is_some_and(|output| output.space == declared.space);
        match self.inner.signature.output(name) {
            Some(declared) if !same(declared) => Params::default(),
            _ => params.clone(),
        }
    }
}

impl fmt::Debug for Rdm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let config = self.config();
        let mut module = f.debug_struct("Rdm");
        module
            .field("id", &self.inner.id)
            .field("signature", &self.inner.signature)
            .field("tools", &self.inner.tools);
        module.field("max_iters", &config.max_iters).field("max_depth", &config.max_depth);
        module
            .field("max_predictor_calls", &config.max_predictor_calls)
            .field("max_tokens", &config.max_tokens)
            .finish()
    }
}

/// What changed in a request's branches since a ruling saw them, as far as the ruling goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Since {
    /// Nothing it would act on otherwise.
    Nothing,
    /// A branch it used said more so far, and none came back.
    Said,
    /// A branch came back.
    Back,
}

/// How deep a request may run tools: under a limit, with what the call asked for itself, or as an earlier turn settled it.
enum Depth {
    Under(usize, Option<usize>),
    Kept(usize, bool),
}

/// A request, ready to be recorded: the questions as they are offered, and the inputs as they are now.
struct Request {
    signature: Signature,
    /// The inputs as they were given: a live one is read again at the next turn.
    inputs: Inputs,
    /// What they showed when the request was made.
    shown: Arc<Map<String, Json>>,
    /// The depth at which tools are no longer run.
    limit: usize,
    answers_only: bool,
    key: Key,
}

/// A request that has been recorded.
struct Opened {
    index: usize,
    /// The outputs that had one option, each with its answer.
    only: IndexMap<String, Decision>,
    status: Status,
}

enum Status {
    Admitted,
    Skipped(Reason),
    Repeat(Arc<Answers>),
}

/// What one turn came to.
struct Turn {
    answers: Answers,
    observations: IndexMap<String, Json>,
    reason: Reason,
    arbitration: Option<Arbitration>,
}

impl Turn {
    /// The answers of a request a limit kept from being asked: every output absent.
    fn unanswered(signature: &Signature, reason: Reason) -> Self {
        Turn {
            answers: signature.outputs().keys().map(|name| (name.clone(), None)).collect(),
            observations: IndexMap::new(),
            reason,
            arbitration: None,
        }
    }
}

/// The tool an output pointed at, what it returned, and the requests asked on the way.
struct Step {
    name: String,
    result: Json,
    path: Path,
}

/// One module's limits on requests and tokens, and what has been used under it in a run.
pub(crate) struct Budget {
    max_predictor_calls: usize,
    max_tokens: Option<u64>,
    predictor_calls: usize,
    pub(crate) tokens: u64,
}

impl Budget {
    fn new(config: &Config) -> Self {
        Budget {
            max_predictor_calls: config.max_predictor_calls,
            max_tokens: config.max_tokens,
            predictor_calls: 0,
            tokens: 0,
        }
    }

    fn out_of_tokens(&self) -> bool {
        self.max_tokens.is_some_and(|most| self.tokens >= most)
    }

    /// The limit that is used up, if one is.
    fn spent(&self) -> Option<Reason> {
        if self.predictor_calls >= self.max_predictor_calls {
            return Some(Reason::MaxPredictorCalls);
        }
        self.out_of_tokens().then_some(Reason::MaxTokens)
    }

    pub(crate) fn left(&self) -> usize {
        self.max_predictor_calls.saturating_sub(self.predictor_calls)
    }
}

/// A run on its way to its end: if it does not come to it, by an error or a panic, what it let
/// finish unwaited is stopped.
struct Unended {
    run: Arc<Run>,
    ended: bool,
}

impl Drop for Unended {
    fn drop(&mut self) {
        if !self.ended {
            let shadows = self.run.shadows.lock().unwrap_or_else(PoisonError::into_inner);
            shadows.iter().for_each(|stop| stop.set());
        }
    }
}

/// One run: the history of every request made for it.
#[derive(Default)]
pub(crate) struct Run {
    state: Mutex<RunState>,
    /// Its number, in its events.
    number: u64,
    /// What its events are handed to, if anything.
    record: Option<record::Shared>,
    /// What the record refused with, once it has: every event after it is refused the same.
    refused: Mutex<Option<Error>>,
    /// The branches let finish unwaited in this run: stopped after all if the run fails.
    pub(crate) shadows: Mutex<Vec<Arc<Stop>>>,
    /// What was done outside in this run, as each was done ([`Call::acting`]).
    pub(crate) receipts: Mutex<Vec<crate::prediction::Receipt>>,
}

#[derive(Default)]
pub(crate) struct RunState {
    pub(crate) trajectory: Vec<Entry>,
    /// What the run holds for each module, by the module's number.
    tools: HashMap<u64, Mine>,
    /// The names each of a module's sources has on its map in this run.
    offered: HashMap<(u64, usize), HashSet<String>>,
    /// The budget of every module that has joined the run: the run's own first.
    pub(crate) budgets: Vec<Budget>,
}

/// What a run holds for one module.
#[derive(Clone)]
struct Mine {
    /// The module's tools for this run: a copy of its own, which the run's tools may change.
    tools: Tools,
    /// Held while the module's sources are read and what they offer is put on the map.
    reading: Arc<Mutex<()>>,
}

impl Run {
    /// A run, written down to `record` if there is one.
    fn new(record: Option<record::Shared>) -> Self {
        Run {
            number: RUNS.fetch_add(1, Ordering::Relaxed),
            record,
            ..Run::default()
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, RunState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Hand the record an event, made only if there is a record to hand it to. Never called
    /// with the run locked: a record may take its time.
    pub(crate) fn emit(&self, event: impl FnOnce(u64) -> Event) -> Result<()> {
        let Some(record) = &self.record else {
            return Ok(());
        };
        let mut refused = self.refused.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(error) = &*refused {
            return Err(error.clone());
        }
        drop(refused);
        let kept = record.record(&event(self.number));
        if let Err(error) = &kept {
            refused = self.refused.lock().unwrap_or_else(PoisonError::into_inner);
            refused.get_or_insert_with(|| error.clone());
        }
        kept
    }

    /// Hand the record the `opened` event of a request, as the trajectory has it now. Its
    /// inputs go with it unless an `asked` follows, which shows them.
    fn opened(&self, index: usize, asked: bool) -> Result<()> {
        if self.record.is_none() {
            return Ok(());
        }
        let entry = self.lock().trajectory[index].clone();
        self.emit(|run| Event::Opened {
            run,
            index,
            parent: entry.parent,
            depth: entry.depth,
            iteration: entry.iteration,
            tool: entry.tool_name,
            inputs: (!asked).then_some(entry.inputs),
            repeat: entry.repeat,
            skipped: entry.skipped,
        })
    }
}

/// A request asked on the way to another, with the answers it got and the one asked before it.
struct Asked {
    key: Key,
    answers: Arc<Answers>,
    before: Path,
}

/// Every request asked on the way to one, the nearest first.
type Path = Option<Arc<Asked>>;

/// A long loop makes a long path: it is taken apart one request at a time, not one inside another.
impl Drop for Asked {
    fn drop(&mut self) {
        let mut before = self.before.take();
        while let Some(asked) = before.and_then(Arc::into_inner) {
            let mut asked = asked;
            before = asked.before.take();
        }
    }
}

/// The answers a request got when it was asked on the way here, if it was.
fn asked_before(path: &Path, key: &Key) -> Option<Arc<Answers>> {
    let mut at = path.as_ref();
    while let Some(asked) = at {
        if asked.key == *key {
            return Some(asked.answers.clone());
        }
        at = asked.before.as_ref();
    }
    None
}

/// The request whose tool is running: what a call made from inside that tool is a sub-call of.
#[derive(Clone)]
pub(crate) struct Active {
    pub(crate) run: Arc<Run>,
    pub(crate) depth: usize,
    pub(crate) index: usize,
    /// The depth at which a call made from here is answered and its tools are not run.
    limit: usize,
    /// Every request asked on the way here, each with the answers it got.
    path: Path,
    /// The module whose tool is running.
    pub(crate) module: Rdm,
    /// The budget of every module the request is under: the run's first.
    pub(crate) budgets: Arc<[usize]>,
    /// What tools read and no request shows: the nearest call's that gave one.
    pub(crate) state: Arc<Json>,
    /// The tool that is running.
    tool: Option<Arc<str>>,
    /// What stopped something in a call made from here, other than a decision.
    cuts: Arc<Mutex<Vec<Reason>>>,
    /// What the branch the running tool is, or is under, is stopped with: nothing more is begun
    /// for a stopped branch. None outside every arbitrated request.
    pub(crate) stop: Option<Arc<Stop>>,
    /// The branches of the request whose tool is running, where it is arbitrated.
    pub(crate) branches: Option<Branches>,
    /// Whether the running tool was started by arbitration and not by the request's answers.
    pub(crate) late: bool,
    /// What tells the request's arbitration that the running tool has something to show.
    pub(crate) wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Active {
    /// Whether the branch the running tool is, or is under, has been stopped.
    pub(crate) fn stopped(&self) -> bool {
        self.stop.as_ref().is_some_and(|stop| stop.is_set())
    }

    /// Nothing, unless the branch has been stopped: then what it is stopped with. A run whose
    /// record has refused an event has ended: then what it refused with.
    pub(crate) fn go_on(&self) -> Result<()> {
        if let Some(refused) = self.run.refused.lock().unwrap_or_else(PoisonError::into_inner).clone() {
            return Err(refused);
        }
        match self.stopped() {
            true => Err(stopped()),
            false => Ok(()),
        }
    }
}

/// What a branch that was stopped ends with.
pub(crate) fn stopped() -> Error {
    Error::Stopped("The branch was stopped: what it was for is settled without it.".to_owned())
}

/// Run a tool for a request, with its intent kept before it runs and what it returned after.
fn running(run: &Arc<Run>, index: usize, output: String, tool: &Tool, call: &Call) -> Result<Json> {
    // A branch stopped while its request was being asked starts no tool that request selected.
    call.go_on()?;
    let tool_name = tool.name().to_owned();
    // The intent is kept before the tool runs, and a record that refuses it keeps it from running.
    run.emit(|run| Event::Started {
        run,
        index,
        output: output.clone(),
        tool: tool_name.clone(),
    })?;
    let observation = run_tool(tool, call)?;
    run.emit(|run| Event::Returned {
        run,
        index,
        output,
        tool: tool_name,
        observation: observation.clone(),
    })?;
    Ok(observation)
}

/// A request whose selected tools are being settled by an arbiter.
struct Settling<'a> {
    run: &'a Arc<Run>,
    config: &'a Config,
    active: &'a Active,
    request: &'a Request,
    prediction: &'a Arc<Prediction>,
    iteration: usize,
    index: usize,
}

/// The stops of every tool of a settled request, set however the request is left: by an
/// error, a panic or its end. On its end each that still runs has been stopped already, or is
/// let finish unwaited, and is let be; on an error none of them is wanted, those let finish
/// included, and nothing more is begun for any. It stops, and undoes nothing: what one of
/// them is in the middle of doing outside, it finishes.
struct StopAll {
    stops: Vec<Arc<Stop>>,
    /// Whether the request came to its end, and did not fail.
    ended: bool,
}

impl Drop for StopAll {
    fn drop(&mut self) {
        let ended = self.ended;
        self.stops
            .iter()
            .filter(|stop| !(ended && stop.shadowed_here()))
            .for_each(|stop| stop.set());
    }
}

/// One tool of a settled request: a branch, or one it runs and waits for unasked.
struct Running {
    output: String,
    tool: Tool,
    /// A branch arbitration is asked about; else the pointer's tool, or one the arbiter leaves alone.
    branch: bool,
    stop: Arc<Stop>,
}

impl Settling<'_> {
    /// Start the tool at `at`, without waiting for it.
    fn start(&self, arrivals: &mut Arrivals<Json>, all: &[Running], branches: &Branches, at: usize, late: bool) {
        let one = &all[at];
        let name: Arc<str> = one.tool.name().into();
        let call = Call {
            active: Active {
                tool: Some(name.clone()),
                stop: Some(one.stop.clone()),
                branches: Some(branches.clone()),
                late,
                wake: Some(Arc::new(arrivals.waker())),
                ..self.active.clone()
            },
            rdm: self.active.module.clone(),
            prediction: Some(self.prediction.clone()),
            inputs: self.request.inputs.clone(),
            shown: self.request.shown.clone(),
            iteration: self.iteration,
            role: Role::Tool(name),
        };
        let (run, index, output, tool) = (self.run.clone(), self.index, one.output.clone(), one.tool.clone());
        let branches = branches.clone();
        arrivals.start(at, move || {
            let came = running(&run, index, output.clone(), &tool, &call)?;
            // What it came back with is the ledger's at once, for its siblings to read and the
            // next ruling to see, whether or not a ruling is being asked now.
            if let Some(branch) = crate::arbitrate::lock(&branches)
                .iter_mut()
                .find(|branch| branch.output == output && branch.state == State::Running)
            {
                branch.state = State::Returned;
                branch.result = Some(came.clone());
            }
            Ok(came)
        });
    }

    /// Ask the arbiter about the branches as they are, and return its ruling, whether the
    /// arbiter failed to make it, the branches as it saw them, and the round of a ruling made
    /// before that stands in its place. An arbiter whose own working fails ([`Error::Tool`])
    /// rules as there were none, and its failure is recorded with the ruling and counted;
    /// unless `stands` gives a ruling made before, set aside, that stands in its place: then
    /// that ruling is this round's, with the branches as it saw them, and the record says which
    /// round it was made in (a `revived` event). A decision model that failed, or a replay that has
    /// no answer for the ruling, is the run's failure, as it is for any request: no ruling is
    /// made up in its place.
    fn rule(
        &self,
        arbiter: &dyn Arbiter,
        branches: &Branches,
        before: &[Ruling],
        round: usize,
        first: bool,
        stands: &dyn Fn() -> Option<(Ruling, Ledger)>,
    ) -> Result<(Ruling, bool, Ledger, Option<usize>)> {
        let answers = self
            .prediction
            .answers
            .keys()
            .map(|name| (name.clone(), self.prediction.value(name)));
        let ledger = Ledger {
            signature: self.request.signature.clone(),
            shown: (*self.request.shown).clone(),
            inputs: self.request.inputs.clone(),
            answers: answers.collect(),
            branches: crate::arbitrate::lock(branches).clone(),
            round,
            before: before.to_vec(),
            first,
            request: (self.run.number, self.index),
        };
        let name: Arc<str> = ARBITRATE.into();
        let call = Call {
            active: Active {
                tool: Some(name),
                branches: Some(branches.clone()),
                ..self.active.clone()
            },
            rdm: self.active.module.clone(),
            prediction: Some(self.prediction.clone()),
            inputs: self.request.inputs.clone(),
            shown: self.request.shown.clone(),
            iteration: self.iteration,
            role: Role::Arbitrate,
        };
        let (ruling, error, ledger, revived_from) = match arbiter.arbitrate(&ledger, &call) {
            Ok(ruling) => (ruling, None, ledger, None),
            Err(Error::Tool(error)) => match stands() {
                // What was decided before stands, on what it saw then: it is this round's.
                Some((ruling, then)) => {
                    let from = then.round;
                    (ruling, Some(error), Ledger { round, ..then }, Some(from))
                }
                None => (Ruling::all(&ledger), Some(error), ledger, None),
            },
            // A stop from above, a model that failed, a module used wrongly: the request's.
            Err(error) => return Err(error),
        };
        let fell_back = error.is_some() && revived_from.is_none();
        let saw = ledger
            .branches
            .iter()
            .filter(|branch| branch.shown().is_some())
            .map(|branch| {
                // What it had said so far, running or stopped, by its version; 0 for what it came back with.
                let version = match branch.state {
                    State::Running | State::Stopped => branch.version,
                    _ => 0,
                };
                (branch.output.clone(), version)
            })
            .collect();
        self.run.emit(|run| Event::Arbitrated {
            run,
            index: self.index,
            round,
            ruling: ruling.clone(),
            saw,
            error,
        })?;
        if let Some(from) = revived_from {
            self.run.emit(|run| Event::Revived {
                run,
                index: self.index,
                round,
                from,
            })?;
        }
        Ok((ruling, fell_back, ledger, revived_from))
    }

    /// Stop a branch that is running, and say so to the record. One that came back meanwhile
    /// is not stopped: what it came back with stands, for the next ruling.
    fn stop(&self, one: &Running, branches: &Branches, arrivals: &mut Arrivals<Json>, at: usize) -> Result<bool> {
        {
            let mut ledger = crate::arbitrate::lock(branches);
            let Some(branch) = ledger
                .iter_mut()
                .find(|branch| branch.output == one.output && branch.state == State::Running)
            else {
                return Ok(true);
            };
            // One in the middle of doing something outside is waited for: what it did is to be said.
            if !one.stop.stop_unless_acting() {
                return Ok(false);
            }
            branch.state = State::Stopped;
        }
        // One that has not started yet is never started.
        let started = !arrivals.withdraw(at);
        self.run.emit(|run| Event::Stopped {
            run,
            index: self.index,
            output: one.output.clone(),
            tool: one.tool.name().to_owned(),
            started,
        })?;
        Ok(true)
    }

    /// Let a running branch finish unwaited, and return true; or, where it is in the middle of
    /// doing something outside, keep waiting for it, and return false. One that came back
    /// meanwhile stands as it is.
    fn shadow(&self, one: &Running, branches: &Branches) -> bool {
        let mut ledger = crate::arbitrate::lock(branches);
        if let Some(branch) = ledger
            .iter_mut()
            .find(|branch| branch.output == one.output && branch.state == State::Running)
        {
            if !one.stop.shadow_unless_acting() {
                return false;
            }
            branch.state = State::Shadowed;
            self.run
                .shadows
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(one.stop.clone());
        }
        true
    }

    /// Run what the answers selected, settling it as it comes back, and return the turn.
    fn settle(
        &self,
        arbiter: Option<Arc<dyn Arbiter>>,
        tools: &Tools,
        selected: Vec<(String, Tool)>,
        finished: bool,
    ) -> Result<(Turn, Option<Step>)> {
        let (signature, active) = (&self.request.signature, self.active);
        let pointer = signature.pointer();
        let beside = arbiter.as_ref().map(|arbiter| arbiter.beside()).unwrap_or_default();
        // With no arbiter, every branch is waited for and used, and a failure or a panic goes on
        // to the caller once all have come back, as from tools run at once and joined: a panic
        // before any error, and of errors the one of the tool selected first.
        let mut panicked: Option<Box<dyn Any + Send>> = None;
        let mut errors: Vec<(usize, Error)> = Vec::new();
        let mut all: Vec<Running> = Vec::new();
        for (output, tool) in &selected {
            let branch = Some(output.as_str()) != pointer && !beside.contains(&tool.name().to_owned());
            all.push(Running {
                output: output.clone(),
                tool: tool.clone(),
                branch,
                stop: Stop::under(active.stop.clone()),
            });
        }
        // What arbitration may start: a tool an output of the request is named after, that
        // selects it, and that the answers did not select.
        for (name, output) in signature.outputs() {
            let unselected = !selected.iter().any(|(chosen, _)| chosen == name) && Some(name.as_str()) != pointer;
            if unselected
                && arbiter.is_some()
                && !output.is_pointer()
                && !beside.contains(name)
                && let Some(tool) = tools.get(name)
            {
                all.push(Running {
                    output: name.clone(),
                    tool,
                    branch: true,
                    stop: Stop::under(active.stop.clone()),
                });
            }
        }
        let ledger: Vec<Branch> = all
            .iter()
            .enumerate()
            .filter(|(_, one)| one.branch)
            .map(|(at, one)| Branch {
                output: one.output.clone(),
                tool: one.tool.name().to_owned(),
                about: one.tool.description().to_owned(),
                selected: at < selected.len(),
                state: State::Idle,
                result: None,
                partial: None,
                version: 0,
            })
            .collect();
        let branches: Branches = Arc::new(Mutex::new(ledger));
        let mut left = StopAll {
            stops: all.iter().map(|one| one.stop.clone()).collect(),
            ended: false,
        };
        let at_of = |output: &str| all.iter().position(|one| one.branch && one.output == output);
        // With no arbiter nothing is settled while a lone branch runs: it runs here, as one did.
        let mut arrivals = match arbiter {
            None => Arrivals::new(self.config.threads).alone_here(),
            Some(_) => Arrivals::new(self.config.threads),
        };
        let mut results: Vec<Option<Json>> = vec![None; all.len()];
        let (mut rounds, mut started_late, mut fallbacks) = (0, Vec::new(), 0);
        // The last ruling, and the branches as it saw them: what the request comes to.
        let mut last: Option<(Ruling, Ledger)> = None;
        // Every ruling so far, oldest first: what each later one is shown was tried.
        let mut before: Vec<Ruling> = Vec::new();
        // The rulings set aside since the last one acted on, only by what the branches they
        // used said since, each with the branches as it saw them.
        let mut aside: Vec<(Ruling, Ledger)> = Vec::new();
        // The round of a ruling held that stood, and of the ruling after it that could not be made.
        let mut reused: Option<(usize, usize)> = None;
        let begin = |arrivals: &mut Arrivals<Json>, at: usize, late: bool| {
            if let Some(branch) = crate::arbitrate::lock(&branches)
                .iter_mut()
                .find(|branch| branch.output == all[at].output)
            {
                branch.state = State::Running;
            }
            self.start(arrivals, &all, &branches, at, late);
        };
        // What runs unasked starts first, then the branches the answers selected: all of them, or
        // those a first ruling starts now.
        for at in (0..selected.len()).filter(|at| !all[*at].branch) {
            begin(&mut arrivals, at, false);
        }
        let chosen: Vec<usize> = (0..selected.len()).filter(|at| all[*at].branch).collect();
        let now: Vec<usize> = match arbiter.as_ref().is_some_and(|arbiter| arbiter.first()) && chosen.len() > 1 {
            true => {
                let Some(arbiter) = &arbiter else {
                    unreachable!("asked first only of an arbiter")
                };
                let (ruling, fell_back, _, _) = self.rule(arbiter.as_ref(), &branches, &before, rounds, true, &|| None)?;
                before.push(ruling.clone());
                rounds += 1;
                fallbacks += usize::from(fell_back);
                let mut now: Vec<usize> = ruling.start.iter().filter_map(|output| at_of(output)).collect();
                // Holding back everything the answers selected would leave nothing to settle.
                if !now.iter().any(|at| chosen.contains(at)) {
                    now.extend(chosen.iter().copied());
                }
                now.sort_unstable();
                now.dedup();
                now
            }
            false => chosen.clone(),
        };
        for at in now {
            let late = !chosen.contains(&at);
            if late {
                started_late.push(all[at].output.clone());
            }
            begin(&mut arrivals, at, late);
        }
        let state_of = |output: &str| {
            crate::arbitrate::lock(&branches)
                .iter()
                .find(|branch| branch.output == output)
                .map(|branch| branch.state)
        };
        let mut over = false;
        while let Some(first) = arrivals.next() {
            let mut came = vec![first];
            came.extend(std::iter::from_fn(|| arrivals.ready()));
            let mut settles = false;
            for came in came {
                let (at, result) = match came {
                    Came::Back(at, result) => (at, result),
                    // A branch has something to show before it comes back: a ruling sees it.
                    Came::Woken => {
                        settles = true;
                        continue;
                    }
                };
                let one = &all[at];
                let result = match (result, &arbiter) {
                    (Ok(result), _) => result,
                    (Err(panic), Some(_)) => std::panic::resume_unwind(panic),
                    (Err(panic), None) => {
                        panicked.get_or_insert(panic);
                        continue;
                    }
                };
                let let_be = one.branch && matches!(state_of(&one.output), Some(State::Stopped | State::Shadowed));
                match result {
                    // A branch that was stopped has ended, as it was asked to, or has come
                    // back after it no longer mattered, or was let finish unwaited: what it
                    // came to is in the record.
                    _ if let_be => {}
                    Err(error) if arbiter.is_none() => errors.push((at, error)),
                    Err(error) => return Err(error),
                    // A branch's result is on the ledger already, put there as it came back.
                    Ok(_) if one.branch => settles = true,
                    Ok(result) => results[at] = Some(result),
                }
            }
            if over || !settles {
                continue;
            }
            let Some(arbiter) = &arbiter else { continue };
            // What came back, or was said so far, while a ruling was asked was not seen by it.
            // What a branch said so far since counts only where the ruling used it: it would
            // publish what is no longer so. One the ruling stops or leaves unused may say more
            // faster than a ruling is made, and would otherwise set every ruling aside; one it
            // waits for that said more is ruled on next (`came_since`).
            let since = |seen: &Ledger, ruling: &Ruling| {
                let now = crate::arbitrate::lock(&branches);
                let back = now.iter().any(|branch| {
                    let then = seen.branch(&branch.output);
                    branch.state == State::Returned && then.is_none_or(|then| then.state != State::Returned)
                });
                let said = now.iter().any(|branch| {
                    let then = seen.branch(&branch.output);
                    branch.state == State::Running
                        && branch.version > 0
                        && then.is_none_or(|then| then.version != branch.version)
                        && ruling.used.contains(&branch.output)
                });
                match (back, said) {
                    (true, _) => Since::Back,
                    (false, true) => Since::Said,
                    (false, false) => Since::Nothing,
                }
            };
            // Where this ruling cannot be made, the last of the rulings set aside since the last
            // acted on stands in its place, if they were set aside by no more than what the
            // branches they used said since and all came to the same: that is what was decided.
            let stands = || {
                let (last, then) = aside.last()?;
                let agreed = aside.iter().all(|(one, _)| one.same_as(last));
                (agreed && since(then, last) != Since::Back).then(|| (last.clone(), then.clone()))
            };
            let (ruling, fell_back, seen, revived_from) = self.rule(arbiter.as_ref(), &branches, &before, rounds, false, &stands)?;
            rounds += 1;
            fallbacks += usize::from(fell_back);
            match revived_from {
                // It is acted on as this round's ruling, made in the round it came from: the one
                // whose place it stands in was never made, and the one it was made in was set
                // aside, and stays so.
                Some(from) => {
                    reused = Some((from, seen.round));
                    aside.clear();
                    arbiter.revived(&seen, &ruling, from);
                }
                None => match since(&seen, &ruling) {
                    Since::Nothing => {
                        aside.clear();
                        arbiter.acted(&seen, &ruling, true);
                    }
                    // Set aside by what a branch it used said since, it is acted on where the
                    // ruling set aside before it came to the same: the decision has settled, though
                    // the branch says more. It stands on what it saw.
                    Since::Said if aside.last().is_some_and(|(last, _)| last.same_as(&ruling)) => {
                        aside.clear();
                        arbiter.acted(&seen, &ruling, true);
                    }
                    // A ruling on what is no longer so is not acted on: what came since is ruled
                    // on next.
                    why => {
                        arbiter.acted(&seen, &ruling, false);
                        match why {
                            Since::Said => aside.push((ruling, seen)),
                            _ => aside.clear(),
                        }
                        continue;
                    }
                },
            }
            // Only a ruling acted on is one a later ruling is shown was tried.
            before.push(ruling.clone());
            let mut wanted = false;
            for (at, one) in all.iter().enumerate().filter(|(_, one)| one.branch) {
                match state_of(&one.output) {
                    Some(State::Running) if ruling.wait.contains(&one.output) => wanted = true,
                    Some(State::Running) if ruling.shadow.contains(&one.output) => wanted |= !self.shadow(one, &branches),
                    Some(State::Running) => wanted |= !self.stop(one, &branches, &mut arrivals, at)?,
                    _ => {}
                }
            }
            let mut starting = false;
            for output in &ruling.start {
                if let Some(at) = at_of(output).filter(|_| state_of(output) == Some(State::Idle)) {
                    starting = true;
                    let late = !chosen.contains(&at);
                    if late {
                        started_late.push(output.clone());
                    }
                    begin(&mut arrivals, at, late);
                }
            }
            // A branch the ruling waited for that came back, or said more, while the ruling was
            // acted on keeps the request from ending without it: that is ruled on next. One the
            // ruling used, or did not want, stands as the ruling saw it.
            let came_since = crate::arbitrate::lock(&branches).iter().any(|branch| {
                let then = seen.branch(&branch.output);
                let back = branch.state == State::Returned && then.is_none_or(|then| then.state != State::Returned);
                let said = branch.version > 0 && then.is_none_or(|then| then.version != branch.version);
                ruling.wait.contains(&branch.output) && (back || said)
            });
            last = Some((ruling, seen));
            // Over when nothing it still waits for runs and nothing starts: a running branch is
            // stopped unless it is still waited for or let finish, so a request the ruling holds
            // answered with one still waited for waits for it.
            if !wanted && !starting && !came_since {
                over = true;
                for (at, one) in all.iter().enumerate().filter(|(_, one)| one.branch) {
                    // One in the middle of doing something outside keeps the request waiting.
                    over &= self.stop(one, &branches, &mut arrivals, at)?;
                }
            }
            // Over, with nothing left that runs unasked: the request does not wait for what was
            // stopped, nor for what it let finish.
            let unasked = all.iter().enumerate().any(|(at, one)| !one.branch && results[at].is_none());
            if over && !unasked {
                break;
            }
        }
        // A failure or a panic of a node with no arbiter goes on once every branch has come back.
        if let Some(panic) = panicked {
            std::panic::resume_unwind(panic);
        }
        if let Some((_, error)) = errors.into_iter().min_by_key(|(at, _)| *at) {
            return Err(error);
        }
        let ledger = crate::arbitrate::lock(&branches).clone();
        let (ruling, seen) = match last {
            Some((ruling, seen)) => (ruling, Some(seen)),
            None => (Ruling::default(), None),
        };
        // A branch is used for what it had to show when the last ruling saw it.
        let used: Vec<&Branch> = seen
            .iter()
            .flat_map(|seen| seen.showing())
            .filter(|branch| ruling.used.contains(&branch.output))
            .collect();
        let is_used = |output: &String| used.iter().any(|branch| branch.output == *output);
        let named = |state: State, selected: bool| -> Vec<String> {
            ledger
                .iter()
                .filter(|branch| branch.state == state && (!selected || branch.selected))
                .map(|branch| branch.output.clone())
                .collect()
        };
        let arbitration = Arbitration {
            used: used.iter().map(|branch| branch.output.clone()).collect(),
            partial: used
                .iter()
                .filter(|branch| matches!(branch.state, State::Running | State::Stopped))
                .map(|branch| branch.output.clone())
                .collect(),
            versions: used
                .iter()
                .filter(|branch| matches!(branch.state, State::Running | State::Stopped))
                .map(|branch| (branch.output.clone(), branch.version))
                .collect(),
            unused: named(State::Returned, false)
                .into_iter()
                .filter(|output| !is_used(output))
                .collect(),
            stopped: named(State::Stopped, false),
            shadowed: named(State::Shadowed, false),
            started: started_late,
            held: named(State::Idle, true),
            rounds,
            fallbacks,
            answered: ruling.done,
            unresolved: ruling.unresolved,
            conflict: ruling.conflict,
            same: ruling.same.clone(),
            reused,
        };
        let mut step = finished.then(|| Step {
            name: FINISH.to_owned(),
            result: Json::Null,
            path: active.path.clone(),
        });
        // A branch used is used for what the ruling saw of it, whatever it came to after; one
        // not used is kept for what it came back with, if it came back while it was wanted.
        let came_back = |output: &str| {
            let seen = used.iter().find(|branch| branch.output == output);
            let back = || {
                ledger
                    .iter()
                    .find(|branch| branch.output == output && branch.state == State::Returned)
                    .and_then(|branch| branch.result.clone())
            };
            seen.and_then(|branch| branch.shown().cloned()).or_else(back)
        };
        let mut observations = IndexMap::new();
        for (one, result) in all.iter().zip(results) {
            let result = match one.branch {
                true => came_back(&one.output),
                false => result,
            };
            let Some(result) = result else { continue };
            if Some(one.output.as_str()) == pointer {
                step = Some(Step {
                    name: one.tool.name().to_owned(),
                    result: result.clone(),
                    path: active.path.clone(),
                });
            }
            observations.insert(one.output.clone(), result);
        }
        self.run.lock().trajectory[self.index].observations = Some(observations.clone());
        let reason = Reason::of(active.cuts.lock().unwrap_or_else(PoisonError::into_inner).iter().copied());
        let turn = Turn {
            answers: self.prediction.answers.clone(),
            observations,
            reason,
            // A node with no arbiter says nothing of arbitration: every result stands.
            arbitration: arbiter.is_some().then_some(arbitration),
        };
        left.ended = true;
        Ok((turn, step))
    }
}

/// Run one tool and return what it returned.
fn run_tool(tool: &Tool, call: &Call) -> Result<Json> {
    let failed = |error: &str| Ok(Json::String(format!("Execution error in {}: {error}", tool.name())));
    // What a tool leaves half done when it panics is its own: the run's locks are taken whatever state they were left in.
    match catch_unwind(AssertUnwindSafe(|| tool.call(call))) {
        // A tool that fails is an observation and not the end of the run.
        Ok(Err(Error::Tool(error))) => failed(&error),
        // A module used wrongly inside a tool is the caller's error wherever it happens, and so is a model that failed.
        Ok(other) => other,
        // A tool that panics has failed too: an index out of bounds, a key that is not there.
        Err(panic) => failed(&said(panic.as_ref())),
    }
}

/// What a panic said.
fn said(panic: &(dyn Any + Send)) -> String {
    let text = panic.downcast_ref::<&str>().map(|text| (*text).to_owned());
    text.or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "the tool panicked".to_owned())
}

/// The answers that were given, without the places of those that were not.
fn decided(answers: &Answers) -> IndexMap<String, Decision> {
    answers
        .iter()
        .filter_map(|(name, answer)| Some((name.clone(), answer.clone()?)))
        .collect()
}

/// What makes two requests the same request: the same module's, with the same questions about the same state.
fn key(module: u64, signature: &Signature, shown: &Map<String, Json>) -> Key {
    let inputs = signature
        .inputs()
        .iter()
        .map(|(name, field)| json!([name, field.type_name, field.description, null]));
    let outputs = signature.outputs().iter().map(|(name, output)| {
        let kind = output.kind().map_or("tool", |kind| kind.as_str());
        json!([name, kind, output.description, output.criteria()])
    });
    let fields: Vec<Json> = inputs.chain(outputs).collect();
    // Written straight into the hash: the inputs may be long, and are not copied to be read.
    let mut hash = Sha256::new();
    if serde_json::to_writer(&mut hash, &(module, signature.instructions(), &fields, shown)).is_err() {
        // JSON values can always be written, and a hash takes whatever it is given.
        hash.update(b"unwritten");
    }
    hash.finalize().into()
}
