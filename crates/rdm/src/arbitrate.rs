//! Arbitration: many possibilities at once, settled into one stream of what is done.
//!
//! A request's answers select tools, and the tools run at once: that is what might be done,
//! each a branch. What is done is one thing after another. Arbitration is the request going
//! from the one to the other: each time a branch comes back, what has come back so far is
//! settled, and that settles what is done next. A branch that has come back answers the
//! request or does not; one still running is still wanted or is stopped; one not started is
//! started now or not; and the request is answered or it is not. The request is over when
//! nothing it still wants is running and nothing starts. It does not wait for a branch it no
//! longer wants.
//!
//! ```text
//! answer(request):
//!     answers <- predict(request)               the possibilities: what the answers select
//!     start the branches selected               none of them is waited for by the others
//!     while a branch is running:
//!         result <- whichever comes back first  and any that came back with it
//!         ruling <- arbitrate(branches as now)  which results answer it, which branches are still
//!                                               wanted, which to start, whether it is answered
//!         stop the running branches not wanted  start the ones the ruling starts
//!         if nothing wanted runs or starts:  the request is over
//!     return answers, results, what arbitration came to
//! ```
//!
//! An [`Arbiter`] makes the ruling. A module with none waits for every branch and uses every
//! result, as a module always has. [`Arbitrate`] asks the decision model: one request about
//! the branches as they are now, a sub-call of the request it settles, answers only, so it
//! selects no tool and is never arbitrated itself. It is recorded and replayed as any request
//! is. So the possibilities are as many as the answers select, and what is done is a sequence
//! of rulings, each on what had come back when it was made.
//!
//! A stopped branch is told so through its [`Call`] ([`Call::stopped`]): the module begins
//! nothing more for it, no request, no written reply and no tool, and a tool that does
//! something long asks between its steps. What it did before it was stopped stands, and a
//! tool that is in the middle of doing something outside finishes it: a branch is stopped,
//! never undone. Its events go on being recorded until it has ended, after the request has
//! moved on.
//!
//! A branch need not come back for what it has to count. One that works for a while says
//! what it has so far ([`Call::partial`]), and that is settled as a result is: a ruling may
//! use it and stop the branch, or wait for more. And a branch the request no longer waits for
//! may be let finish rather than stopped, its result kept to be held against what the request
//! came to: it is shadowed ([`Call::shadowed`]). The module cannot know what a host's tool
//! does outside, so a shadowed branch acting outside is for the host to refuse: a host that
//! does not, lets none be shadowed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use systemone::Json;

use crate::call::{Ask, Call};
use crate::error::{Error, Result};
use crate::inputs::Inputs;
use crate::module::Rdm;
use crate::observation::as_observation;
use crate::signature::Signature;
use systemone::DecisionModel;

/// The name a request's arbitration is recorded under, as the tool of the sub-call it asks.
pub const ARBITRATE: &str = "arbitrate";

/// Where a branch is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Not started: the answers did not select it, or arbitration held it before it started.
    Idle,
    /// Started, and not come back.
    Running,
    /// Come back, with what it came back with.
    Returned,
    /// Stopped by arbitration, before it started or while it ran. What it did stands, and so
    /// does what it had said so far: every later ruling is shown it.
    Stopped,
    /// Let finish, unwaited: the request no longer waits for it, and what it comes to is
    /// recorded to be held against what the request came to.
    Shadowed,
}

impl State {
    /// As a request shows it.
    pub fn as_str(self) -> &'static str {
        match self {
            State::Idle => "not started",
            State::Running => "running",
            State::Returned => "returned",
            State::Stopped => "stopped",
            State::Shadowed => "let finish",
        }
    }
}

/// A tool a request could run: one its answers selected, or one an output of its signature is
/// named after, which arbitration may start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    /// The output that selects it.
    pub output: String,
    /// The tool.
    pub tool: String,
    /// What the tool does, as the model is told.
    pub about: String,
    /// Whether the request's answers selected it.
    pub selected: bool,
    /// Where it is.
    pub state: State,
    /// What it came back with, once it has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Json>,
    /// What it has so far, while it runs, where it said ([`Call::partial`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial: Option<Json>,
    /// How many times it has said what it has so far: which of those `partial` is.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub version: usize,
}

fn is_zero(count: &usize) -> bool {
    *count == 0
}

impl Branch {
    /// What it has to show: what it came back with, or what it has so far while it runs or had
    /// when it was stopped.
    pub fn shown(&self) -> Option<&Json> {
        match self.state {
            State::Returned => self.result.as_ref(),
            // Stopped, what it had said so far stands, at the version it was stopped at: a ruling
            // that used it and stopped it leaves it for every later ruling to use, as it was.
            State::Running | State::Stopped => self.partial.as_ref(),
            _ => None,
        }
    }
}

/// A request's branches as they are now, to be settled.
#[derive(Clone, Debug)]
pub struct Ledger {
    /// The request's own signature.
    pub signature: Signature,
    /// What the request showed when it was asked: what its answers were given.
    pub shown: Map<String, Json>,
    /// The request's inputs as they were given: a live one ([`crate::Shared`]) shows what it
    /// holds now, which a tool may have changed since the request was asked.
    pub inputs: Inputs,
    /// Each of the request's answers, by output, as a plain value: a bool, a label, a number.
    pub answers: Map<String, Json>,
    /// Every branch, in the order of the request's outputs.
    pub branches: Vec<Branch>,
    /// How many rulings were made on the request before this one, set aside or acted on: its
    /// place among them, from 0. `before` holds only those acted on.
    pub round: usize,
    /// The rulings acted on before this one, oldest first: what was tried, which the branches
    /// show what came of. A ruling set aside because something came back while it was asked is
    /// not among them. Every ruling made, set aside or not, is an `arbitrated` event, and is
    /// counted in `rounds`.
    pub before: Vec<Ruling>,
    /// Whether it is the ruling before any branch has started ([`Arbiter::first`]): only which
    /// of those the answers selected start now is asked.
    pub first: bool,
    /// The request being settled, known by its run and its place in the run: the same for every
    /// ruling made on it, and no other request's. A host arbiter keeps what it works out across
    /// one request's rulings under this.
    pub request: (u64, usize),
}

impl Ledger {
    /// The branch selected by an output.
    pub fn branch(&self, output: &str) -> Option<&Branch> {
        self.branches.iter().find(|branch| branch.output == output)
    }

    /// The branches in one state.
    pub fn in_state(&self, state: State) -> impl Iterator<Item = &Branch> {
        self.branches.iter().filter(move |branch| branch.state == state)
    }

    /// The branches with something to show: come back, or running with what they have so far.
    pub fn showing(&self) -> impl Iterator<Item = &Branch> {
        self.branches.iter().filter(|branch| branch.shown().is_some())
    }
}

/// What arbitration rules about a request's branches as they are.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ruling {
    /// The branches whose results answer the request, or a part of it: what it comes to. A
    /// running branch can be used for what it has so far.
    pub used: Vec<String>,
    /// The running branches still waited for. A running branch neither waited for nor let
    /// finish is stopped.
    pub wait: Vec<String>,
    /// The running branches let finish without being waited for: the request does not wait for
    /// them, and what they come to is recorded, to be held against what it came to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shadow: Vec<String>,
    /// The branches not started that start now.
    pub start: Vec<String>,
    /// Whether the request is answered by what it comes to. It ends nothing by itself: the
    /// request is over when nothing it still waits for runs and nothing starts.
    pub done: bool,
    /// Whether what came back leaves the request unanswered, and nothing more could answer it:
    /// it is to be said what is not known.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unresolved: bool,
    /// Whether two results said different things about the same matter.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conflict: bool,
    /// Results that say the same as one shown before them, where it was asked
    /// ([`Arbitrate::agrees`]): each `[result, the earlier one, how likely]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub same: Vec<(String, String, f64)>,
}

impl Ruling {
    /// As a later ruling is shown it: what was to happen next, and to which branches.
    pub fn shown(&self) -> Json {
        let next = match (self.done, self.unresolved, self.start.first(), self.wait.is_empty()) {
            (true, _, _, _) => FINISHED.to_owned(),
            (_, true, _, _) => UNKNOWN.to_owned(),
            (_, _, Some(start), _) => format!("start_{start}"),
            (_, _, None, false) => WAIT.to_owned(),
            _ => FINISHED.to_owned(),
        };
        json!({"next": next, "used": self.used, "waited": self.wait, "started": self.start, "let_finish": self.shadow})
    }

    /// The ruling of a module with no arbiter: every result used, every running branch waited
    /// for, nothing started, and over only when nothing runs. It is also what arbitration
    /// comes to when the arbiter fails: the request goes on as if there were none.
    pub fn all(ledger: &Ledger) -> Ruling {
        let named = |state| ledger.in_state(state).map(|branch| branch.output.clone()).collect();
        Ruling {
            used: named(State::Returned),
            wait: named(State::Running),
            ..Ruling::default()
        }
    }

    /// Whether it comes to what another ruling does: the same branches used, waited for, let
    /// finish and started (so the same stopped), and the same said of the request, whatever
    /// the order they are named in and however likely each result is held to agree.
    pub fn same_as(&self, other: &Ruling) -> bool {
        fn same(one: &[String], other: &[String]) -> bool {
            let set = |names: &[String]| names.iter().cloned().collect::<std::collections::BTreeSet<String>>();
            set(one) == set(other)
        }
        same(&self.used, &other.used)
            && same(&self.wait, &other.wait)
            && same(&self.shadow, &other.shadow)
            && same(&self.start, &other.start)
            && (self.done, self.unresolved, self.conflict) == (other.done, other.unresolved, other.conflict)
    }
}

/// What settles a request's branches as they come back.
///
/// A module is given one with [`crate::Rdm::arbiter`]. It is asked each time branches come
/// back, or one that runs says what it has so far ([`Call::partial`]), about all of them as
/// they are then; and, where [`Arbiter::first`] says so, once before any has started, about
/// which of those the answers selected start now: the rest are held, to be started later or
/// not at all. What comes back while it is being asked is settled the next time, and the
/// request is not over before it is. It is not asked about the tool an output that points
/// selected, which runs and is waited for as in a module with no arbiter, nor about the tools
/// it names in [`Arbiter::beside`].
///
/// ```
/// use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
/// use rdm::{Call, Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// /// Settled by the first branch that comes back with something: it is used, and the rest are stopped.
/// struct First;
/// impl Arbiter for First {
///     fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
///         let used: Vec<String> = ledger.in_state(State::Returned).map(|branch| branch.output.clone()).take(1).collect();
///         Ok(Ruling { done: !used.is_empty(), used, ..Ruling::default() })
///     }
/// }
///
/// let ask = Signature::new("Find a word.").input("word").noul("near", "Look nearby?").noul("far", "Look far away?");
/// let near = Tool::new("near", "Look nearby.", |_| Ok("found it nearby"));
/// // Asks the module once more, and is stopped there: the request did not wait for it.
/// let far = Tool::new("far", "Look far away.", |call| {
///     while !call.stopped() {
///         std::thread::yield_now();
///     }
///     call.sub(inputs! { "word" => "again" })?;
///     Ok("found it far away")
/// });
/// let model = Scripted::new(|_| [("near", Answer::noul(0.9)), ("far", Answer::noul(0.9))]);
/// let result = Rdm::new(ask).tool(near).tool(far).arbiter(First).model(model).call(inputs! { "word" => "w" }).unwrap();
/// let arbitration = result.arbitration.unwrap();
/// assert_eq!(arbitration.used, ["near"]);
/// assert_eq!(arbitration.stopped, ["far"]);
/// assert_eq!(result.observations.get("far"), None);
/// ```
pub trait Arbiter: Send + Sync {
    /// Rule on the branches as they are. A ruling that names a branch where it is not (one
    /// not started as used, one come back as started) is read only where it fits. An error of
    /// its own working ([`Error::Tool`]), a spent budget among them, is no ruling, and the error
    /// is recorded. The request goes on as [`Ruling::all`] would have it, and its arbitration
    /// counts it among its `fallbacks`; unless the rulings set aside since the last acted on
    /// were set aside by no more than what the branches they used said since, and all came to
    /// the same. Then the last of them stands in its place: it is acted on as this round's, on
    /// the branches as it saw them, which may be older than what the branches have said since
    /// ([`Arbiter::revived`], the record's `revived` event, the arbitration's `reused`), and it
    /// is no fallback. Any other error, a decision model that failed, is the request's.
    fn arbitrate(&self, ledger: &Ledger, call: &Call) -> Result<Ruling>;

    /// Whether it is asked once before any branch starts, about which of those the answers
    /// selected start now. False unless an arbiter says otherwise. It is not asked where the
    /// answers selected fewer than two.
    fn first(&self) -> bool {
        false
    }

    /// The tools it leaves alone: they run where the answers select them and are waited for,
    /// as in a module with no arbiter, and it is not asked about them. None unless an arbiter
    /// says otherwise.
    fn beside(&self) -> Vec<String> {
        Vec::new()
    }

    /// Told, after each ruling it made after branches came back, whether the request acted on
    /// it: false where something came back while it was made, so it was set aside and is made
    /// again. Nothing, unless an arbiter says otherwise.
    fn acted(&self, ledger: &Ledger, ruling: &Ruling, acted: bool) {
        let _ = (ledger, ruling, acted);
    }

    /// Told, in place of [`Arbiter::acted`], that a round's ruling could not be made and the
    /// ruling made in round `from`, set aside then, was acted on as this round's: the ledger is
    /// this round's, with the branches as that ruling saw them. Round `from` was told it was not
    /// acted on, and stays so; every round is told once. Unless an arbiter says otherwise, it is
    /// told this round's ruling was acted on.
    fn revived(&self, ledger: &Ledger, ruling: &Ruling, from: usize) {
        let _ = from;
        self.acted(ledger, ruling, true);
    }
}

/// What arbitration came to for a request: in its prediction ([`crate::Prediction::arbitration`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Arbitration {
    /// The branches whose results the request comes to, by output, in the order of its outputs.
    pub used: Vec<String>,
    /// Of those, the ones used for what they had so far, while they ran: their observation is
    /// what they had then, and they were stopped or let finish.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partial: Vec<String>,
    /// Which of what each of those said it had so far was used: its count, from 1, as the
    /// `partial` events of the record number them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub versions: Vec<(String, usize)>,
    /// The branches that came back and were not used. Their results are in the observations.
    pub unused: Vec<String>,
    /// The branches stopped, before they started or while they ran. A branch stopped while it
    /// ran has no observation: what it came to, if it came to anything, is in the record.
    pub stopped: Vec<String>,
    /// The branches let finish after the request was over, not waited for: what they come to
    /// is in the record, to be held against what the request came to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shadowed: Vec<String>,
    /// The branches arbitration started that the answers had not selected.
    pub started: Vec<String>,
    /// The branches the answers selected that arbitration held and never started.
    pub held: Vec<String>,
    /// How many rulings there were.
    pub rounds: usize,
    /// How many of them the arbiter failed to make, and were those of no arbiter: every
    /// result used, every branch waited for. A request with any is not settled by arbitration
    /// alone, and should not be shown as if it were.
    pub fallbacks: usize,
    /// Whether the last ruling held the request answered by what it comes to.
    pub answered: bool,
    /// Whether the last ruling held it unanswered with nothing more to try: what is not known
    /// is to be said.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unresolved: bool,
    /// Whether the last ruling held that two results said different things about one matter.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conflict: bool,
    /// The results the last ruling held to say the same as one before them
    /// ([`Arbitrate::agrees`]): each `[result, the earlier one, how likely]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub same: Vec<(String, String, f64)>,
    /// Where a ruling could not be made and one set aside before it stood in its place
    /// ([`Arbiter::revived`]): `[the round it was made in, the round it stood for]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused: Option<(usize, usize)>,
}

/// The branches of a request, shared with its tools as they run ([`Call::siblings`]).
pub(crate) type Branches = Arc<Mutex<Vec<Branch>>>;

/// What a branch is stopped with, and what everything under it reads: a stop is the branch's,
/// and every branch below it is stopped with it. A branch let finish unwaited is shadowed, and
/// so is everything under it.
#[derive(Debug, Default)]
pub(crate) struct Stop {
    stopped: AtomicBool,
    shadowed: AtomicBool,
    /// How many things outside the branch is in the middle of doing ([`Call::acting`]). A
    /// stop and the start of such a thing take it in turn, so neither comes between the other's
    /// look and its change.
    acting: Mutex<usize>,
    above: Option<Arc<Stop>>,
}

impl Stop {
    /// A stop of its own, under the one above, if there is one.
    pub(crate) fn under(above: Option<Arc<Stop>>) -> Arc<Stop> {
        Arc::new(Stop { above, ..Stop::default() })
    }

    pub(crate) fn set(&self) {
        self.stopped.store(true, Ordering::Relaxed);
    }

    pub(crate) fn is_set(&self) -> bool {
        self.stopped.load(Ordering::Relaxed) || self.above.as_ref().is_some_and(|above| above.is_set())
    }

    pub(crate) fn shadow(&self) {
        self.shadowed.store(true, Ordering::Relaxed);
    }

    /// Whether it was let finish unwaited, it or a branch above it.
    pub(crate) fn is_shadowed(&self) -> bool {
        self.shadowed.load(Ordering::Relaxed) || self.above.as_ref().is_some_and(|above| above.is_shadowed())
    }

    /// This stop and every one above it, outermost first.
    fn chain(&self) -> Vec<&Stop> {
        let mut chain = vec![self];
        while let Some(above) = chain.last().and_then(|stop| stop.above.as_deref()) {
            chain.push(above);
        }
        chain.reverse();
        chain
    }

    /// Begin something outside, unless the branch, or one above it, is stopped: false then, and
    /// nothing begun. It is counted on every branch above as well, so that none of them is
    /// stopped while it is done. Their locks are taken outermost first, which is the order
    /// every other taker of more than one keeps too.
    pub(crate) fn begin_acting(&self) -> bool {
        let chain = self.chain();
        let mut held: Vec<_> = chain
            .iter()
            .map(|stop| stop.acting.lock().unwrap_or_else(PoisonError::into_inner))
            .collect();
        // Nothing is begun outside for a branch stopped, nor for one let finish unwaited.
        if self.is_set() || self.is_shadowed() {
            return false;
        }
        held.iter_mut().for_each(|acting| **acting += 1);
        true
    }

    pub(crate) fn end_acting(&self) {
        for stop in self.chain() {
            let mut acting = stop.acting.lock().unwrap_or_else(PoisonError::into_inner);
            *acting = acting.saturating_sub(1);
        }
    }

    /// Let the branch finish unwaited, unless it is in the middle of something outside: false
    /// then, and it is waited for, so that what it does is said.
    pub(crate) fn shadow_unless_acting(&self) -> bool {
        let acting = self.acting.lock().unwrap_or_else(PoisonError::into_inner);
        if *acting > 0 {
            return false;
        }
        self.shadow();
        true
    }

    /// Stop the branch unless it is in the middle of something outside: false then, and it is
    /// not stopped.
    pub(crate) fn stop_unless_acting(&self) -> bool {
        let acting = self.acting.lock().unwrap_or_else(PoisonError::into_inner);
        if *acting > 0 {
            return false;
        }
        self.set();
        true
    }

    /// Whether it was itself let finish unwaited.
    pub(crate) fn shadowed_here(&self) -> bool {
        self.shadowed.load(Ordering::Relaxed)
    }
}

pub(crate) fn lock(branches: &Branches) -> std::sync::MutexGuard<'_, Vec<Branch>> {
    branches.lock().unwrap_or_else(PoisonError::into_inner)
}

type Shows = dyn Fn(&Branch) -> Json + Send + Sync;

/// What `next` is answered with when what came back answers it all.
pub const FINISHED: &str = "finish";
/// What `next` is answered with when something running is to be waited for.
pub const WAIT: &str = "wait";
/// What `next` is answered with when nothing more could answer it.
pub const UNKNOWN: &str = "unknown";

/// The arbiter that asks the decision model.
///
/// Each ruling is one request: what the request being settled showed, with what it is for and
/// what it asked in the ruling's own instructions, and the branches, each with its name, what
/// it does, whether the answers selected it, where it is and what it came back with or has so
/// far. Its questions:
///
/// - `use_{name}`, of each branch with something to show: does it answer the request, or a
///   part of it, from what it found? A yes or no.
/// - `wait_{name}`, of each branch running: could it still answer a part not answered? A yes
///   or no.
/// - `conflict`, where two or more have something to show: do two of the results that answer
///   the request say different things about the same matter? A yes or no. It cannot see which
///   are used, being asked beside the uses: where it is a yes and two or more are used, a
///   second request, shown only those used, asks `contradict`: do they say different things
///   about the same matter? Its answer is the ruling's `conflict`.
/// - `next`, what is to happen now: a choice of one, `finish` (what came back answers it all),
///   `wait` (something running is to be waited for), `start_{name}` for each branch not
///   started, or `unknown` (nothing more could answer it). It is the ruling: the questions
///   above are asked beside it, about each branch, and say what is shown and what goes on
///   running, but whether the request goes on is this one answer. A finish chosen where
///   nothing is used, or where results used are in conflict, is not what the ruling's own answers say:
///   the likeliest of the other options is taken in its place; so for a wait chosen where no
///   branch running is held to be worth waiting for.
///
/// With [`Arbitrate::shadows`], `shadow_{name}` is asked of each branch running as well: let it
/// finish after the request is over, unwaited, so that what it comes to can be checked against
/// what was used? Before any branch starts, where it is asked then ([`Arbitrate::first`]),
/// only `start_{name}` is asked, of each branch the answers selected, in its own words. Every
/// question's words can be the host's: `{name}` in them is the branch's name and `{about}`
/// what it does. The branches are shown as an input named `branches`, or, where the request
/// has an input of that name, `branches_` and so on.
///
/// ```
/// use rdm::arbitrate::Arbitrate;
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let ask = Signature::new("Answer a question.").input("question").noul("guess", "Guess?").noul("look", "Look it up?");
/// let guess = Tool::new("guess", "Guess from what is known.", |_| Ok("maybe Lyon"));
/// let look = Tool::new("look", "Look it up.", |_| Ok("Paris, by the atlas"));
/// // The decision model holds the guess to answer nothing, and the looking up to answer it.
/// let model = Scripted::new(|request| {
///     let asked = |name: &str| request.questions.contains_key(name);
///     if !asked("next") {
///         return vec![("guess", Answer::noul(0.9)), ("look", Answer::noul(0.9))];
///     }
///     let options = request.questions["next"].labels();
///     let next = if asked("use_look") { "finish" } else { "wait" };
///     let mut answers = vec![("next", Answer::chose(next, options.iter().map(String::as_str)))];
///     for (name, p) in [("use_guess", 0.2), ("use_look", 0.95), ("wait_look", 0.9), ("wait_guess", 0.1), ("conflict", 0.1)] {
///         if asked(name) {
///             answers.push((name, Answer::noul(p)));
///         }
///     }
///     answers
/// });
/// let rdm = Rdm::new(ask).tool(guess).tool(look).arbiter(Arbitrate::new()).model(model);
/// let result = rdm.call(inputs! { "question" => "What is the capital of France?" }).unwrap();
/// // Whichever came back first, the request comes to the one that answers it.
/// let arbitration = result.arbitration.unwrap();
/// assert_eq!(arbitration.used, ["look"]);
/// assert!(arbitration.answered);
/// assert_eq!(result.observations["look"], "Paris, by the atlas");
/// ```
#[derive(Clone)]
pub struct Arbitrate {
    instructions: String,
    branches: String,
    uses: String,
    waits: String,
    starts: String,
    starts_first: Option<String>,
    shadows: Option<String>,
    agrees: Option<String>,
    conflict: String,
    next: String,
    finished: String,
    wait: String,
    unknown: String,
    beside: Vec<String>,
    shows: Option<Arc<Shows>>,
    contradicts: String,
    /// The module its rulings are put to, where they have a decision model of their own.
    on: Option<Rdm>,
}

impl std::fmt::Debug for Arbitrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arbitrate")
            .field("beside", &self.beside)
            .field("shows", &self.shows.is_some())
            .field("on", &self.on.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for Arbitrate {
    fn default() -> Self {
        Self::new()
    }
}

impl Arbitrate {
    /// The arbiter, in its own words, not asked before branches start and letting none finish
    /// unwaited.
    pub fn new() -> Self {
        Arbitrate {
            instructions: "Settle what a request has come to so far: which of the results that have come back, or of \
                           what a branch still running has so far, answer it; which branches still running could still \
                           answer a part of it; and what is to happen now."
                .to_owned(),
            branches: "What the request could set going, each with its `name`, what it does (`about`), whether \
                       the request's answers chose it (`selected`), where it is (`state`: not started, running, \
                       returned or stopped), and what it came back with (`result`) or, still running or stopped, has \
                       so far (`so_far`)."
                .to_owned(),
            uses: "Does what `{name}` came back with, or has so far, answer what the request asks, or a part of it, \
                   from what it found? It is a no where its own attempt failed: it could not do what it was set, found \
                   nothing, or was cut short before the part asked. So is a result about something else, and a guess \
                   what it found does not bear out. What it reports is another matter: a result that reports what was \
                   asked, faithfully, is a yes even where what it reports is a failure, an error or bad news."
                .to_owned(),
            waits: "Could what `{name}` is still working on answer a part of what the request asks that no result \
                    answers yet?"
                .to_owned(),
            starts: "Start `{name}` now ({about}): a part of what the request asks is not answered by what came back, \
                     and it could answer that part."
                .to_owned(),
            starts_first: None,
            shadows: None,
            agrees: None,
            conflict: "Of the results that answer the request, do two say different things about the same matter, so \
                       that both cannot be right? A result that does not answer it, or a guess another result shows \
                       wrong, is not one of them."
                .to_owned(),
            next: "What is to happen now, given what came back and what is still running?".to_owned(),
            finished: "Finish: what came back answers everything the request asks, and agrees; nothing still running \
                       or not started is needed."
                .to_owned(),
            wait: "Wait: a part of what the request asks is not answered yet, and something still running could \
                   answer it."
                .to_owned(),
            unknown: "Say what is not known: a part of what the request asks is not answered, or what came back \
                      disagrees, and nothing running or not started could settle it."
                .to_owned(),
            beside: Vec::new(),
            shows: None,
            contradicts: "Do two of these results say different things about the same matter, so that both cannot be \
                          right? Results about different parts of what was asked do not contradict each other."
                .to_owned(),
            on: None,
        }
    }

    /// What the request it asks is told it is for.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// What it is told the branches are.
    pub fn branches(mut self, description: impl Into<String>) -> Self {
        self.branches = description.into();
        self
    }

    /// The words of `use_{name}`.
    pub fn uses(mut self, words: impl Into<String>) -> Self {
        self.uses = words.into();
        self
    }

    /// The words of `wait_{name}`.
    pub fn waits(mut self, words: impl Into<String>) -> Self {
        self.waits = words.into();
        self
    }

    /// What `start_{name}` means as an option of `next`, once branches have come back.
    pub fn starts(mut self, words: impl Into<String>) -> Self {
        self.starts = words.into();
        self
    }

    /// Ask before any branch starts, about which of those the answers selected start now, in
    /// these words: a yes or no of each.
    pub fn first(mut self, words: impl Into<String>) -> Self {
        self.starts_first = Some(words.into());
        self
    }

    /// Ask of each branch running whether to let it finish unwaited, in these words: what it
    /// comes to is recorded, and nothing it does acts outside ([`Call::shadowed`]).
    pub fn shadows(mut self, words: impl Into<String>) -> Self {
        self.shadows = Some(words.into());
        self
    }

    /// Ask of each result shown after another which one before it says the same thing, if
    /// any, in these words: a choice among them, or none. Its answers are the ruling's `same`,
    /// for a host that counts two results as one outcome.
    pub fn agrees(mut self, words: impl Into<String>) -> Self {
        self.agrees = Some(words.into());
        self
    }

    /// The words of `contradict`: asked, where results disagree and more than one is used, of
    /// those used alone.
    pub fn contradicts(mut self, words: impl Into<String>) -> Self {
        self.contradicts = words.into();
        self
    }

    /// Whether the results used contradict each other: one request, shown only them.
    fn contradict(&self, ledger: &Ledger, call: &Call, used: &[String]) -> Result<bool> {
        let results = Self::named(ledger, "results");
        let signature = ledger
            .signature
            .inputs()
            .iter()
            .fold(
                Signature::new(format!(
                    "Check whether the results a request comes to agree.\n\n{}",
                    Self::request(ledger)
                )),
                |signature, (name, field)| signature.input_with(name, &field.type_name, &field.description),
            )
            .input_with(
                &results,
                "list[dict]",
                "The results the request comes to, each with its `name` and `result`.",
            )
            .noul("contradict", &self.contradicts);
        let mut inputs = ledger.inputs.clone();
        let shown: Vec<Json> = ledger
            .showing()
            .filter(|branch| used.contains(&branch.output))
            .map(|branch| self.shown(branch))
            .collect();
        inputs.set(results, shown);
        let ask = Ask::new(inputs).signature(signature).answers_only();
        let checked = match &self.on {
            Some(module) => call.on(module).sub_with(ask)?,
            None => call.sub_with(ask)?,
        };
        if !checked.answered() {
            return Err(Error::Tool(format!(
                "whether the results used agree was not asked: {}",
                checked.termination_reason.as_str()
            )));
        }
        Ok(checked.yes("contradict"))
    }

    /// The words of `conflict`.
    pub fn conflict(mut self, words: impl Into<String>) -> Self {
        self.conflict = words.into();
        self
    }

    /// The question `next` is, and what its options `finish`, `wait` and `unknown` mean.
    pub fn next(
        mut self,
        question: impl Into<String>,
        finished: impl Into<String>,
        wait: impl Into<String>,
        unknown: impl Into<String>,
    ) -> Self {
        (self.next, self.finished, self.wait, self.unknown) = (question.into(), finished.into(), wait.into(), unknown.into());
        self
    }

    /// Put its rulings to a decision model of their own, and not to the module's: a small one
    /// made to settle, say, beside a larger one that routes. Each ruling is still a request of
    /// the run, one level below the request it settles, recorded and replayed as any is.
    pub fn model(mut self, model: impl DecisionModel + 'static) -> Self {
        self.on = Some(Rdm::new(Signature::new("Settle a request.")).model(model));
        self
    }

    /// Tools it leaves alone ([`Arbiter::beside`]).
    pub fn beside<N: Into<String>>(mut self, names: impl IntoIterator<Item = N>) -> Self {
        self.beside.extend(names.into_iter().map(Into::into));
        self
    }

    /// How a branch's result, or what it has so far, is shown: as the next request after a
    /// tool is shown what it returned, unless this says otherwise.
    pub fn shows(mut self, shows: impl Fn(&Branch) -> Json + Send + Sync + 'static) -> Self {
        self.shows = Some(Arc::new(shows));
        self
    }

    fn worded(words: &str, branch: &Branch) -> String {
        words.replace("{name}", &branch.output).replace("{about}", &branch.about)
    }

    fn shown(&self, branch: &Branch) -> Json {
        let mut shown = json!({"name": branch.output, "about": branch.about, "selected": branch.selected,
                               "state": branch.state.as_str()});
        if let Some(value) = branch.shown() {
            let key = if matches!(branch.state, State::Running | State::Stopped) {
                "so_far"
            } else {
                "result"
            };
            shown[key] = match &self.shows {
                Some(shows) => shows(branch),
                None => Json::String(as_observation(value)),
            };
        }
        shown
    }

    /// The name an input of the ruling's own is shown under: as it is, unless the request has
    /// an input of that name.
    fn named(ledger: &Ledger, name: &str) -> String {
        let mut name = name.to_owned();
        while ledger.signature.inputs().contains_key(&name) {
            name.push('_');
        }
        name
    }

    /// What the request is for and what it asked, each question with what its answers mean and
    /// what it was answered: a ruling is about this request and no other.
    fn request(ledger: &Ledger) -> String {
        let asked: Vec<String> = ledger
            .signature
            .outputs()
            .iter()
            .map(|(name, output)| {
                let answered = ledger.answers.get(name).map_or("not asked".to_owned(), Json::to_string);
                match output.criteria() {
                    Some(criteria) => format!("- `{name}`: {} (its answers: {criteria}). Answered: {answered}", output.description),
                    None => format!("- `{name}`: {} Answered: {answered}", output.description),
                }
            })
            .collect();
        format!(
            "The request being settled is this. {}\nIts questions:\n{}",
            ledger.signature.instructions(),
            asked.join("\n")
        )
    }

    /// The request a ruling asks, about the ledger as it is.
    pub fn signature(&self, ledger: &Ledger) -> Result<Signature> {
        let instructions = format!("{}\n\n{}", self.instructions, Self::request(ledger));
        let signature = ledger
            .signature
            .inputs()
            .iter()
            .fold(Signature::new(instructions), |signature, (name, field)| {
                signature.input_with(name, &field.type_name, &field.description)
            });
        let mut signature = signature
            .input_with(Self::named(ledger, "branches"), "list[dict]", &self.branches)
            .input_with(
                Self::named(ledger, "before"),
                "list[dict]",
                "What was ruled before on this request, oldest first: each time, what was to happen next (`next`), \
                 which results were used, which branches were waited for, started or stopped. What came of it is in \
                 the branches as they are now.",
            );
        if ledger.first {
            for branch in ledger.in_state(State::Idle).filter(|branch| branch.selected) {
                let words = self.starts_first.as_deref().unwrap_or(&self.starts);
                signature = signature.noul(format!("start_{}", branch.output), Self::worded(words, branch));
            }
            return Ok(signature);
        }
        for branch in ledger.showing() {
            signature = signature.noul(format!("use_{}", branch.output), Self::worded(&self.uses, branch));
        }
        for branch in ledger.in_state(State::Running) {
            signature = signature.noul(format!("wait_{}", branch.output), Self::worded(&self.waits, branch));
            if let Some(words) = &self.shadows {
                signature = signature.noul(format!("shadow_{}", branch.output), Self::worded(words, branch));
            }
        }
        if ledger.showing().count() > 1 {
            signature = signature.noul("conflict", &self.conflict);
        }
        if let Some(words) = &self.agrees {
            let showing: Vec<&Branch> = ledger.showing().collect();
            for (at, branch) in showing.iter().enumerate().skip(1) {
                let earlier = showing[..at].iter().map(|earlier| (earlier.output.clone(), earlier.about.clone()));
                signature = signature.choice_or_none(
                    format!("same_{}", branch.output),
                    Self::worded(words, branch),
                    earlier,
                    "None of them: it says something else.",
                );
            }
        }
        let mut options = vec![(FINISHED.to_owned(), self.finished.clone())];
        if ledger.in_state(State::Running).next().is_some() {
            options.push((WAIT.to_owned(), self.wait.clone()));
        }
        for branch in ledger.in_state(State::Idle) {
            options.push((format!("start_{}", branch.output), Self::worded(&self.starts, branch)));
        }
        options.push((UNKNOWN.to_owned(), self.unknown.clone()));
        Ok(signature.choice("next", &self.next, options))
    }
}

impl Arbiter for Arbitrate {
    fn arbitrate(&self, ledger: &Ledger, call: &Call) -> Result<Ruling> {
        let signature = self.signature(ledger)?;
        // The inputs as they are now: a live one is read again for the ruling.
        let mut inputs = ledger.inputs.clone();
        let shown: Vec<Json> = ledger.branches.iter().map(|branch| self.shown(branch)).collect();
        inputs.set(Self::named(ledger, "branches"), shown);
        let before: Vec<Json> = ledger.before.iter().map(Ruling::shown).collect();
        inputs.set(Self::named(ledger, "before"), before);
        let ask = Ask::new(inputs).signature(signature).answers_only();
        let ruled = match &self.on {
            Some(module) => call.on(module).sub_with(ask)?,
            None => call.sub_with(ask)?,
        };
        if !ruled.answered() {
            return Err(Error::Tool(format!(
                "the ruling was not asked: {}",
                ruled.termination_reason.as_str()
            )));
        }
        let said = |prefix: &str, branches: &mut dyn Iterator<Item = &Branch>| -> Vec<String> {
            branches
                .filter(|branch| ruled.yes(&format!("{prefix}{}", branch.output)))
                .map(|branch| branch.output.clone())
                .collect()
        };
        if ledger.first {
            return Ok(Ruling {
                start: said("start_", &mut ledger.in_state(State::Idle)),
                ..Ruling::default()
            });
        }
        let used = said("use_", &mut ledger.showing());
        let waited = said("wait_", &mut ledger.in_state(State::Running));
        let shadow = said("shadow_", &mut ledger.in_state(State::Running));
        // Disagreement among all that came back is not yet a conflict in what the request comes
        // to: the question beside the uses cannot see which are used. Where results disagree and
        // more than one is used, a second request, shown only those used, asks whether they
        // contradict each other: that is the conflict.
        let conflict = match ruled.yes("conflict") && used.len() > 1 {
            false => false,
            true => self.contradict(ledger, call, &used)?,
        };
        let same = ledger
            .showing()
            .filter_map(|branch| {
                let name = format!("same_{}", branch.output);
                let earlier = ruled.chosen(&name)?;
                let sure = ruled
                    .choice(&name)
                    .and_then(|choice| choice.probabilities.get(earlier).copied())
                    .unwrap_or(0.0);
                Some((branch.output.clone(), earlier.to_owned(), sure))
            })
            .collect();
        // Finishing means what came back answers it and agrees. Chosen with nothing used, or with
        // results held to disagree, it is not what the ruling's own answers say, and the likeliest
        // of the other options is taken: waiting, starting a branch, or saying what is not known.
        let chosen = ruled.chosen("next").unwrap_or(UNKNOWN);
        // Waiting, likewise, is for something still running that could answer: chosen where no
        // branch running is held to be worth it, it gives way too.
        let coherent = |option: &str| match option {
            FINISHED => !used.is_empty() && !conflict,
            WAIT => !waited.is_empty(),
            _ => true,
        };
        let next = match coherent(chosen) {
            true => chosen.to_owned(),
            false => ruled
                .choice("next")
                .and_then(|choice| {
                    let others = choice.probabilities.iter().filter(|(option, _)| coherent(option));
                    others.max_by(|a, b| a.1.total_cmp(b.1)).map(|(option, _)| option.clone())
                })
                .unwrap_or_else(|| UNKNOWN.to_owned()),
        };
        let next = next.as_str();
        let ruling = match next {
            FINISHED => Ruling {
                done: !used.is_empty(),
                unresolved: used.is_empty(),
                ..Ruling::default()
            },
            WAIT => Ruling {
                wait: waited,
                ..Ruling::default()
            },
            UNKNOWN => Ruling {
                unresolved: true,
                ..Ruling::default()
            },
            start => Ruling {
                start: start.strip_prefix("start_").map(str::to_owned).into_iter().collect(),
                wait: waited,
                ..Ruling::default()
            },
        };
        let shadow = shadow.into_iter().filter(|output| !ruling.wait.contains(output)).collect();
        Ok(Ruling {
            used,
            shadow,
            conflict,
            same,
            ..ruling
        })
    }

    fn first(&self) -> bool {
        self.starts_first.is_some()
    }

    fn beside(&self) -> Vec<String> {
        self.beside.clone()
    }
}
