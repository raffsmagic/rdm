//! What the tests share: a stand-in decision model that answers as a function says, and a few readers.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rdm::systemone::{Answer, Kind, Question, Request, Scripted};
use rdm::{Entry, Json, Prediction};

/// Whether a thread can be started here: anywhere but in a WebAssembly module built without threads.
pub const THREADS: bool = cfg!(not(all(target_family = "wasm", not(target_feature = "atomics"))));

/// Run on a thread with a stack of a set size, as a caller that is not the main thread would.
/// Where no thread can be started, on the one stack there is.
pub fn on_a_stack<T: Send + 'static>(size: usize, run: impl FnOnce() -> T + Send + 'static) -> T {
    if !THREADS {
        return run();
    }
    let thread = std::thread::Builder::new().stack_size(size).spawn(run).unwrap();
    thread.join().expect("the run panicked")
}

/// What a stand-in model says to one question.
pub enum Say {
    /// A yes or a no, as likely as a model that is sure gives it: 0.9 or 0.1.
    Yes(bool),
    /// How likely a yes is.
    P(f64),
    /// The option chosen, as certain.
    Pick(String),
    /// The level scored, as certain.
    Level(usize),
}

impl From<bool> for Say {
    fn from(yes: bool) -> Self {
        Say::Yes(yes)
    }
}

impl From<f64> for Say {
    fn from(probability: f64) -> Self {
        Say::P(probability)
    }
}

impl From<&str> for Say {
    fn from(option: &str) -> Self {
        Say::Pick(option.to_owned())
    }
}

impl From<String> for Say {
    fn from(option: String) -> Self {
        Say::Pick(option)
    }
}

impl From<usize> for Say {
    fn from(level: usize) -> Self {
        Say::Level(level)
    }
}

/// An answer as the probabilities a System One model returns for that kind of question.
fn evidence(question: &Question, said: Say) -> Answer {
    match (question.kind, said) {
        (Kind::Noul, Say::Yes(yes)) => Answer::noul(if yes { 0.9 } else { 0.1 }),
        (Kind::Noul, Say::P(probability)) => Answer::noul(probability),
        (Kind::Choice, Say::Pick(option)) => {
            let labels = question.labels();
            Answer::chose(&option, labels.iter().map(String::as_str))
        }
        (Kind::Score, Say::Level(level)) => Answer::score((0..question.labels().len()).map(|at| if at == level { 1.0 } else { 0.0 })),
        (kind, _) => panic!(
            "the stand-in model was given an answer of another kind for a {} question",
            kind.as_str()
        ),
    }
}

/// A stand-in System One model: `answer(inputs, name, question)` gives each question's answer.
pub fn client<S: Into<Say>>(answer: impl Fn(&Json, &str, &Question) -> S + Send + Sync + 'static) -> Arc<Scripted> {
    client_using(0, answer)
}

/// As [`client`], reporting tokens for every request.
pub fn client_using<S: Into<Say>>(tokens: u64, answer: impl Fn(&Json, &str, &Question) -> S + Send + Sync + 'static) -> Arc<Scripted> {
    let answers = move |request: &Request| {
        let inputs = &request.state["inputs"];
        let each = request
            .questions
            .iter()
            .map(|(name, question)| (name.clone(), evidence(question, answer(inputs, name, question).into())));
        each.collect::<Vec<_>>()
    };
    Arc::new(Scripted::new(answers).named("jev-test").tokens(tokens))
}

/// A decision model that points at these names, in order.
pub fn scripted<const N: usize>(names: [&str; N]) -> Arc<Scripted> {
    let script = Mutex::new(names.iter().map(|name| name.to_string()).collect::<VecDeque<_>>());
    client(move |_, _, _| script.lock().unwrap().pop_front().expect("the script has run out"))
}

/// The inputs each request showed, as the model received them.
pub fn shown(model: &Scripted) -> Vec<Json> {
    model.requests().iter().map(|request| request.state["inputs"].clone()).collect()
}

/// The names of the questions each request asked.
pub fn asked(model: &Scripted) -> Vec<Vec<String>> {
    model
        .requests()
        .iter()
        .map(|request| request.questions.keys().cloned().collect())
        .collect()
}

/// The options a request offered for a choice.
pub fn offered(request: &Request, name: &str) -> Vec<String> {
    request.questions[name].labels()
}

/// The entries of a run at one depth.
pub fn at_depth(result: &Prediction, depth: usize) -> Vec<&Entry> {
    result.trajectory.iter().filter(|entry| entry.depth == depth).collect()
}

/// The entries a limit stopped.
pub fn skipped(result: &Prediction, reason: &str) -> Vec<Entry> {
    result
        .trajectory
        .iter()
        .filter(|entry| entry.skipped.is_some_and(|skipped| skipped == reason))
        .cloned()
        .collect()
}

/// A meeting point: `meet` returns true once as many have arrived as it was made for, and false
/// if they have not within ten seconds. A test that would hang on a barrier fails on this.
pub struct Meeting {
    wanted: usize,
    arrived: Mutex<usize>,
    all: Condvar,
}

impl Meeting {
    pub fn of(wanted: usize) -> Arc<Self> {
        Arc::new(Meeting {
            wanted,
            arrived: Mutex::new(0),
            all: Condvar::new(),
        })
    }

    pub fn meet(&self) -> bool {
        let mut arrived = self.arrived.lock().unwrap();
        *arrived += 1;
        self.all.notify_all();
        let (arrived, waited) = self
            .all
            .wait_timeout_while(arrived, Duration::from_secs(10), |arrived| *arrived < self.wanted)
            .unwrap();
        drop(arrived);
        !waited.timed_out()
    }
}
