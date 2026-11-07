//! Near: gives the names on a map labels, and offers the names nearest a request.
//!
//! The names on a module's map have one dimension, a description, and the decision model
//! reads every one each time it points. [`Near`] gives each name labels as well: its answers
//! to the questions of a signature, each question a dimension with a name.
//!
//! ```text
//! let about = Signature::new("Say what a text is about.")
//!     .input("task")
//!     .noul("heating", "Is `inputs.task` about how warm it is?")
//!     .noul("lights", "Is `inputs.task` about a light?");
//!
//! let rdm = Rdm::new(operate).tools_from(steps).fit(Near::new(about).top(8));
//!
//! "cool by degrees: Make it cooler by a number of degrees."    {"heating": 0.97, "lights": 0.0}
//! "kitchen lights off: Turn the kitchen lights off."            {"heating": 0.0, "lights": 0.99}
//! "It is too warm in here."                                     {"heating": 0.91, "lights": 0.0}
//! ```
//!
//! A name is asked once, when it is first on the map, and what it answered is kept:
//! [`Near::answers`] holds what the model gave for each text and dimension. It is part of the
//! program, so it is plain data that is saved and loaded ([`Labels`], [`Saved`]), and a copy
//! has its own. Each request is asked the same questions, and the pointer is offered the
//! `top` names nearest it. Every number has a name, so it can be read why a name was offered.
//!
//! A dimension is a yes or no question, a score, or a choice, whose options are a dimension
//! each. A dimension is also a name and a question that can be written while the module runs:
//! [`Near::add`] and [`Near::remove`] take plain text, so a language model writes dimensions
//! as it writes anything else. The language model coins the label; the decision model says
//! of each thing whether it applies.
//!
//! `Near` has no threshold or weight of its own. A label is what the model answered, until
//! its dimension is given a parameter ([`Near::set_params`]): a threshold, under which a yes
//! or no counts as nothing; cuts, which give a score its level; weights, which multiply a
//! choice's options. Those are parameters of the program, saved with it, and they are read
//! each time a label is, so changing one asks nothing again.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::panic::resume_unwind;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering as Counting};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use systemone::{Decision, DecisionModel, Json, Params};

use crate::call::{Ask, Call};
use crate::error::{Error, Result, listed, sorted};
use crate::inputs::Inputs;
use crate::module::Fit;
use crate::observation::as_observation;
use crate::signature::{NONE, Opt, Output, RESERVED, Signature, Space};

/// How a request uses a label, where a `Near` asks: what the request is asked of each dimension.
const USES: [(&str, &str); 3] = [
    ("filter", "Only what has this label will do."),
    ("boost", "What has this label is better."),
    ("ignore", "This label does not matter here."),
];

/// What the request that asks how each dimension is used is for.
const USE_INSTRUCTIONS: &str = "Say how a request uses each label of the things it could be given.";

/// The measures there are by name, in alphabetical order.
pub const MEASURES: [&str; 5] = ["cosine", "dot", "euclidean", "jaccard", "manhattan"];

/// A text's coordinates: a number for every yes or no dimension and every score, by the
/// dimension's name, and one for each option of a choice, as `name.option`.
pub type Coordinates = IndexMap<String, f64>;

/// Coordinates written by hand: the vector of what is wanted, which can be read.
///
/// ```
/// use rdm::near::{Measure, coordinates};
///
/// let wanted = coordinates([("money", 1.0), ("buying", 0.5)]);
/// let refund = coordinates([("money", 0.9), ("buying", 0.0)]);
/// assert_eq!((Measure::Dot.score(&wanted, &refund) * 100.0).round(), 90.0);
/// ```
pub fn coordinates<N: Into<String>>(named: impl IntoIterator<Item = (N, f64)>) -> Coordinates {
    named.into_iter().map(|(name, value)| (name.into(), value)).collect()
}

/// A name on the map as it is asked about: the name with its description, or the name alone
/// where it has none. Labelling a long map before a run asks about these.
///
/// ```
/// assert_eq!(rdm::near::text("cooler", "Make it one degree cooler."), "cooler: Make it one degree cooler.");
/// assert_eq!(rdm::near::text("cooler", ""), "cooler");
/// ```
pub fn text(name: &str, description: &str) -> String {
    if description.is_empty() {
        name.to_owned()
    } else {
        format!("{name}: {description}")
    }
}

/// An answer as it is kept: the probability of a yes, or the probabilities of a score's
/// levels or a choice's options. It is what the model gave, before any parameter is read.
///
/// As JSON it is a number, or `{"probabilities": {...}, "confidence": ...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Given {
    /// How likely a yes is.
    Noul(f64),
    /// How likely each option of a choice is, or each level of a score, by its index.
    Probabilities {
        /// By option, or by level index as a string: `"0"`, `"1"`, ...
        probabilities: IndexMap<String, f64>,
        /// How sure the answer was, as the model or the protocol gives it.
        confidence: f64,
    },
}

impl Given {
    /// How likely a yes is, for a yes or no dimension.
    pub fn noul(&self) -> Option<f64> {
        match self {
            Given::Noul(probability) => Some(*probability),
            Given::Probabilities { .. } => None,
        }
    }

    /// How likely each option or level is, for a choice or a score.
    pub fn probabilities(&self) -> Option<&IndexMap<String, f64>> {
        match self {
            Given::Noul(_) => None,
            Given::Probabilities { probabilities, .. } => Some(probabilities),
        }
    }

    fn of(decision: &Decision) -> Self {
        match decision {
            Decision::Noul(noul) => Given::Noul(noul.probability),
            Decision::Choice(choice) => Given::Probabilities {
                probabilities: choice.probabilities.clone(),
                confidence: choice.confidence,
            },
            Decision::Score(score) => Given::Probabilities {
                probabilities: score.probabilities.clone(),
                confidence: score.confidence,
            },
        }
    }
}

/// A yes or no dimension's answer is a number, and reads as one.
impl PartialEq<f64> for Given {
    fn eq(&self, other: &f64) -> bool {
        self.noul() == Some(*other)
    }
}

/// An answer is written as it reads. A yes or no dimension's is its number, so `{value:.2}` is how likely a yes
/// is to two places; a choice's or a score's is each option or level with how likely it is.
///
/// ```
/// use rdm::near::Given;
///
/// assert_eq!(format!("money {:.2}", Given::Noul(0.9)), "money 0.90");
/// let room: Given = serde_json::from_str(r#"{"probabilities": {"hall": 0.25, "kitchen": 0.75}, "confidence": 0.5}"#).unwrap();
/// assert_eq!(format!("{room:.1}"), "hall 0.2, kitchen 0.8");
/// ```
impl fmt::Display for Given {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Given::Noul(probability) => fmt::Display::fmt(probability, f),
            Given::Probabilities { probabilities, .. } => {
                for (position, (label, probability)) in probabilities.iter().enumerate() {
                    f.write_str(if position == 0 { "" } else { ", " })?;
                    write!(f, "{label} ")?;
                    fmt::Display::fmt(probability, f)?;
                }
                Ok(())
            }
        }
    }
}

/// What each text answered, by dimension.
pub type Answers = IndexMap<String, IndexMap<String, Given>>;

/// The labels: every dimension's question; every text's answers, as the model gave them; how
/// each request uses each dimension; and the model that gave them all.
///
/// They are plain data, and part of the program: written as JSON they are what the Python
/// module keeps, key for key, and are read back with [`Near::set_labels`].
///
/// ```
/// use rdm::near::Labels;
/// use serde_json::json;
///
/// let kept = json!({
///     "questions": {"heating": "Is `inputs.task` about how warm it is?"},
///     "answers": {"cooler: Make it one degree cooler.": {"heating": 0.9}},
///     "uses": {},
///     "model": "jev-latest",
/// });
/// let labels: Labels = serde_json::from_value(kept.clone()).unwrap();
/// assert_eq!(labels.answers["cooler: Make it one degree cooler."]["heating"], 0.9);
/// assert_eq!(serde_json::to_value(&labels).unwrap(), kept);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    /// Every dimension's question, by name.
    #[serde(default)]
    pub questions: IndexMap<String, String>,
    /// What each text asked so far answered, by dimension.
    #[serde(default)]
    pub answers: Answers,
    /// How each request said it uses each dimension, by `{dimension}_use`.
    #[serde(default)]
    pub uses: Answers,
    /// The model that gave the answers. They are one model's, and are dropped when another is asked.
    #[serde(default)]
    pub model: Option<String>,
}

/// Everything of a [`Near`] that is the program's: the labels, and the parameters its
/// dimensions were given. What [`Near::saved`] returns and [`Near::load`] takes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    /// The questions, the answers, the uses and the model.
    #[serde(default)]
    pub labels: Labels,
    /// Each dimension's threshold, cuts or weights, by the dimension's name.
    #[serde(default)]
    pub dimensions: IndexMap<String, Params>,
    /// The weights of filter, boost and ignore for a dimension, by the dimension's name.
    #[serde(default)]
    pub uses: IndexMap<String, Params>,
}

/// A measure that is a function of the request's coordinates and a thing's.
type Custom = dyn Fn(&Coordinates, &Coordinates) -> f64 + Send + Sync;

/// How two sets of coordinates are compared: the larger, the nearer.
///
/// Each is over the request's dimensions. A dimension the thing lacks counts as nothing.
///
/// ```
/// use rdm::near::{Measure, coordinates};
///
/// let request = coordinates([("heating", 1.0), ("lights", 0.0)]);
/// let both = coordinates([("heating", 1.0), ("lights", 1.0)]);
/// // What is shared, as much as both have of it: a dimension only the thing has costs it nothing.
/// assert_eq!(Measure::Dot.score(&request, &both), 1.0);
/// // The direction shared: a dimension only the thing has counts against it.
/// assert_eq!((Measure::Cosine.score(&request, &both) * 1000.0).round(), 707.0);
/// assert_eq!("jaccard".parse::<Measure>().unwrap().score(&request, &both), 0.5);
/// // A measure of your own is a function of the request's coordinates and a thing's.
/// assert_eq!(Measure::custom(|_, thing| thing["lights"]).score(&request, &both), 1.0);
/// ```
#[derive(Clone)]
pub enum Measure {
    /// The dot product: what the two share, each dimension as much as both have of it. A
    /// dimension only one of them has costs nothing.
    Dot,
    /// The direction the two share, whatever their lengths. A dimension only one of them has
    /// counts against them.
    Cosine,
    /// One over one plus the straight-line distance.
    Euclidean,
    /// One over one plus the distance summed a dimension at a time: as euclidean, with large
    /// differences counted no more than small ones.
    Manhattan,
    /// What the two share over what either has (Tanimoto): for coordinates that are close to
    /// yes or no, the share of dimensions in common.
    Jaccard,
    /// A function of the request's coordinates and a thing's. See [`Measure::custom`].
    Custom(Arc<Custom>),
}

impl Measure {
    /// A measure of your own: a function of the request's coordinates and a thing's, the larger the nearer.
    pub fn custom(measure: impl Fn(&Coordinates, &Coordinates) -> f64 + Send + Sync + 'static) -> Self {
        Measure::Custom(Arc::new(measure))
    }

    /// The measure's name: one of [`MEASURES`], or `custom`.
    pub fn name(&self) -> &'static str {
        match self {
            Measure::Dot => "dot",
            Measure::Cosine => "cosine",
            Measure::Euclidean => "euclidean",
            Measure::Manhattan => "manhattan",
            Measure::Jaccard => "jaccard",
            Measure::Custom(_) => "custom",
        }
    }

    /// How near a thing is to a request: the larger, the nearer.
    pub fn score(&self, request: &Coordinates, thing: &Coordinates) -> f64 {
        let of = |name: &String| thing.get(name).copied().unwrap_or(0.0);
        match self {
            Measure::Dot => dot(request, thing),
            Measure::Cosine => {
                let lengths = dot(request, request).sqrt() * dot(thing, thing).sqrt();
                if lengths == 0.0 { 0.0 } else { dot(request, thing) / lengths }
            }
            Measure::Euclidean => {
                let squares = total(request.iter().map(|(name, value)| (value - of(name)).powi(2)));
                1.0 / (1.0 + squares.sqrt())
            }
            Measure::Manhattan => {
                let apart = total(request.iter().map(|(name, value)| (value - of(name)).abs()));
                1.0 / (1.0 + apart)
            }
            Measure::Jaccard => {
                let either = total(request.iter().map(|(name, value)| value.max(of(name))));
                let both = total(request.iter().map(|(name, value)| value.min(of(name))));
                if either == 0.0 { 0.0 } else { both / either }
            }
            Measure::Custom(measure) => measure(request, thing),
        }
    }
}

impl fmt::Debug for Measure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A measure by its name, as a configuration gives it. A name that is none of [`MEASURES`] is a usage error.
impl FromStr for Measure {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "dot" => Ok(Measure::Dot),
            "cosine" => Ok(Measure::Cosine),
            "euclidean" => Ok(Measure::Euclidean),
            "manhattan" => Ok(Measure::Manhattan),
            "jaccard" => Ok(Measure::Jaccard),
            other => Err(Error::Usage(format!(
                "measure must be one of {}, or a function of two sets of coordinates; got {}.",
                listed(MEASURES),
                repr(other)
            ))),
        }
    }
}

/// The dot product: what the two share, each dimension as much as both have of it.
fn dot(request: &Coordinates, thing: &Coordinates) -> f64 {
    total(request.iter().map(|(name, value)| value * thing.get(name).copied().unwrap_or(0.0)))
}

/// A sum as the reference adds numbers up: what each addition rounds away is carried along and
/// added at the end (Neumaier's sum, which is Python's from 3.12). Two names that are as near
/// a request as each other then come out so, where adding them up one at a time would put one
/// of them ahead by the last digit, and the wrong one would be offered. Nothing at all sums to 0.
fn total(numbers: impl IntoIterator<Item = f64>) -> f64 {
    let mut numbers = numbers.into_iter();
    let Some(first) = numbers.next() else { return 0.0 };
    let (mut sum, mut lost) = (0.0 + first, 0.0);
    for number in numbers {
        let next = sum + number;
        lost += if sum.abs() >= number.abs() {
            (sum - next) + number
        } else {
            (number - next) + sum
        };
        sum = next;
    }
    if lost != 0.0 && lost.is_finite() { sum + lost } else { sum }
}

/// What a `Near` is set to and keeps. A request reads it as it is when the request is made.
struct State {
    top: usize,
    measure: Measure,
    uses: bool,
    threads: usize,
    /// The model that is asked where no run is: a `Near` used alone.
    model: Option<Arc<dyn DecisionModel>>,
    labels: Labels,
    /// The questions that make the dimensions, as they are asked now.
    dimensions: Signature,
    /// Each dimension's threshold, cuts or weights.
    params: IndexMap<String, Params>,
    /// The weights of filter, boost and ignore, by dimension.
    use_params: IndexMap<String, Params>,
    /// Counts the times the questions changed, or the model: an answer to questions that have changed since it
    /// was asked for, or from a model that is no longer the one asked, is not kept.
    version: u64,
}

struct Inner {
    /// The signature the dimensions were given as: its input, its instructions, and the kind of each output.
    asks: Signature,
    /// The dimensions' one input.
    input: String,
    /// What is wrong with the signature it was built with, if anything is. Every use says so.
    problem: Option<String>,
    state: Mutex<State>,
    /// What this `Near` has in common with every copy made of it, and of them, and with no other.
    family: Arc<()>,
}

/// Labels for the names on a map, and the names nearest a request.
///
/// Given to an [`crate::Rdm`] as a fit, it cuts an output that points at a tool, where the map
/// has more than `top` names, to the `top` nearest the request, in the order they had.
/// `finish` and the names the module always offers stay. A name that shares nothing with the
/// request is not offered, and where no name shares anything, the request is asked as it is.
/// What it asks are sub-calls of the run, so they count against its limits and are in its
/// trajectory.
///
/// The dimensions are a signature with one input, and an output for each: a yes or no, a
/// score or a choice. The input is given a request's input of the same name, and each name on
/// the map as `"name: description"`, so name it after the request's input that says what is
/// wanted. It may have no outputs yet, where every dimension is to be written with
/// [`Near::add`].
///
/// It is a handle, cheap to clone: a clone is the same `Near`, so one can be given to a module
/// and another kept to read the labels by. [`Near::copy`] is one of its own.
///
/// Used alone, [`Near::embed`] gives each text's coordinates and [`Near::score`] compares two
/// of them by the measure: a search by named dimensions, with no map. It then asks a decision
/// model of its own, given with [`Near::model`].
///
/// ```
/// use std::sync::Arc;
///
/// use rdm::near::Near;
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Kind, Scripted};
///
/// // A stand-in decision model: a label applies where the text has its word, and the pointer
/// // takes the first step offered, then `finish`.
/// let model = Arc::new(Scripted::new(|request| {
///     let inputs = &request.state["inputs"];
///     let text = inputs["task"].as_str().unwrap_or_default().to_lowercase();
///     let answer = |(name, question): (&String, &systemone::Question)| {
///         let labels = question.labels();
///         let answer = match question.kind {
///             Kind::Noul => Answer::noul(if text.contains(name.as_str()) { 0.9 } else { 0.0 }),
///             _ if inputs["observation"] == "" => Answer::chose(&labels[0], labels.iter().map(String::as_str)),
///             _ => Answer::chose("finish", labels.iter().map(String::as_str)),
///         };
///         (name.clone(), answer)
///     };
///     request.questions.iter().map(answer).collect::<Vec<_>>()
/// }));
///
/// let operate = Signature::new("Operate a house, one step at a time.")
///     .input("task")
///     .input("observation")
///     .tool("next", "What should happen next for `inputs.task`?");
/// let step = |name: &'static str, does: &'static str| Tool::new(name, does, move |_| Ok(format!("did {name}")));
/// let steps = [
///     step("hall lamp", "Turn the lamp in the hall on."),
///     step("porch lamp", "Turn the lamp on the porch on."),
///     step("cooler", "Turn the heat down."),
///     step("warmer", "Turn the heat up."),
/// ];
///
/// // The dimensions: each output is one, a name and a question.
/// let about = Signature::new("Say what a text is about.")
///     .input("task")
///     .noul("heat", "Is `inputs.task` about the heat?")
///     .noul("lamp", "Is `inputs.task` about a lamp?");
///
/// let near = Near::new(about).top(2); // of the four steps, the two nearest a request are offered, with `finish`
/// let rdm = Rdm::new(operate).tools_from(steps).fit(near.clone()).model(model.clone());
/// let result = rdm.call(inputs! { "task" => "There is too much heat in here." }).unwrap();
///
/// assert_eq!(result.pointed("next"), ["cooler", "finish"]);
/// let pointing = model.requests().into_iter().find(|request| request.questions.contains_key("next")).unwrap();
/// assert_eq!(pointing.questions["next"].labels(), ["cooler", "warmer", "finish"]);
/// // Every number has a name, so it can be read why a name was offered.
/// assert_eq!(near.answers()["cooler: Turn the heat down."]["heat"], 0.9);
/// assert_eq!(near.answers().len(), 5); // the request and the four names, each asked once
/// ```
#[derive(Clone)]
pub struct Near {
    inner: Arc<Inner>,
}

impl Near {
    /// Labels by the questions of a signature: one input, and an output for each dimension.
    ///
    /// The defaults: `top` 8, the dot product, and no question about how a request uses a
    /// dimension. A signature that is not one input and a yes or no, a score or a choice for
    /// each output is returned by [`Near::check`] and by everything that asks.
    pub fn new(signature: Signature) -> Self {
        let other: Vec<&String> = signature
            .outputs()
            .iter()
            .filter(|(_, output)| output.is_pointer())
            .map(|(name, _)| name)
            .collect();
        let problem = (signature.inputs().len() != 1 || !other.is_empty()).then(|| {
            let which = if other.is_empty() {
                ".".to_owned()
            } else {
                format!("; {} are not.", listed(&other))
            };
            format!("The dimensions are a signature with one input and a `Noul`, `Score` or `Choice` output for each dimension{which}")
        });
        // What is near is measured on text: a picture has no coordinates here.
        let problem = problem.or_else(|| {
            let picture = signature.images().first().map(|name| name.to_string());
            picture.map(|name| format!("The dimensions' input `{name}` is a picture, and a picture has no coordinates: the input is the text that says what is wanted."))
        });
        let questions = signature
            .outputs()
            .iter()
            .map(|(name, output)| (name.clone(), output.description.clone()));
        let labels = Labels {
            questions: questions.collect(),
            ..Labels::default()
        };
        let state = State {
            top: 8,
            measure: Measure::Dot,
            uses: false,
            threads: 8,
            model: None,
            labels,
            dimensions: signature.clone(),
            params: IndexMap::new(),
            use_params: IndexMap::new(),
            version: 0,
        };
        let inner = Inner {
            input: signature.inputs().keys().next().cloned().unwrap_or_default(),
            asks: signature,
            problem,
            state: Mutex::new(state),
            family: Arc::default(),
        };
        Near { inner: Arc::new(inner) }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn set(self, change: impl FnOnce(&mut State)) -> Self {
        change(&mut self.state());
        self
    }

    /// How many names a request offers at most, beside those always offered. 8 unless set.
    pub fn top(self, names: usize) -> Self {
        self.set(|state| state.top = names)
    }

    /// How two sets of coordinates are compared, the larger the nearer. The dot product unless set.
    pub fn measure(self, measure: Measure) -> Self {
        self.set(|state| state.measure = measure)
    }

    /// Whether a request is also asked how it uses each dimension: as a filter, so that only
    /// names that have it are offered; as a boost; or not at all. One request more for each
    /// request. A filter that would leave no name is let go.
    pub fn uses(self, asked: bool) -> Self {
        self.set(|state| state.uses = asked)
    }

    /// How many texts are asked about at once where no run is. 8 unless set. In a run, the
    /// module's own `threads` say.
    pub fn threads(self, width: usize) -> Self {
        self.set(|state| state.threads = width.max(1))
    }

    /// The decision model that is asked where no run is: [`Near::embed`], used alone. In a
    /// run the questions are sub-calls of the run, and the module's model answers them.
    pub fn model(self, model: impl DecisionModel + 'static) -> Self {
        self.set_model(model);
        self
    }

    /// A threshold, cuts or weights for a dimension, by its name. See [`Near::set_params`].
    pub fn params(self, dimension: impl Into<String>, params: Params) -> Self {
        self.set_params(dimension, params);
        self
    }

    /// The same `Near` with these labels in place of its own. See [`Near::set_labels`].
    pub fn with_labels(self, labels: Labels) -> Self {
        self.set_labels(labels);
        self
    }

    /// Whether it is built in a way it can be used: the dimensions one input and a decision
    /// for each output, and `top` at least 1. Everything that asks checks.
    pub fn check(&self) -> Result<()> {
        if let Some(problem) = &self.inner.problem {
            return Err(Error::Usage(problem.clone()));
        }
        if self.state().top < 1 {
            return Err(Error::Usage("top must be at least 1: a request has to offer something.".to_owned()));
        }
        Ok(())
    }

    /// Give it the decision model that is asked where no run is, or another one.
    pub fn set_model(&self, model: impl DecisionModel + 'static) {
        self.state().model = Some(Arc::new(model));
    }

    /// The decision model that is asked where no run is, if it has one.
    pub fn decision_model(&self) -> Option<Arc<dyn DecisionModel>> {
        self.state().model.clone()
    }

    /// The name of the dimensions' input: what a request's input must be named to be asked about.
    pub fn input(&self) -> &str {
        &self.inner.input
    }

    /// The questions that make the dimensions, as they are asked now: the signature it was
    /// given, with what [`Near::add`] and [`Near::remove`] have done to it since.
    pub fn dimensions(&self) -> Signature {
        self.state().dimensions.clone()
    }

    /// Every dimension's question, by name.
    pub fn questions(&self) -> IndexMap<String, String> {
        self.state().labels.questions.clone()
    }

    /// What each text asked so far answered, by dimension, as the model gave it.
    ///
    /// They are the answers of one model, and are dropped when another is asked.
    pub fn answers(&self) -> Answers {
        self.state().labels.answers.clone()
    }

    /// The labels as they are now: the questions, the answers, the uses, and the model.
    pub fn labels(&self) -> Labels {
        self.state().labels.clone()
    }

    /// Put labels in place of its own: those of a program that was saved. The dimensions are
    /// then the ones the labels hold, whatever it was built with: one it was built with keeps
    /// its kind, and any other is a yes or no. A parameter of a dimension that is gone goes too.
    pub fn set_labels(&self, labels: Labels) {
        let mut state = self.state();
        state.labels = labels;
        let State {
            labels,
            params,
            use_params,
            ..
        } = &mut *state;
        params.retain(|name, _| labels.questions.contains_key(name));
        use_params.retain(|name, _| labels.questions.contains_key(name));
        state.sync(&self.inner.asks);
    }

    /// Everything that is the program's: the labels, and the parameters of the dimensions.
    ///
    /// ```
    /// use rdm::near::{Near, Saved};
    /// use rdm::{Params, Signature};
    ///
    /// let about = || Signature::new("Say what a text is about.").input("task");
    /// let near = Near::new(about());
    /// near.add("heating", "Is `inputs.task` about how warm it is?").unwrap();
    /// near.set_params("heating", Params::threshold(0.7));
    ///
    /// // The embedding is part of the program: written as JSON, and read into one built the same way.
    /// let written = serde_json::to_string(&near.saved()).unwrap();
    /// let fresh = Near::new(about());
    /// fresh.load(serde_json::from_str::<Saved>(&written).unwrap());
    /// assert_eq!(fresh.questions(), near.questions());
    /// assert_eq!(fresh.params_of("heating"), Some(Params::threshold(0.7)));
    /// ```
    pub fn saved(&self) -> Saved {
        let state = self.state();
        Saved {
            labels: state.labels.clone(),
            dimensions: state.params.clone(),
            uses: state.use_params.clone(),
        }
    }

    /// Read what was saved in place of what it has: the labels, and the parameters of the dimensions.
    pub fn load(&self, saved: Saved) {
        let mut state = self.state();
        state.labels = saved.labels;
        state.params = saved.dimensions;
        state.use_params = saved.uses;
        state.sync(&self.inner.asks);
    }

    /// A `Near` of its own that is set as this one is, with a copy of the labels and the
    /// parameters. Writing a dimension into one from then on does not change the other. A
    /// copy of a module ([`crate::Rdm::copy`]) is given one.
    pub fn copy(&self) -> Near {
        let state = self.state();
        let copied = State {
            top: state.top,
            measure: state.measure.clone(),
            uses: state.uses,
            threads: state.threads,
            model: state.model.clone(),
            labels: state.labels.clone(),
            dimensions: state.dimensions.clone(),
            params: state.params.clone(),
            use_params: state.use_params.clone(),
            version: state.version,
        };
        let inner = Inner {
            asks: self.inner.asks.clone(),
            input: self.inner.input.clone(),
            problem: self.inner.problem.clone(),
            state: Mutex::new(copied),
            family: self.inner.family.clone(),
        };
        Near { inner: Arc::new(inner) }
    }

    /// Set the numbers a dimension's label is read with: a yes or no's threshold, under which
    /// it counts as nothing; a score's cuts, which give it its level; a choice's weights,
    /// which multiply its options. They are read each time a label is, so changing one asks
    /// nothing again.
    ///
    /// ```
    /// use rdm::near::{Near, coordinates};
    /// use rdm::{Params, Signature};
    /// use systemone::{Answer, Scripted};
    ///
    /// let about = Signature::new("Say what a text is about.").input("task").noul("heating", "Is `inputs.task` about how warm it is?");
    /// let near = Near::new(about).model(Scripted::new(|_| [("heating", Answer::noul(0.6))]));
    /// assert_eq!(near.embed(["It is a bit stuffy."]).unwrap(), [coordinates([("heating", 0.6)])]);
    /// near.set_params("heating", Params::threshold(0.75));
    /// assert_eq!(near.embed(["It is a bit stuffy."]).unwrap(), [coordinates([("heating", 0.0)])]);
    /// assert_eq!(near.answers()["It is a bit stuffy."]["heating"], 0.6); // what is kept is what the model gave
    /// ```
    pub fn set_params(&self, dimension: impl Into<String>, params: Params) {
        self.state().params.insert(dimension.into(), params);
    }

    /// Forget the numbers set for a dimension: its label is what the model answered again.
    pub fn unset_params(&self, dimension: &str) -> Option<Params> {
        self.state().params.shift_remove(dimension)
    }

    /// The numbers set for a dimension, if any were.
    pub fn params_of(&self, dimension: &str) -> Option<Params> {
        self.state().params.get(dimension).cloned()
    }

    /// Every dimension that has numbers set, with them.
    pub fn all_params(&self) -> IndexMap<String, Params> {
        self.state().params.clone()
    }

    /// Set the weights of `filter`, `boost` and `ignore` for a dimension: what each is
    /// multiplied by when a request says how it uses the dimension.
    pub fn set_use_params(&self, dimension: impl Into<String>, params: Params) {
        self.state().use_params.insert(dimension.into(), params);
    }

    /// Forget the weights set for how a dimension is used.
    pub fn unset_use_params(&self, dimension: &str) -> Option<Params> {
        self.state().use_params.shift_remove(dimension)
    }

    /// Every dimension that has weights set for how it is used, with them.
    pub fn all_use_params(&self) -> IndexMap<String, Params> {
        self.state().use_params.clone()
    }

    /// Add a dimension, a yes or no question, or put another question under a name there already is.
    ///
    /// The name is one word, as a name is written in code. The question is a yes or no
    /// question about a text, which is referred to as the dimensions' input is
    /// (`inputs.task`, say). What was answered to a question that was there before goes: the
    /// names on the map are asked the new one when next they are needed.
    ///
    /// A name or a question that will not do is a tool's failure and not a usage error, since
    /// a writer wrote it: in a tool it becomes the observation, and the writer can try again.
    ///
    /// ```
    /// use rdm::Signature;
    /// use rdm::near::Near;
    ///
    /// let near = Near::new(Signature::new("Say what a text is about.").input("task"));
    /// let said = near.add("doors", "Is `inputs.task` about a door or its lock?").unwrap();
    /// assert_eq!(said, "`doors` is now a dimension: Is `inputs.task` about a door or its lock?");
    /// assert_eq!(near.show(), "doors: Is `inputs.task` about a door or its lock?");
    /// assert_eq!(near.remove("doors"), "`doors` is no longer a dimension.");
    /// assert_eq!(near.show(), "There is no dimension yet.");
    /// ```
    pub fn add(&self, name: &str, question: &str) -> Result<String> {
        self.added(name, question, None)
    }

    /// As [`Near::add`], inside a run: the change is recorded in the trajectory, under
    /// `changes` on the request whose tool made it.
    ///
    /// Where the module that is running is a copy ([`crate::Rdm::copy`]) of the one this
    /// `Near` was given to, the dimension is written into the copy's own `Near`, which is the
    /// one the copy's next request is cut by. So it is for [`Near::remove_in`].
    pub fn add_in(&self, name: &str, question: &str, call: &Call) -> Result<String> {
        self.of(call).added(name, question, Some(call))
    }

    /// The `Near` of the run a tool is in: the one the module that is running has for a fit,
    /// where that is this one or a copy of it, and otherwise this one. A tool holds the `Near`
    /// it was made with, and a copy of its module has the same tool and a copy of the `Near`.
    fn of(&self, call: &Call) -> Near {
        let fits = call.active.module.fits();
        let related = fits
            .iter()
            .filter_map(|fit| fit.as_any()?.downcast_ref::<Near>())
            .filter(|near| Arc::ptr_eq(&near.inner.family, &self.inner.family));
        let (mut this, mut copy) = (false, None);
        for near in related {
            this |= Arc::ptr_eq(&near.inner, &self.inner);
            copy.get_or_insert(near);
        }
        match copy {
            Some(copy) if !this => copy.clone(),
            _ => self.clone(),
        }
    }

    fn added(&self, name: &str, question: &str, call: Option<&Call>) -> Result<String> {
        self.check()?;
        let (name, question) = (stripped(name), stripped(question));
        let input = &self.inner.input;
        if !one_word(name) || name == input || question.is_empty() {
            return Err(Error::Tool(format!(
                "A dimension is a name, written as one word and not `{input}`, and a question; got {} and {}.",
                repr(name),
                repr(question)
            )));
        }
        if RESERVED.contains(&name) {
            return Err(Error::Tool(format!(
                "A dimension cannot be named `{name}`: a call keeps the names {} for itself.",
                sorted(RESERVED)
            )));
        }
        let mut state = self.state();
        let known = state.labels.questions.contains_key(name);
        if state.labels.questions.get(name).map(String::as_str) != Some(question) {
            state.labels.questions.insert(name.to_owned(), question.to_owned());
            state.forget(name); // what was answered was the answer to another question
            state.sync(&self.inner.asks);
            drop(state);
            if let Some(call) = call {
                call.changed(json!({"by": "near", "did": "add", "name": name}));
            }
        }
        Ok(if known {
            format!("`{name}` now asks: {question}")
        } else {
            format!("`{name}` is now a dimension: {question}")
        })
    }

    /// Remove a dimension, with what was answered for it and any parameter it was given.
    /// Returns what a writer is told: that it is gone, or that there was none of that name.
    pub fn remove(&self, name: &str) -> String {
        self.removed(name, None)
    }

    /// As [`Near::remove`], inside a run: the change is recorded in the trajectory, under
    /// `changes` on the request whose tool made it.
    pub fn remove_in(&self, name: &str, call: &Call) -> String {
        self.of(call).removed(name, Some(call))
    }

    fn removed(&self, name: &str, call: Option<&Call>) -> String {
        let mut state = self.state();
        if state.labels.questions.shift_remove(name).is_none() {
            return format!(
                "There is no dimension `{name}`: the dimensions are {}.",
                listed(state.labels.questions.keys())
            );
        }
        state.forget(name);
        state.sync(&self.inner.asks);
        drop(state);
        if let Some(call) = call {
            call.changed(json!({"by": "near", "did": "remove", "name": name}));
        }
        format!("`{name}` is no longer a dimension.")
    }

    /// Every dimension there is: its name and its question. What a writer reads first.
    pub fn show(&self) -> String {
        let state = self.state();
        let lines: Vec<String> = state
            .labels
            .questions
            .iter()
            .map(|(name, question)| format!("{name}: {question}"))
            .collect();
        if lines.is_empty() {
            "There is no dimension yet.".to_owned()
        } else {
            lines.join("\n")
        }
    }

    /// The coordinates of each text: a number for every yes or no dimension and every score,
    /// and one for each option of a choice.
    ///
    /// A text is asked once, and only a text that lacks an answer is asked: all its dimensions
    /// in one request, put to the model given with [`Near::model`]. A dimension's threshold,
    /// cuts or weights are read now, so changing one asks nothing again. With no dimension
    /// yet there is nothing to ask, and every text is nowhere.
    ///
    /// This is outside any run and counts against no limit: it is how a long map is labelled
    /// before a run, so that the run asks about its request only. Inside a run, use
    /// [`Near::embed_in`].
    ///
    /// ```
    /// use rdm::Signature;
    /// use rdm::near::{Near, coordinates};
    /// use systemone::{Answer, Scripted};
    ///
    /// let about = Signature::new("Say what a text is about.")
    ///     .input("task")
    ///     .noul("heating", "Is `inputs.task` about how warm it is?")
    ///     .noul("lights", "Is `inputs.task` about a light?");
    /// let model = Scripted::new(|request| {
    ///     let text = request.state["inputs"]["task"].as_str().unwrap_or_default();
    ///     [("heating", Answer::noul(if text.contains("cooler") { 0.9 } else { 0.0 })), ("lights", Answer::noul(if text.contains("lamp") { 0.8 } else { 0.0 }))]
    /// });
    /// let near = Near::new(about).model(model);
    ///
    /// let embedded = near.embed(["Make it cooler.", "The lamp in the hall."]).unwrap();
    /// assert_eq!(embedded, [coordinates([("heating", 0.9), ("lights", 0.0)]), coordinates([("heating", 0.0), ("lights", 0.8)])]);
    /// // Things are compared by what their coordinates share: a search by named dimensions.
    /// let wanted = coordinates([("heating", 1.0), ("lights", 0.0)]);
    /// assert!(near.score(&wanted, &embedded[0]) > near.score(&wanted, &embedded[1]));
    /// ```
    pub fn embed<T: AsRef<str>>(&self, texts: impl IntoIterator<Item = T>) -> Result<Vec<Coordinates>> {
        self.embedded(&owned(texts), None)
    }

    /// As [`Near::embed`], inside a run: what is asked are sub-calls of the call, for the
    /// answers only, admitted as one batch and put to the module's own decision model. They
    /// count against the run's limits and are in its trajectory, one level down. A text whose
    /// request a limit stopped is not kept, and is nowhere.
    pub fn embed_in<T: AsRef<str>>(&self, texts: impl IntoIterator<Item = T>, call: &Call) -> Result<Vec<Coordinates>> {
        self.embedded(&owned(texts), Some(call))
    }

    /// Each text as a row of numbers, one for each coordinate, in the order of
    /// [`Near::columns`]: what a search that takes plain vectors is given.
    ///
    /// The rows are as the dimensions and their parameters are now, so build an index of them
    /// after those are settled.
    pub fn rows<T: AsRef<str>>(&self, texts: impl IntoIterator<Item = T>) -> Result<Vec<Vec<f64>>> {
        let embedded = self.embed(texts)?;
        Ok(embedded
            .into_iter()
            .map(|coordinates| coordinates.into_values().collect())
            .collect())
    }

    /// The name of each coordinate, in the order a row has them.
    pub fn columns(&self) -> Vec<String> {
        self.state().coordinates(None).into_keys().collect()
    }

    /// How near a thing is to a request, by the measure chosen: the larger, the nearer.
    pub fn score(&self, request: &Coordinates, thing: &Coordinates) -> f64 {
        let measure = self.state().measure.clone();
        measure.score(request, thing)
    }

    fn embedded(&self, texts: &[String], call: Option<&Call>) -> Result<Vec<Coordinates>> {
        self.check()?;
        let (signature, names, version, new, model, threads) = {
            let mut state = self.state();
            if state.labels.questions.is_empty() {
                return Ok(texts.iter().map(|_| Coordinates::new()).collect()); // no dimension yet: nothing to ask
            }
            let model = match call {
                Some(call) => call.rdm().decision_model(),
                None => state.model.clone(),
            };
            state.same_model(model.as_deref());
            let new: IndexSet<&str> = texts.iter().map(String::as_str).filter(|text| !state.answered(text)).collect();
            let names: Vec<String> = state.labels.questions.keys().cloned().collect();
            (state.dimensions.clone(), names, state.version, new, model, state.threads)
        };
        let new: Vec<&str> = new.into_iter().collect();
        let asked = self.ask(&signature, &new, call, model, threads)?;
        // In the order they were asked: the labels are kept in the order the texts were given.
        let mut fresh: IndexMap<&str, IndexMap<String, Given>> = IndexMap::new();
        for (text, answered) in new.into_iter().zip(asked) {
            // A request a limit stopped answers nothing.
            let given: Option<IndexMap<String, Given>> = answered.and_then(|answered| {
                let each = names.iter().map(|name| Some((name.clone(), Given::of(answered.get(name)?))));
                each.collect()
            });
            fresh.extend(given.map(|given| (text, given)));
        }
        let mut state = self.state();
        if state.version == version {
            for (text, given) in &fresh {
                state.labels.answers.insert((*text).to_owned(), given.clone());
            }
        }
        let read = |text: &String| state.coordinates(fresh.get(text.as_str()).or_else(|| state.labels.answers.get(text)));
        Ok(texts.iter().map(read).collect())
    }

    /// Each text's answers to a signature's questions. None for a text whose request a limit stopped.
    fn ask(
        &self,
        signature: &Signature,
        texts: &[&str],
        call: Option<&Call>,
        model: Option<Arc<dyn DecisionModel>>,
        threads: usize,
    ) -> Result<Vec<Option<IndexMap<String, Decision>>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input = &self.inner.input;
        if let Some(call) = call {
            // In a run: sub-calls, for the answers only, which the module puts to its model.
            let asks = texts.iter().map(|text| {
                let inputs = Inputs::new().with(input.clone(), *text);
                Ask::new(inputs).signature(signature.clone()).answers_only()
            });
            let predictions = call.batch(asks)?;
            return Ok(predictions
                .into_iter()
                .map(|prediction| prediction.answers.into_iter().map(|(name, answer)| Some((name, answer?))).collect())
                .collect());
        }
        let model = model.ok_or_else(|| {
            Error::Usage(
                "The dimensions have no decision model of their own: give one with `model(...)` or `set_model(...)`, \
                 or ask inside a run with `embed_in`."
                    .to_owned(),
            )
        })?;
        // A choice with one option leaves nothing to decide: it is answered with that option and not asked.
        let only: IndexMap<String, Decision> = signature
            .outputs()
            .iter()
            .filter_map(|(name, output)| Some((name.clone(), output.only()?)))
            .collect();
        let asked = only.keys().fold(signature.clone(), |asked, name| asked.without_output(name));
        let one = |position: usize| -> Result<Option<IndexMap<String, Decision>>> {
            let mut answers = only.clone();
            if !asked.outputs().is_empty() {
                let mut shown = Map::new();
                shown.insert(input.clone(), Json::String(texts[position].to_owned()));
                let request = asked.request(&shown)?;
                let response = model.decide(&request)?;
                for (name, question) in &request.questions {
                    let answer = response
                        .answers
                        .get(name)
                        .ok_or_else(|| systemone::Error::Evidence(format!("The model gave no answer for '{name}'.")))?;
                    // What is kept is what the model gave: a dimension's parameters are read when a label is.
                    answers.insert(name.clone(), systemone::decide(name, question, answer, &Params::default())?);
                }
            }
            Ok(Some(answers))
        };
        at_once(threads, texts.len(), one).into_iter().collect()
    }

    /// How a request uses each dimension: how much it counts, and which dimensions are filters.
    fn used(&self, text: &str, call: &Call) -> Result<(IndexMap<String, f64>, Vec<String>)> {
        let (signature, version, kept, model, threads) = {
            let state = self.state();
            let kept = state.labels.uses.get(text).filter(|kept| state.use_answered(kept)).cloned();
            let model = call.rdm().decision_model();
            (state.uses_signature(&self.inner.asks), state.version, kept, model, state.threads)
        };
        let kept = match kept {
            Some(kept) => kept,
            None => {
                let answered = self.ask(&signature, &[text], Some(call), model, threads)?.pop().flatten();
                let given: Option<IndexMap<String, Given>> = answered.and_then(|answered| {
                    let each = signature
                        .outputs()
                        .keys()
                        .map(|name| Some((name.clone(), Given::of(answered.get(name)?))));
                    each.collect()
                });
                // A limit stopped the request: every dimension counts as it is.
                let Some(given) = given else {
                    return Ok((IndexMap::new(), Vec::new()));
                };
                let mut state = self.state();
                if state.version == version {
                    state.labels.uses.insert(text.to_owned(), given.clone());
                }
                given
            }
        };
        let state = self.state();
        let (mut weights, mut filters) = (IndexMap::new(), Vec::new());
        for name in state.labels.questions.keys() {
            let Some(likely) = kept.get(&format!("{name}_use")).and_then(Given::probabilities) else {
                continue; // the questions have changed since: the dimension counts as it is
            };
            let weight = state.use_params.get(name).and_then(|params| params.weights.as_ref());
            let weighted = |label: &String| weight.and_then(|weight| weight.get(label)).copied().unwrap_or(1.0);
            let scores: Vec<(&str, f64)> = likely
                .iter()
                .map(|(label, probability)| (label.as_str(), probability * weighted(label)))
                .collect();
            let whole = total(scores.iter().map(|(_, score)| *score));
            let ignored = scores.iter().find(|(label, _)| *label == "ignore").map_or(0.0, |(_, score)| *score);
            // A dimension counts as far as the request does not ignore it, and is a filter where that is its likeliest use.
            weights.insert(name.clone(), 1.0 - ignored / if whole == 0.0 { 1.0 } else { whole });
            let mut likeliest: Option<(&str, f64)> = None;
            for (label, score) in scores {
                if likeliest.is_none_or(|(_, held)| score > held) {
                    likeliest = Some((label, score));
                }
            }
            if likeliest.is_some_and(|(label, _)| label == "filter") {
                filters.push(name.clone());
            }
        }
        Ok((weights, filters))
    }
}

impl State {
    /// Make the dimensions ask the questions the labels hold: after one is written, and after a program is loaded.
    fn sync(&mut self, asks: &Signature) {
        let questions = &self.labels.questions;
        let mut signature = Signature::new(asks.instructions());
        for (name, field) in asks.inputs() {
            signature = signature.with_input(name, field.clone());
        }
        for (name, output) in asks.outputs() {
            if let Some(question) = questions.get(name) {
                signature = signature.with_output(name, output.clone().with_description(question));
            }
        }
        for (name, question) in questions {
            if signature.output(name).is_none() {
                signature = signature.with_output(name, Output::noul(question)); // a dimension that was written
            }
        }
        self.dimensions = signature;
        self.version += 1;
    }

    /// The questions a request is asked about how it uses each dimension.
    fn uses_signature(&self, asks: &Signature) -> Signature {
        let mut signature = Signature::new(USE_INSTRUCTIONS);
        for (name, field) in asks.inputs() {
            signature = signature.with_input(name, field.clone());
        }
        let input = asks.inputs().keys().next().map_or("", String::as_str);
        for (name, question) in &self.labels.questions {
            let asks = format!("How does `inputs.{input}` use the label `{name}`, which asks of a thing: {question}");
            signature = signature.choice(format!("{name}_use"), asks, USES);
        }
        signature
    }

    /// Drop what another model answered: a label is one model's answer.
    fn same_model(&mut self, model: Option<&dyn DecisionModel>) {
        if let Some(model) = model.map(DecisionModel::name) {
            if self.labels.model.as_ref().is_some_and(|kept| *kept != model) {
                self.labels.answers.clear();
                self.labels.uses.clear();
                self.version += 1; // what the other model is still being asked is not kept either
            }
            self.labels.model = Some(model);
        }
    }

    /// Drop what was answered for a dimension, and its parameters where the dimension is gone.
    fn forget(&mut self, name: &str) {
        for answered in self.labels.answers.values_mut() {
            answered.shift_remove(name);
        }
        self.labels.uses.clear();
        if !self.labels.questions.contains_key(name) {
            self.params.shift_remove(name);
            self.use_params.shift_remove(name);
        }
    }

    /// Whether a text has an answer to every dimension, each of the kind its dimension is.
    fn answered(&self, text: &str) -> bool {
        let kept = self.labels.answers.get(text);
        self.labels.questions.keys().all(|name| {
            let given = kept.and_then(|kept| kept.get(name));
            let kind = self.dimensions.output(name).map(|output| &output.space);
            matches!(
                (given, kind),
                (Some(Given::Noul(_)), Some(Space::Noul { .. }))
                    | (Some(Given::Probabilities { .. }), Some(Space::Choice(_) | Space::Score(_)))
            )
        })
    }

    /// Whether what a request said of its uses covers every dimension there is.
    fn use_answered(&self, kept: &IndexMap<String, Given>) -> bool {
        let said = |name: &String| {
            kept.get(&format!("{name}_use"))
                .is_some_and(|given| given.probabilities().is_some())
        };
        self.labels.questions.keys().all(said)
    }

    /// A text's coordinates from what it answered, under the parameters of the dimensions as they are now.
    fn coordinates(&self, answered: Option<&IndexMap<String, Given>>) -> Coordinates {
        let mut coordinates = Coordinates::new();
        for (name, output) in self.dimensions.outputs() {
            let given = answered.and_then(|answered| answered.get(name));
            let params = self.params.get(name);
            match &output.space {
                Space::Choice(options) => {
                    // A dimension for each option, as likely as the option was, times its weight where it has one.
                    let weights = params.and_then(|params| params.weights.as_ref());
                    let likely = given.and_then(Given::probabilities);
                    for option in options.iter().filter(|option| !option.none && option.label != NONE) {
                        let probability = likely.and_then(|likely| likely.get(&option.label)).copied().unwrap_or(0.0);
                        let weight = weights.and_then(|weights| weights.get(&option.label)).copied().unwrap_or(1.0);
                        coordinates.insert(format!("{name}.{}", option.label), probability * weight);
                    }
                }
                Space::Score(levels) => {
                    // How far up the rubric: the level its cuts give it where it has cuts, the mean level otherwise.
                    let value = mean(given);
                    let cuts = params.and_then(|params| params.cuts.as_ref()).filter(|cuts| !cuts.is_empty());
                    let place = match cuts {
                        Some(cuts) => cuts.iter().filter(|cut| value >= **cut).count() as f64 / cuts.len() as f64,
                        None => value / levels.len().saturating_sub(1).max(1) as f64,
                    };
                    coordinates.insert(name.clone(), place);
                }
                Space::Noul { .. } => {
                    // How likely a yes is, and nothing under its threshold where it has one.
                    let probability = given.and_then(Given::noul).unwrap_or(0.0);
                    let threshold = params.and_then(|params| params.threshold);
                    let counts = threshold.is_none_or(|threshold| probability >= threshold);
                    coordinates.insert(name.clone(), if counts { probability } else { 0.0 });
                }
                Space::Tool => {}
            }
        }
        coordinates
    }
}

/// Given to a module as its fit, a `Near` cuts the output that points at a tool to the names
/// nearest the request. Any other output, and any other choice, is left as it is.
impl Fit for Near {
    fn fit(&self, mut signature: Signature, inputs: Inputs, call: &Call) -> Result<(Signature, Inputs)> {
        self.check()?;
        // The request's input the dimensions are asked about is text. Any other input may be a picture.
        let input = &self.inner.input;
        if signature.images().contains(&input.as_str()) {
            return Err(Error::Usage(format!(
                "The dimensions take `{input}`, and the request's `{input}` is a picture: what is near is measured on the text that says what is wanted."
            )));
        }
        let names: Vec<String> = signature.outputs().keys().cloned().collect();
        for name in names {
            let (top, uses, measure, none) = {
                let state = self.state();
                (state.top, state.uses, state.measure.clone(), state.labels.questions.is_empty())
            };
            let Some(output) = signature.output(&name).filter(|output| output.points && !none) else {
                continue;
            };
            let Some(options) = output.options() else { continue };
            let (always, measured): (Vec<Opt>, Vec<Opt>) = options.iter().cloned().partition(|option| output.stays(option));
            if measured.len() <= top {
                continue;
            }
            let input = &self.inner.input;
            let Some(asked) = inputs.value(input) else {
                return Err(Error::Usage(format!(
                    "The dimensions take `{input}`, and the request has only {}: name the dimensions' input after the \
                     request's input that says what is wanted.",
                    sorted(inputs.names())
                )));
            };
            let asked = shown(&asked);
            let things = measured.iter().map(|option| text(&option.label, &option.description));
            let texts: Vec<String> = std::iter::once(asked.clone()).chain(things).collect();
            let mut things = self.embedded(&texts, Some(call))?;
            let mut request = things.remove(0);
            let mut filters = Vec::new();
            if uses {
                let (weights, filtering) = self.used(&asked, call)?;
                for (key, value) in request.iter_mut() {
                    *value *= weights.get(dimension(key)).copied().unwrap_or(1.0);
                }
                filters = filtering;
            }
            // What shares nothing with the request is not near it at all, however it is measured.
            let mut near: Vec<usize> = (0..things.len())
                .filter(|position| dot(&request, &things[*position]) > 0.0)
                .collect();
            // A dimension the request filters on: only a name that shares it will do. A filter that leaves nothing is let go.
            for filter in &filters {
                let part: Coordinates = request
                    .iter()
                    .filter(|(key, _)| dimension(key) == filter)
                    .map(|(key, value)| (key.clone(), *value))
                    .collect();
                let has: Vec<usize> = near
                    .iter()
                    .copied()
                    .filter(|position| dot(&part, &things[*position]) > 0.0)
                    .collect();
                if !has.is_empty() {
                    near = has;
                }
            }
            if near.is_empty() {
                continue; // nothing sets the names apart
            }
            // The nearest, in the order they had.
            let scores: HashMap<usize, f64> = near
                .iter()
                .map(|position| (*position, measure.score(&request, &things[*position])))
                .collect();
            let mut nearest = near;
            nearest.sort_by(|one, other| scores[other].partial_cmp(&scores[one]).unwrap_or(Ordering::Equal));
            nearest.truncate(top);
            nearest.sort_unstable();
            let offered = nearest.into_iter().map(|position| measured[position].clone()).chain(always);
            signature = signature.with_options(&name, offered);
        }
        Ok((signature, inputs))
    }

    fn copy(&self) -> Option<Arc<dyn Fit>> {
        Some(Arc::new(Near::copy(self)))
    }

    /// So that a tool that holds a `Near` finds the one the module running was given.
    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

impl fmt::Debug for Near {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state();
        let mut near = f.debug_struct("Near");
        near.field("input", &self.inner.input).field("top", &state.top);
        near.field("measure", &state.measure).field("uses", &state.uses);
        near.field("questions", &state.labels.questions)
            .field("texts", &state.labels.answers.len())
            .field("model", &state.labels.model)
            .finish()
    }
}

/// The dimension a coordinate is of: its own name, or for an option of a choice the name before the point.
fn dimension(key: &str) -> &str {
    key.split('.').next().unwrap_or(key)
}

/// The mean level of a score as it was kept.
fn mean(given: Option<&Given>) -> f64 {
    let Some(levels) = given.and_then(Given::probabilities) else {
        return 0.0;
    };
    let whole = total(levels.values().copied());
    let weighted = total(
        levels
            .iter()
            .map(|(level, probability)| level.parse::<f64>().unwrap_or(0.0) * probability),
    );
    weighted / if whole == 0.0 { 1.0 } else { whole }
}

/// A request's input as the text that is asked about: a text as it is, anything else as JSON,
/// written as the reference writes it: a space after each comma and colon, and whatever is not
/// plain ASCII as an escape. The text is what the labels are kept under, so it has to be the same.
fn shown(value: &Json) -> String {
    let written = match value {
        Json::String(text) => return text.clone(),
        Json::Null => "null".to_owned(),
        other => as_observation(other),
    };
    // Outside a text JSON is ASCII, and inside one what is below a space is an escape already.
    let mut plain = String::with_capacity(written.len());
    let mut units = [0; 2];
    for character in written.chars() {
        if character.is_ascii() && character != '\u{7f}' {
            plain.push(character);
        } else {
            for unit in character.encode_utf16(&mut units) {
                plain.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    plain
}

fn owned<T: AsRef<str>>(texts: impl IntoIterator<Item = T>) -> Vec<String> {
    texts.into_iter().map(|text| text.as_ref().to_owned()).collect()
}

/// Whether a name is one word, as a name is written in code.
fn one_word(name: &str) -> bool {
    let mut characters = name.chars();
    let starts = characters.next().is_some_and(|first| first.is_alphabetic() || first == '_');
    starts && characters.all(|character| character.is_alphanumeric() || character == '_')
}

/// A text as Python prints one inside a message, which is how the messages of the reference quote what they were
/// given: in single quotes, or in double ones where it has a single quote and no double one, with what cannot be
/// seen written as an escape.
fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut quoted = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            other if other == quote => {
                quoted.push('\\');
                quoted.push(other);
            }
            other if unseen(other) => {
                let code = u32::from(other);
                let escape = match code {
                    0..=0xff => format!("\\x{code:02x}"),
                    0x100..=0xffff => format!("\\u{code:04x}"),
                    _ => format!("\\U{code:08x}"),
                };
                quoted.push_str(&escape);
            }
            other => quoted.push(other),
        }
    }
    quoted.push(quote);
    quoted
}

/// Whether a character is one that cannot be seen when printed: a control, a space other than the plain one, or
/// one that only formats what is round it. The ones a writer's text may carry; not every one there is.
fn unseen(character: char) -> bool {
    let formats = matches!(
        u32::from(character),
        0xad | 0x61c | 0x180e | 0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x206f | 0xfeff | 0xfff9..=0xfffb | 0xe000..=0xf8ff
    );
    character.is_control() || (character.is_whitespace() && character != ' ') || formats
}

/// A text without the space round it, as Python's `strip` leaves it.
fn stripped(text: &str) -> &str {
    text.trim_matches(|character: char| character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character))
}

/// Do a thing for each position at once, at most `width` at a time, and return what each came to, in order.
/// Where no thread can be started, each is done after the one before.
fn at_once<T: Send>(width: usize, count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if count <= 1 || width <= 1 || !crate::parallel::THREADS {
        return (0..count).map(work).collect();
    }
    let next = AtomicUsize::new(0);
    let share = || {
        let mut done = Vec::new();
        loop {
            let position = next.fetch_add(1, Counting::Relaxed);
            if position >= count {
                return done;
            }
            done.push((position, work(position)));
        }
    };
    let mut done: Vec<(usize, T)> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..count.min(width)).map(|_| scope.spawn(share)).collect();
        let joined = threads
            .into_iter()
            .map(|thread| thread.join().unwrap_or_else(|panic| resume_unwind(panic)));
        joined.flatten().collect()
    });
    done.sort_by_key(|(position, _)| *position);
    done.into_iter().map(|(_, result)| result).collect()
}
