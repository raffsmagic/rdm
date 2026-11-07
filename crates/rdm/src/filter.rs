//! Filter: sorts what a decision model rated, likeliest first, and keeps the likeliest.
//!
//! A tool of an [`Rdm`](crate::Rdm) often has to act on more than the one answer a decision
//! came to: follow the two likeliest options of a choice, or keep the items a yes or no was
//! said of. [`Filter`] is the sorting and the keeping:
//!
//! ```text
//! let likeliest = Filter::new().top(2);
//!
//! Tool::new("category", "...", move |call| {
//!     for (option, probability) in likeliest.options(call.answer("category")) {   // the choice's options, likeliest first
//!         ...
//!     }
//!     let verdicts = call.batch(leaves.iter().map(|leaf| Ask::new(inputs! { "leaf" => leaf }).signature(fits.clone()).max_depth(0)))?;
//!     Ok(likeliest.rated(leaves.iter().zip(verdicts.iter().map(|verdict| verdict.noul("fits")))))   // the leaves said yes of, likeliest first
//! })
//! ```
//!
//! It has no cut of its own. A yes is a yes where the threshold of its output says so, and
//! that threshold is a parameter of the module ([`Rdm::set_params`](crate::Rdm::set_params)).

use std::cmp::Ordering;

use systemone::{Choice, Decision, Noul};

use crate::prediction::Prediction;

/// Sorts what a decision model rated by its probabilities, and keeps the likeliest.
///
/// Each of its readers returns `(thing, probability)` pairs, likeliest first; things rated
/// alike keep the order they came in. What was rated can be:
///
/// - a choice answer ([`Filter::options`]): its options, each by its label;
/// - a prediction ([`Filter::yeses`]): its yes or no answers, by the name of the output;
/// - pairs of anything with the yes or no given about it, or with a probability
///   ([`Filter::rated`]).
///
/// Of yes or no answers only the yeses are kept. Whether an answer is a yes is its output's
/// threshold, so the cut is a parameter of the module and not of the filter: set it on the
/// module (`rdm.set_params("fits", Params::threshold(0.7))`).
///
/// ```
/// use rdm::filter::Filter;
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let ways = Signature::new("Find the way out.").input("room").choice("go", "Which way out of `inputs.room`?", ["north", "south", "east"]);
/// let likeliest = Filter::new().top(2);
/// // Follows the two ways the model rated likeliest, not only the one it chose.
/// let go = Tool::new("go", "Try the likeliest ways.", move |call| {
///     let tried: Vec<String> = likeliest.options(call.answer("go")).into_iter().map(|(way, probability)| format!("{way} ({probability})")).collect();
///     Ok(tried)
/// });
/// let model = Scripted::new(|_| [("go", Answer::choice([("north", 0.3), ("south", 0.1), ("east", 0.6)]))]);
///
/// let result = Rdm::new(ways).tool(go).model(model).call(inputs! { "room" => "the cellar" }).unwrap();
/// assert_eq!(result.chosen("go"), Some("east"));
/// assert_eq!(result.observations["go"], serde_json::json!(["east (0.6)", "north (0.3)"]));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    top: Option<usize>,
}

impl Filter {
    /// A filter that only sorts: it keeps everything that was rated, likeliest first.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many to keep at most: a setting like the width of a beam.
    ///
    /// # Panics
    ///
    /// With a `top` of 0: a filter that keeps nothing has nothing to give.
    pub fn top(mut self, top: usize) -> Self {
        assert!(top >= 1, "top must be at least 1: a filter that keeps nothing has nothing to give.");
        self.top = Some(top);
        self
    }

    /// The options of a choice answer, each with its probability, likeliest first.
    ///
    /// The answer is given as a tool has it: `call.answer("category")`,
    /// `prediction.choice("category")`, or the [`Choice`] itself. Each option is its label; the
    /// option of choosing none of the others is the label [`NONE`](crate::NONE). An answer that
    /// came with no probabilities gives the one option chosen, as certain. An answer that is
    /// absent, as one a limit kept from being asked is, or that is not a choice, gives nothing.
    ///
    /// ```
    /// use rdm::Output;
    /// use rdm::filter::Filter;
    /// use rdm::systemone::{Answer, Params, decide};
    ///
    /// let question = Output::choice("Which way?", ["north", "south", "east"]).question().unwrap();
    /// let evidence = Answer::choice([("north", 0.5), ("south", 0.3), ("east", 0.2)]);
    /// let answer = decide("way", &question, &evidence, &Params::default()).unwrap();
    /// assert_eq!(Filter::new().top(2).options(&answer), [("north".to_owned(), 0.5), ("south".to_owned(), 0.3)]);
    /// ```
    pub fn options(&self, answer: impl Chosen) -> Vec<(String, f64)> {
        let Some(choice) = answer.chosen() else {
            return Vec::new();
        };
        if choice.probabilities.is_empty() {
            return vec![(choice.value.clone(), 1.0)];
        }
        let options = choice
            .probabilities
            .iter()
            .map(|(label, probability)| (label.clone(), *probability));
        self.likeliest(options.collect())
    }

    /// The yes or no answers of a prediction that are a yes, by the name of the output, likeliest first.
    pub fn yeses(&self, prediction: &Prediction) -> Vec<(String, f64)> {
        let nouls = prediction
            .answers
            .iter()
            .filter_map(|(name, answer)| Some((name.clone(), answer.as_ref()?.as_noul()?)));
        self.rated(nouls)
    }

    /// Things, each with what was said of it: the ones kept, each with its probability, likeliest first.
    ///
    /// What was said is a yes or no answer (a [`Noul`], as `prediction.noul("fits")` gives it)
    /// or a probability. A thing said no of is dropped, and so is one with no answer: a
    /// request a limit kept from being asked says yes of nothing. A thing can be anything.
    ///
    /// ```
    /// use rdm::filter::Filter;
    ///
    /// let rated = [("east", 0.1), ("north", 0.5), ("south", 0.4)];
    /// assert_eq!(Filter::new().rated(rated), [("north", 0.5), ("south", 0.4), ("east", 0.1)]);
    /// assert_eq!(Filter::new().top(1).rated(rated), [("north", 0.5)]);
    /// ```
    pub fn rated<T, R: Rating>(&self, rated: impl IntoIterator<Item = (T, R)>) -> Vec<(T, f64)> {
        let kept = rated.into_iter().filter_map(|(thing, said)| Some((thing, said.probability()?)));
        self.likeliest(kept.collect())
    }

    /// What was kept, likeliest first, and no more of it than `top`.
    fn likeliest<T>(&self, mut kept: Vec<(T, f64)>) -> Vec<(T, f64)> {
        // The sort keeps things rated alike in the order they came in. A probability that is
        // not a number comes after every one that is.
        let later = |a: f64, b: f64| b.partial_cmp(&a).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()));
        kept.sort_by(|(_, a), (_, b)| -> Ordering { later(*a, *b) });
        kept.truncate(self.top.unwrap_or(usize::MAX));
        kept
    }
}

/// An answer that may be a choice: what [`Filter::options`] reads.
pub trait Chosen {
    /// The choice, if that is what the answer is.
    fn chosen(&self) -> Option<&Choice>;
}

impl Chosen for Choice {
    fn chosen(&self) -> Option<&Choice> {
        Some(self)
    }
}

impl Chosen for Decision {
    fn chosen(&self) -> Option<&Choice> {
        self.as_choice()
    }
}

impl<A: Chosen> Chosen for Option<A> {
    fn chosen(&self) -> Option<&Choice> {
        self.as_ref().and_then(Chosen::chosen)
    }
}

impl<A: Chosen + ?Sized> Chosen for &A {
    fn chosen(&self) -> Option<&Choice> {
        (**self).chosen()
    }
}

/// What was said of a thing: a yes or no answer, or a probability. What [`Filter::rated`] reads.
pub trait Rating {
    /// How likely the thing is, when it is kept. None for a no, and for no answer.
    fn probability(&self) -> Option<f64>;
}

/// A yes is kept, with how likely it was. A no is dropped.
impl Rating for Noul {
    fn probability(&self) -> Option<f64> {
        self.value.then_some(self.probability)
    }
}

/// A probability is kept as it is.
impl Rating for f64 {
    fn probability(&self) -> Option<f64> {
        Some(*self)
    }
}

/// An answer that is absent says yes of nothing.
impl<R: Rating> Rating for Option<R> {
    fn probability(&self) -> Option<f64> {
        self.as_ref().and_then(Rating::probability)
    }
}

impl<R: Rating + ?Sized> Rating for &R {
    fn probability(&self) -> Option<f64> {
        (**self).probability()
    }
}
