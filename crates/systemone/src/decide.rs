//! From evidence to a decision.
//!
//! A model answers with probabilities. What is decided from them is decided here, by plain
//! arithmetic and a few named numbers: a noul's threshold, a score's cuts, a choice's weights.
//! Those numbers are never sent to a model, so changing one decides again without asking again.
//! What the server itself picked (`choice`, `score`) is not read.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::wire::{Answer, Kind, Question};

/// A yes or no, with how likely the yes was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Noul {
    /// How far the probability is from the threshold: `|p - t| / max(t, 1 - t)`, which is 0
    /// at the threshold and 1 as far from it as a probability can be. It says how sure the
    /// yes or the no is, whichever it was.
    pub confidence: f64,
    /// The decision: whether the probability is at or over the threshold.
    pub value: bool,
    /// How likely a yes is, as the model answered it. A no at 0.04 has this at 0.04.
    pub probability: f64,
}

/// One option of several, with how likely each was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// How far the likeliest option is above chance, from 0 to 1: the server's own where it
    /// gave one, else [`choice_confidence`].
    pub confidence: f64,
    /// The decision: the label of the option chosen, the likeliest once weights are applied.
    pub value: String,
    /// How likely each option is, as the model answered it, in the order the options were declared.
    pub probabilities: IndexMap<String, f64>,
}

/// A place on a rubric: the expected level, the level its cuts give it, and how likely each level was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// How closely the levels gather round their mean, from 0 to 1: the server's own where
    /// it gave one, else [`score_confidence`].
    pub confidence: f64,
    /// The expected level, a number between the lowest level's index and the highest's:
    /// `sum(i * p_i) / sum(p_i)`. With levels 0, 1 and 2 as likely as each other it is 1.0.
    pub value: f64,
    /// By level index, as a string: `"0"`, `"1"`, ...
    pub probabilities: IndexMap<String, f64>,
    /// The decision: the index of the one level the value falls in, which is how many cuts
    /// it is at or over. Where `value` is 1.4 and the cuts are 0.5 and 1.5, it is 1.
    pub level: usize,
}

/// A decided answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Decision {
    /// A yes or no.
    Noul(Noul),
    /// A place on a rubric.
    Score(Score),
    /// One option of several.
    Choice(Choice),
}

impl Decision {
    /// Which kind of question this answers.
    pub fn kind(&self) -> Kind {
        match self {
            Decision::Noul(_) => Kind::Noul,
            Decision::Choice(_) => Kind::Choice,
            Decision::Score(_) => Kind::Score,
        }
    }

    /// The decision without its probabilities: a bool, a label, or a number.
    pub fn value(&self) -> serde_json::Value {
        match self {
            Decision::Noul(noul) => noul.value.into(),
            Decision::Choice(choice) => choice.value.clone().into(),
            Decision::Score(score) => score.value.into(),
        }
    }

    /// How sure the decision is, from 0 to 1, as its kind measures it.
    pub fn confidence(&self) -> f64 {
        match self {
            Decision::Noul(noul) => noul.confidence,
            Decision::Choice(choice) => choice.confidence,
            Decision::Score(score) => score.confidence,
        }
    }

    /// The yes or no, if this is one.
    pub fn as_noul(&self) -> Option<&Noul> {
        if let Decision::Noul(noul) = self { Some(noul) } else { None }
    }

    /// The option chosen, if this is a choice.
    pub fn as_choice(&self) -> Option<&Choice> {
        if let Decision::Choice(choice) = self { Some(choice) } else { None }
    }

    /// The place on the rubric, if this is a score.
    pub fn as_score(&self) -> Option<&Score> {
        if let Decision::Score(score) = self { Some(score) } else { None }
    }

    /// A choice with one option, answered without asking: that option, as certain.
    pub fn only(option: impl Into<String>) -> Self {
        let option = option.into();
        Decision::Choice(Choice {
            confidence: 1.0,
            probabilities: IndexMap::from([(option.clone(), 1.0)]),
            value: option,
        })
    }
}

/// The numbers a decision is made with. Each is for one kind of question, and the default
/// is what a question gets with none: a threshold of 0.5, cuts halfway between levels, and
/// every weight 1.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Params {
    /// Noul: a yes is a probability at or over this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Score: the boundaries between levels, one fewer than there are levels, increasing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuts: Option<Vec<f64>>,
    /// Choice: what each option's probability is multiplied by before the likeliest is taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<IndexMap<String, f64>>,
}

impl Params {
    /// For a noul: a yes is a probability at or over `threshold`, in place of 0.5.
    pub fn threshold(threshold: f64) -> Self {
        Self {
            threshold: Some(threshold),
            ..Self::default()
        }
    }

    /// For a score: the boundaries between levels, one fewer than there are levels and
    /// increasing, in place of those halfway between.
    pub fn cuts(cuts: impl IntoIterator<Item = f64>) -> Self {
        Self {
            cuts: Some(cuts.into_iter().collect()),
            ..Self::default()
        }
    }

    /// For a choice: what an option's probability is multiplied by before the likeliest is
    /// taken. An option not named has a weight of 1.
    pub fn weights<L: Into<String>>(weights: impl IntoIterator<Item = (L, f64)>) -> Self {
        Self {
            weights: Some(weights.into_iter().map(|(label, weight)| (label.into(), weight)).collect()),
            ..Self::default()
        }
    }

    /// Whether none of the numbers is set, so that every question is decided with the defaults.
    pub fn is_empty(&self) -> bool {
        self.threshold.is_none() && self.cuts.is_none() && self.weights.is_none()
    }
}

fn unit(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

/// Decide the answer to one question from the evidence a model gave for it.
///
/// - Noul: `value = p >= threshold`, `confidence = |p - t| / max(t, 1 - t)`.
/// - Choice: the option with the largest `p * weight`. A tie goes to the option that was
///   likeliest before weights, and then to the one declared first.
/// - Score: `value = sum(i * p_i) / sum(p_i)`, and `level` is how many cuts the value is at or over.
///
/// Evidence that does not fit the question is an error: a probability outside 0 to 1, labels
/// other than the ones declared, or no probability anywhere.
pub fn decide(name: &str, question: &Question, answer: &Answer, params: &Params) -> Result<Decision, Error> {
    let bad = |what: &str| Error::Evidence(format!("Invalid {} answer for '{name}': {what}", question.kind.as_str()));
    match question.kind {
        Kind::Noul => {
            let probability = answer.noul.ok_or_else(|| bad("no `noul` probability"))?;
            let threshold = params.threshold.unwrap_or(0.5);
            if !unit(probability) {
                return Err(bad("the probability is not between 0 and 1"));
            }
            if !unit(threshold) {
                return Err(Error::Invalid(format!(
                    "Invalid threshold for '{name}': it must be between 0 and 1."
                )));
            }
            let confidence = (probability - threshold).abs() / threshold.max(1.0 - threshold);
            Ok(Decision::Noul(Noul {
                confidence,
                value: probability >= threshold,
                probability,
            }))
        }
        Kind::Choice => {
            let labels = question.labels();
            let given = answer.probabilities.as_ref().ok_or_else(|| bad("no `probabilities`"))?;
            let probabilities = ordered(&labels, given).ok_or_else(|| bad("the probabilities are not for the options declared"))?;
            if probabilities.values().any(|p| !unit(*p)) || probabilities.values().sum::<f64>() <= 0.0 {
                return Err(bad("the probabilities are not a distribution"));
            }
            let empty = IndexMap::new();
            let weights = params.weights.as_ref().unwrap_or(&empty);
            if weights.keys().any(|label| !labels.contains(label)) || weights.values().any(|weight| !weight.is_finite() || *weight < 0.0) {
                return Err(Error::Invalid(format!(
                    "Invalid weights for '{name}': a weight is a number, zero or more, for a declared option."
                )));
            }
            let raw = likeliest(&probabilities).unwrap_or_default().to_owned();
            let mut best: Option<(&str, f64)> = None;
            for (label, probability) in &probabilities {
                let weighted = probability * weights.get(label).copied().unwrap_or(1.0);
                let better = match best {
                    None => true,
                    Some((_, held)) => weighted > held || (weighted == held && *label == raw),
                };
                if better {
                    best = Some((label, weighted));
                }
            }
            let (value, weighted) = best.ok_or_else(|| bad("there are no options"))?;
            if weighted <= 0.0 {
                return Err(Error::Invalid(format!(
                    "Choice weights for '{name}' leave no positive probability mass."
                )));
            }
            let confidence = match answer.confidence {
                Some(confidence) if unit(confidence) => confidence,
                Some(_) => return Err(bad("the confidence is not between 0 and 1")),
                None => choice_confidence(probabilities.values().copied()),
            };
            Ok(Decision::Choice(Choice {
                confidence,
                value: value.to_owned(),
                probabilities,
            }))
        }
        Kind::Score => {
            let labels = question.labels();
            let levels = labels.len();
            if !(2..=10).contains(&levels) {
                return Err(Error::Invalid(format!(
                    "Invalid rubric for '{name}': a score has two to ten levels."
                )));
            }
            let given = answer.probabilities.as_ref().ok_or_else(|| bad("no `probabilities`"))?;
            let probabilities = ordered(&labels, given).ok_or_else(|| bad("the probabilities are not for the levels declared"))?;
            let total: f64 = probabilities.values().sum();
            if probabilities.values().any(|p| !unit(*p)) || total <= 0.0 {
                return Err(bad("the probabilities are not a distribution"));
            }
            let value = probabilities.values().enumerate().map(|(level, p)| level as f64 * p).sum::<f64>() / total;
            let default: Vec<f64> = (0..levels - 1).map(|level| level as f64 + 0.5).collect();
            let cuts = params.cuts.as_ref().unwrap_or(&default);
            let top = (levels - 1) as f64;
            let increasing = cuts.windows(2).all(|pair| pair[0] < pair[1]);
            if cuts.len() != levels - 1 || !increasing || cuts.iter().any(|cut| !cut.is_finite() || *cut <= 0.0 || *cut >= top) {
                return Err(Error::Invalid(format!(
                    "Invalid cuts for '{name}': require ordered boundaries inside the Score index range."
                )));
            }
            let level = cuts.iter().filter(|cut| value >= **cut).count();
            let confidence = match answer.confidence {
                Some(confidence) if unit(confidence) => confidence,
                Some(_) => return Err(bad("the confidence is not between 0 and 1")),
                None => score_confidence(&probabilities.values().copied().collect::<Vec<_>>()),
            };
            Ok(Decision::Score(Score {
                confidence,
                value,
                probabilities,
                level,
            }))
        }
    }
}

/// The probabilities in the order the labels were declared, if they are for exactly those labels.
/// A server may give them in another order, and which comes first decides a tie.
fn ordered(labels: &[String], given: &IndexMap<String, f64>) -> Option<IndexMap<String, f64>> {
    if given.len() != labels.len() {
        return None;
    }
    labels.iter().map(|label| given.get(label).map(|p| (label.clone(), *p))).collect()
}

/// The likeliest label; of several as likely, the first.
pub fn likeliest(probabilities: &IndexMap<String, f64>) -> Option<&str> {
    let mut best: Option<(&str, f64)> = None;
    for (label, probability) in probabilities {
        if best.is_none_or(|(_, held)| *probability > held) {
            best = Some((label, *probability));
        }
    }
    best.map(|(label, _)| label)
}

/// How far the likeliest option is above chance: `(p_max - 1/n) / (1 - 1/n)`.
pub fn choice_confidence(probabilities: impl IntoIterator<Item = f64>) -> f64 {
    let probabilities: Vec<f64> = probabilities.into_iter().collect();
    let count = probabilities.len() as f64;
    if count <= 1.0 {
        return 1.0;
    }
    let total: f64 = probabilities.iter().sum();
    let most = probabilities.iter().copied().fold(0.0, f64::max) / if total > 0.0 { total } else { 1.0 };
    ((most - 1.0 / count) / (1.0 - 1.0 / count)).clamp(0.0, 1.0)
}

/// How closely the levels gather round their mean, against levels all as likely as each other:
/// `max(0, 1 - sum(p_i * |i - m|) / MAD_uniform)`.
pub fn score_confidence(levels: &[f64]) -> f64 {
    let total: f64 = levels.iter().sum();
    let count = levels.len();
    if count <= 1 || total <= 0.0 {
        return 1.0;
    }
    let mean = levels.iter().enumerate().map(|(level, p)| level as f64 * p).sum::<f64>() / total;
    let spread = levels
        .iter()
        .enumerate()
        .map(|(level, p)| p / total * (level as f64 - mean).abs())
        .sum::<f64>();
    let middle = (count - 1) as f64 / 2.0;
    let uniform = (0..count).map(|level| (level as f64 - middle).abs()).sum::<f64>() / count as f64;
    (1.0 - spread / uniform).clamp(0.0, 1.0)
}
