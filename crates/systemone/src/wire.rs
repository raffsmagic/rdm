//! The request and the response, as they are on the wire.
//!
//! ```text
//! POST {base}/v1/systemone
//! {"model": "...", "state": ..., "questions": {name: {"instructions", "type", "criteria"?}}}
//! {"model": "...", "answers": {name: {"type", ...}}, "usage": {"input_tokens", "output_tokens"}}
//! ```

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Any JSON value.
pub type Json = serde_json::Value;

/// A state and the typed questions asked about it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// The model to ask, or the name a gateway routes by. Empty for "whichever this model is".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// What the questions are about: a string, an object or an array.
    ///
    /// An item of an array that is `{"type": "image_url", "image_url": {"url": "data:..."}}`
    /// is a picture, shown to a model that can see ([`crate::image`]). The key a request is
    /// kept under ([`crate::key`]) covers a picture's bytes: two pictures are two keys.
    pub state: Json,
    /// The questions, by name, in the order they are asked.
    pub questions: IndexMap<String, Question>,
}

impl Request {
    /// A request about a state, with no question yet and no model named.
    pub fn new(state: impl Into<Json>) -> Self {
        Self {
            model: String::new(),
            state: state.into(),
            questions: IndexMap::new(),
        }
    }

    /// The same request, for the model of this name.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// How many pictures the state shows ([`crate::images`]).
    pub fn images(&self) -> usize {
        crate::images(&self.state)
    }

    /// The same request with one more question, under the name its answer comes back under.
    pub fn question(mut self, name: impl Into<String>, question: Question) -> Self {
        self.questions.insert(name.into(), question);
        self
    }
}

/// The three kinds of question there are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Yes or no: answered with the probability of yes.
    Noul,
    /// One of several options: answered with a probability for each.
    Choice,
    /// A level on an ordered rubric: answered with a probability for each level.
    Score,
}

impl Kind {
    /// The kind as it is written on the wire: `"noul"`, `"choice"` or `"score"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }
}

/// One question. `criteria` is its answer space: for a choice an object of option to
/// description (null where an option has none), for a score an array of level descriptions,
/// lowest first, and for a noul nothing, or what `"true"` and `"false"` mean.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// What is asked, in words: a string as a rule. It can refer to the state, as `inputs.message` does.
    #[serde(default)]
    pub instructions: Json,
    /// Which kind of question it is. `"type"` on the wire.
    #[serde(rename = "type")]
    pub kind: Kind,
    /// The answer space: the options of a choice, the levels of a score, and for a noul
    /// nothing, or what a yes and a no mean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Json>,
}

impl Question {
    /// A yes or no question.
    pub fn noul(instructions: impl Into<Json>) -> Self {
        Self {
            instructions: instructions.into(),
            kind: Kind::Noul,
            criteria: None,
        }
    }

    /// A yes or no question that says what each answer means.
    pub fn noul_with(instructions: impl Into<Json>, yes: Option<&str>, no: Option<&str>) -> Self {
        let mut criteria = serde_json::Map::new();
        if let Some(yes) = yes {
            criteria.insert("true".into(), yes.into());
        }
        if let Some(no) = no {
            criteria.insert("false".into(), no.into());
        }
        Self {
            instructions: instructions.into(),
            kind: Kind::Noul,
            criteria: Some(Json::Object(criteria)),
        }
    }

    /// A choice among options, each a label and what it means. An empty description is sent as null.
    pub fn choice<L, D>(instructions: impl Into<Json>, options: impl IntoIterator<Item = (L, D)>) -> Self
    where
        L: Into<String>,
        D: Into<Option<String>>,
    {
        let mut criteria = serde_json::Map::new();
        for (label, description) in options {
            let description = description.into().filter(|text| !text.is_empty());
            criteria.insert(label.into(), description.map_or(Json::Null, Json::String));
        }
        Self {
            instructions: instructions.into(),
            kind: Kind::Choice,
            criteria: Some(Json::Object(criteria)),
        }
    }

    /// A score against levels, lowest first.
    pub fn score<D: Into<Json>>(instructions: impl Into<Json>, levels: impl IntoIterator<Item = D>) -> Self {
        let levels = levels.into_iter().map(Into::into).collect();
        Self {
            instructions: instructions.into(),
            kind: Kind::Score,
            criteria: Some(Json::Array(levels)),
        }
    }

    /// The labels an answer gives a probability to, in the order they were declared: the
    /// options of a choice, `"0"` to `"n-1"` for a score of n levels, nothing for a noul.
    pub fn labels(&self) -> Vec<String> {
        match (self.kind, &self.criteria) {
            (Kind::Choice, Some(Json::Object(options))) => options.keys().cloned().collect(),
            (Kind::Score, Some(Json::Array(levels))) => (0..levels.len()).map(|level| level.to_string()).collect(),
            _ => Vec::new(),
        }
    }
}

/// What a model answered: the evidence, before anything is decided from it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// Which kind of question the server says it answered. `"type"` on the wire.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Kind>,
    /// For a noul: how likely a yes is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,
    /// For a choice: the option the server picked. Nothing here is decided from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,
    /// For a score: the level the server expects. Nothing here is decided from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// For a choice or a score: how sure the server is, from 0 to 1. Where it is given, a
    /// decision keeps it; where it is not, it is worked out from the probabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// For a score: each level's description, by its index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legend: Option<Json>,
    /// For a choice or a score: how likely each label is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<IndexMap<String, f64>>,
}

impl Answer {
    /// A noul answered with the probability of yes.
    pub fn noul(probability: f64) -> Self {
        Self {
            kind: Some(Kind::Noul),
            noul: Some(probability),
            ..Self::default()
        }
    }

    /// A choice answered with a probability for each option. The pick and the confidence
    /// are worked out as TypeSafe publishes them: `(p_max - 1/n) / (1 - 1/n)`.
    pub fn choice<L: Into<String>>(probabilities: impl IntoIterator<Item = (L, f64)>) -> Self {
        let probabilities: IndexMap<String, f64> = probabilities.into_iter().map(|(label, p)| (label.into(), p)).collect();
        let confidence = crate::decide::choice_confidence(probabilities.values().copied());
        let choice = crate::decide::likeliest(&probabilities).map(str::to_owned);
        Self {
            kind: Some(Kind::Choice),
            choice,
            confidence: Some(confidence),
            probabilities: Some(probabilities),
            ..Self::default()
        }
    }

    /// A choice answered with one option, as certain.
    pub fn chose<'a>(option: &str, options: impl IntoIterator<Item = &'a str>) -> Self {
        Self::choice(options.into_iter().map(|label| (label, if label == option { 1.0 } else { 0.0 })))
    }

    /// A score answered with a probability for each level, lowest first.
    pub fn score(levels: impl IntoIterator<Item = f64>) -> Self {
        let levels: Vec<f64> = levels.into_iter().collect();
        let confidence = crate::decide::score_confidence(&levels);
        let total: f64 = levels.iter().sum();
        let expected = if total > 0.0 {
            levels.iter().enumerate().map(|(level, p)| level as f64 * p).sum::<f64>() / total
        } else {
            0.0
        };
        let probabilities = levels.iter().enumerate().map(|(level, p)| (level.to_string(), *p)).collect();
        Self {
            kind: Some(Kind::Score),
            score: Some(expected),
            confidence: Some(confidence),
            probabilities: Some(probabilities),
            ..Self::default()
        }
    }
}

/// The tokens a request used, as the server counted them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// The tokens of the request: the state and the questions, as the model read them.
    #[serde(default)]
    pub input_tokens: u64,
    /// The tokens of the answer.
    #[serde(default)]
    pub output_tokens: u64,
}

impl Usage {
    /// Both together: what the request cost.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// The answers to a request, by the name of the question.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// The model that answered.
    #[serde(default)]
    pub model: String,
    /// The evidence for each question, under the question's name.
    pub answers: IndexMap<String, Answer>,
    /// The tokens the request used. All zero where the server did not say.
    #[serde(default)]
    pub usage: Usage,
}
