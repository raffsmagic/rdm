//! What a call returns, and the record of a run.

use std::fmt;

use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use systemone::{Choice, Decision, Json, Noul, Score};

use crate::signature::{NONE, Signature};

/// Why a call ended: a decision, or what stopped something in it or below it other than one.
///
/// They are in the order of what bounds the most. When several stopped parts of a run, the
/// first of them here is the reason: a limit of the run, then a limit of a call, then a repeat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// The run used the tokens it may.
    MaxTokens,
    /// The run asked the requests it may.
    MaxPredictorCalls,
    /// A pointer's loop took the turns it may.
    MaxIters,
    /// A call at the depth limit was answered, and the tools it selected were not run.
    MaxDepth,
    /// A request had already been asked on the way to it, and was not asked again.
    Repeat,
    /// Nothing was left out: no answer selected a tool, or the pointer chose `finish`.
    Final,
}

impl Reason {
    /// The reason as the trajectory writes it: `max_depth`, `repeat`, `final`.
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::MaxTokens => "max_tokens",
            Reason::MaxPredictorCalls => "max_predictor_calls",
            Reason::MaxIters => "max_iters",
            Reason::MaxDepth => "max_depth",
            Reason::Repeat => "repeat",
            Reason::Final => "final",
        }
    }

    /// The reason that bounds the most, of several. `Final` when nothing was stopped.
    pub fn of(reasons: impl IntoIterator<Item = Reason>) -> Reason {
        reasons.into_iter().min().unwrap_or(Reason::Final)
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for Reason {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<str> for Reason {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

/// One request of a run, as the trajectory records it. It serializes with the keys the
/// Python module uses, so whatever reads a Python run reads this one: an answer that chose
/// none of its options is written with a null `value`, and read back as one.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(try_from = "Read")]
pub struct Entry {
    /// How many calls are above it. The first call of a run is at 0.
    pub depth: usize,
    /// Which turn of a pointer's loop this is.
    pub iteration: usize,
    /// The request it came from: the one whose tool made it, or the turn before it.
    pub parent: Option<usize>,
    /// The tool that led to it: the one that made the sub-call, or the one the turn before ran.
    pub tool_name: Option<String>,
    /// What the request showed, as it was then.
    pub inputs: Json,
    /// What a fit used on a model of its own.
    pub fit_tokens: Option<u64>,
    /// What the request itself used, as the decision model reported it.
    pub tokens: Option<u64>,
    /// The model that answered.
    pub model: Option<String>,
    /// The answers, each with its probabilities. Absent for a request a limit stopped.
    pub outputs: Option<IndexMap<String, Decision>>,
    /// The outputs that had one option, and were answered without asking.
    pub unasked: Option<Vec<String>>,
    /// What each tool the answers selected returned, by the output that selected it.
    pub observations: Option<IndexMap<String, Json>>,
    /// What each tool used on a model of its own, by the tool.
    pub tool_tokens: Option<IndexMap<String, u64>>,
    /// The limit that kept the request from being asked.
    pub skipped: Option<Reason>,
    /// Whether the request had been asked on the way to it, and was answered from then.
    pub repeat: bool,
    /// Changes a tool made to a map while this request's tools ran.
    pub changes: Option<Vec<Json>>,
    /// The outputs whose answer chose the option of choosing none ([`crate::Opt::none`]): their
    /// plain value is null. It is not a key of its own when written: it is the null `value`s.
    pub none: Vec<String>,
}

/// An entry as it is read: an answer may have a null `value`.
#[derive(Deserialize)]
struct Read {
    depth: usize,
    iteration: usize,
    parent: Option<usize>,
    tool_name: Option<String>,
    inputs: Json,
    #[serde(default)]
    fit_tokens: Option<u64>,
    #[serde(default)]
    tokens: Option<u64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    outputs: Option<IndexMap<String, Json>>,
    #[serde(default)]
    unasked: Option<Vec<String>>,
    #[serde(default)]
    observations: Option<IndexMap<String, Json>>,
    #[serde(default)]
    tool_tokens: Option<IndexMap<String, u64>>,
    #[serde(default)]
    skipped: Option<Reason>,
    #[serde(default)]
    repeat: bool,
    #[serde(default)]
    changes: Option<Vec<Json>>,
}

impl TryFrom<Read> for Entry {
    type Error = serde_json::Error;

    fn try_from(read: Read) -> Result<Self, Self::Error> {
        let (mut none, mut outputs) = (Vec::new(), None);
        if let Some(read) = read.outputs {
            let mut decided = IndexMap::new();
            for (name, mut answer) in read {
                // Choosing none is written as a null value, and held as the label of that option.
                if answer.get("value").is_some_and(Json::is_null) {
                    answer["value"] = Json::String(NONE.to_owned());
                    none.push(name.clone());
                }
                decided.insert(name, serde_json::from_value(answer)?);
            }
            outputs = Some(decided);
        }
        Ok(Entry {
            depth: read.depth,
            iteration: read.iteration,
            parent: read.parent,
            tool_name: read.tool_name,
            inputs: read.inputs,
            fit_tokens: read.fit_tokens,
            tokens: read.tokens,
            model: read.model,
            outputs,
            unasked: read.unasked,
            observations: read.observations,
            tool_tokens: read.tool_tokens,
            skipped: read.skipped,
            repeat: read.repeat,
            changes: read.changes,
            none,
        })
    }
}

/// A decision as it is written: one that chose none of its options has a null `value`, as
/// the Python module writes it.
struct Written<'a>(&'a Decision, bool);

impl Serialize for Written<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Written(Decision::Choice(choice), true) => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("confidence", &choice.confidence)?;
                map.serialize_entry("value", &())?;
                map.serialize_entry("probabilities", &choice.probabilities)?;
                map.end()
            }
            Written(decision, _) => decision.serialize(serializer),
        }
    }
}

/// The key order is the Python module's: what is absent is left out.
impl Serialize for Entry {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        /// The answers of a request, by output.
        struct Outputs<'a>(&'a IndexMap<String, Decision>, &'a [String]);
        impl Serialize for Outputs<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(Some(self.0.len()))?;
                for (name, decision) in self.0 {
                    map.serialize_entry(name, &Written(decision, self.1.contains(name)))?;
                }
                map.end()
            }
        }
        fn put<M: SerializeMap, T: Serialize>(map: &mut M, key: &str, value: &Option<T>) -> Result<(), M::Error> {
            value.as_ref().map_or(Ok(()), |value| map.serialize_entry(key, value))
        }
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("depth", &self.depth)?;
        map.serialize_entry("iteration", &self.iteration)?;
        map.serialize_entry("parent", &self.parent)?;
        map.serialize_entry("tool_name", &self.tool_name)?;
        map.serialize_entry("inputs", &self.inputs)?;
        put(&mut map, "fit_tokens", &self.fit_tokens)?;
        put(&mut map, "tokens", &self.tokens)?;
        put(&mut map, "model", &self.model)?;
        put(
            &mut map,
            "outputs",
            &self.outputs.as_ref().map(|outputs| Outputs(outputs, &self.none)),
        )?;
        put(&mut map, "unasked", &self.unasked)?;
        put(&mut map, "observations", &self.observations)?;
        put(&mut map, "tool_tokens", &self.tool_tokens)?;
        put(&mut map, "skipped", &self.skipped)?;
        if self.repeat {
            map.serialize_entry("repeat", &true)?;
        }
        put(&mut map, "changes", &self.changes)?;
        map.end()
    }
}

impl Entry {
    /// Whether a decision model was asked this request: not one a limit stopped, a repeat, or one with nothing to decide.
    pub fn asked(&self) -> bool {
        self.model.is_some()
    }

    /// Every token used for this request: its own, and what its tools and its fit reported.
    /// What the requests below it used is on those requests.
    pub fn spent(&self) -> u64 {
        let tools: u64 = self.tool_tokens.iter().flat_map(|tools| tools.values()).sum();
        self.tokens.unwrap_or(0) + self.fit_tokens.unwrap_or(0) + tools
    }

    /// An answer's plain value: a bool, a label or a number. Null when there is none, and
    /// for a choice that chose none of its options, as [`Prediction::value`] reads it.
    pub fn value(&self, name: &str) -> Json {
        let answer = self.outputs.as_ref().and_then(|outputs| outputs.get(name));
        match answer {
            Some(_) if self.chose_none(name) => Json::Null,
            Some(decision) => decision.value(),
            None => Json::Null,
        }
    }

    /// Whether an answer chose the option of choosing none.
    pub fn chose_none(&self, name: &str) -> bool {
        self.none.iter().any(|none| none == name)
    }

    /// Record the answers of the request: each with its probabilities, and which of them chose none.
    pub(crate) fn answered(&mut self, signature: &Signature, answers: IndexMap<String, Decision>) {
        let chose_none = |(name, decision): (&String, &Decision)| signature.output(name)?.chose_none(decision).then(|| name.clone());
        self.none = answers.iter().filter_map(chose_none).collect();
        self.outputs = Some(answers);
    }
}

/// What a call returns: its own answers, what its tools came to, and why it ended.
///
/// The answers are those of the request itself, asked once: a sub-call's answers are its
/// own, and reach the caller as the observation of the tool that made it. For the call that
/// starts a run there is the trajectory as well.
///
/// ```
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let triage = Signature::new("Triage a message from a customer.")
///     .input("message")
///     .noul("refund", "Is the customer in `inputs.message` asking for a refund?")
///     .choice("team", "Which team should handle `inputs.message`?", ["billing", "sales"]);
/// let refund = Tool::new("refund", "Open a refund with a team.", |call| {
///     let team: String = call.arg("team")?;
///     Ok(format!("refund opened with {team}"))
/// });
/// let model = Scripted::new(|_| [("refund", Answer::noul(0.9)), ("team", Answer::chose("billing", ["billing", "sales"]))]);
///
/// let result = Rdm::new(triage).tool(refund).model(model).call(inputs! { "message" => "I was charged twice." }).unwrap();
/// assert!(result.yes("refund"));
/// assert_eq!(result.noul("refund").unwrap().probability, 0.9);
/// assert_eq!(result.chosen("team"), Some("billing"));
/// assert_eq!(result.observations["refund"], "refund opened with billing");
/// assert_eq!(result.termination_reason, "final");
/// assert_eq!(result.trajectory.len(), 1);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// The answer to each output, with its probabilities. None for a request a limit kept from being asked.
    pub answers: IndexMap<String, Option<Decision>>,
    /// What each selected tool returned, by the output that selected it. For the pointer,
    /// every tool it pointed at in turn, each with what it returned: `[[name, result], ...]`.
    pub observations: IndexMap<String, Json>,
    /// `Final`, or what stopped something in this call or below it other than a decision.
    pub termination_reason: Reason,
    /// Every request of the run, in the order it was recorded. Empty for a sub-call.
    pub trajectory: Vec<Entry>,
    /// What arbitration came to, where the module has an arbiter and the answers selected a
    /// tool: which results the request comes to, and what was stopped, started and held
    /// ([`crate::arbitrate`]). None otherwise: every tool selected ran, and each result stands.
    pub arbitration: Option<crate::arbitrate::Arbitration>,
    /// What was done outside in the run, by any tool at any depth, as each was done
    /// ([`crate::Call::acting`]): kept whatever became of the branch that did it. Empty for a
    /// sub-call.
    pub receipts: Vec<Receipt>,
    /// The outputs that chose the option of choosing none.
    none: Vec<String>,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            answers: IndexMap::new(),
            observations: IndexMap::new(),
            termination_reason: Reason::Final,
            trajectory: Vec::new(),
            arbitration: None,
            receipts: Vec::new(),
            none: Vec::new(),
        }
    }
}

impl Prediction {
    /// The answers to a signature's questions, with nothing observed yet.
    pub fn of(signature: &Signature, answers: IndexMap<String, Option<Decision>>) -> Self {
        let chose_none = |name: &String| {
            let output = signature.output(name)?;
            let decision = answers.get(name)?.as_ref()?;
            output.chose_none(decision).then(|| name.clone())
        };
        let none = answers.keys().filter_map(chose_none).collect();
        Self {
            answers,
            none,
            ..Self::default()
        }
    }

    /// An answer, whatever kind it is. None for one that was not answered.
    pub fn get(&self, name: &str) -> Option<&Decision> {
        self.answers.get(name).and_then(Option::as_ref)
    }

    /// A yes or no, with how likely the yes was.
    pub fn noul(&self, name: &str) -> Option<&Noul> {
        self.get(name).and_then(Decision::as_noul)
    }

    /// A choice, with how likely each option was.
    pub fn choice(&self, name: &str) -> Option<&Choice> {
        self.get(name).and_then(Decision::as_choice)
    }

    /// A score, with how likely each level was.
    pub fn score(&self, name: &str) -> Option<&Score> {
        self.get(name).and_then(Decision::as_score)
    }

    /// Whether a noul was answered yes. False for a no, and for one that was not answered.
    pub fn yes(&self, name: &str) -> bool {
        self.noul(name).is_some_and(|noul| noul.value)
    }

    /// How likely a yes was, as the model gave it.
    pub fn probability(&self, name: &str) -> Option<f64> {
        self.noul(name).map(|noul| noul.probability)
    }

    /// The option a choice came to. None when it chose none of them, or was not answered.
    pub fn chosen(&self, name: &str) -> Option<&str> {
        let choice = self.choice(name)?;
        (!self.none.iter().any(|none| none == name)).then_some(choice.value.as_str())
    }

    /// An answer as a plain value: a bool, a label, a number. Null for choosing none, and for one that was not answered.
    pub fn value(&self, name: &str) -> Json {
        match self.get(name) {
            Some(_) if self.none.iter().any(|none| none == name) => Json::Null,
            Some(decision) => decision.value(),
            None => Json::Null,
        }
    }

    /// Whether the request was answered: false for one a limit kept from being asked.
    pub fn answered(&self) -> bool {
        self.answers.values().any(Option::is_some)
    }

    /// The tokens the run used, each counted once: those of its requests, and those its
    /// tools and fits reported. Zero for a sub-call, which has no trajectory of its own.
    pub fn tokens(&self) -> u64 {
        self.trajectory.iter().map(Entry::spent).sum()
    }

    /// How many requests of the run a decision model was asked.
    pub fn asked(&self) -> usize {
        self.trajectory.iter().filter(|entry| entry.asked()).count()
    }

    /// What the tool an output selected returned.
    pub fn observation(&self, name: &str) -> Option<&Json> {
        self.observations.get(name)
    }

    /// What the tool an output selected returned, read as a type: the lines a search found, say.
    /// None when no tool ran for the output, or what it returned is not of that type.
    pub fn observed<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        self.observations.get(name).and_then(|seen| T::deserialize(seen).ok())
    }

    /// Every tool a pointer pointed at in turn, each with what it returned. `finish` is the
    /// last of them when the loop ended there, with null unless a tool of that name ran.
    pub fn steps(&self, name: &str) -> Vec<(String, Json)> {
        let turns = self.observations.get(name).and_then(Json::as_array);
        let pair = |turn: &Json| Some((turn.get(0)?.as_str()?.to_owned(), turn.get(1)?.clone()));
        turns.map(|turns| turns.iter().filter_map(pair).collect()).unwrap_or_default()
    }

    /// The names a pointer pointed at, in turn.
    pub fn pointed(&self, name: &str) -> Vec<String> {
        self.steps(name).into_iter().map(|(step, _)| step).collect()
    }
}

/// As the Python module's prediction reads: the answers by name, then `observations` and
/// `termination_reason`, `arbitration` where the module has an arbiter and the answers selected
/// a tool (`_arbitration`, with as many underscores as it takes, where an answer has the name),
/// and `trajectory` for the call that started the run. An answer that chose
/// none of its options has a null `value`.
/// What something done outside came to ([`crate::Call::acting`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// The request whose tool did it.
    pub index: usize,
    /// The tool. None for a fit.
    pub tool: Option<String>,
    /// What it came to. Where that cannot be written as JSON, `{"unrecorded": why}`: it was
    /// done all the same.
    pub receipt: Json,
}

impl Serialize for Prediction {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (name, answer) in &self.answers {
            let answer = answer.as_ref().map(|decision| Written(decision, self.none.contains(name)));
            map.serialize_entry(name, &answer)?;
        }
        map.serialize_entry("observations", &self.observations)?;
        map.serialize_entry("termination_reason", &self.termination_reason)?;
        // Under its own name, or, where an answer has it, that name after as many underscores
        // as make it free: neither is lost.
        if let Some(arbitration) = &self.arbitration {
            let mut key = "arbitration".to_owned();
            while self.answers.contains_key(&key) {
                key.insert(0, '_');
            }
            map.serialize_entry(&key, arbitration)?;
        }
        if !self.receipts.is_empty() {
            let mut key = "receipts".to_owned();
            while self.answers.contains_key(&key) {
                key.insert(0, '_');
            }
            map.serialize_entry(&key, &self.receipts)?;
        }
        if !self.trajectory.is_empty() {
            map.serialize_entry("trajectory", &self.trajectory)?;
        }
        map.end()
    }
}
