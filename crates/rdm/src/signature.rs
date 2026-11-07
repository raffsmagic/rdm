//! A signature: what a request shows and what it asks.
//!
//! The inputs are the state, each output is a question, and every answer is a decision. The
//! signature says nothing about how the questions are answered: it is plain data, and a tool
//! can build one while a run goes on.

use indexmap::IndexMap;
use serde_json::{Map, json};
use systemone::{Decision, Image, Json, Kind, Question, Request};

use crate::error::{Error, Result, listed};

/// Names a call takes for itself, and keys every prediction has: no field may take them.
pub const RESERVED: [&str; 6] = [
    "signature",
    "state",
    "max_depth",
    "observations",
    "termination_reason",
    "trajectory",
];
/// The option that ends a pointer's loop.
pub const FINISH: &str = "finish";
/// What `finish` is described as, when no tool of that name says otherwise.
pub const FINISH_DESCRIPTION: &str = "Stop: nothing more should be done.";
/// The input that shows what the last tool pointed at returned.
pub const OBSERVATION: &str = "observation";
/// The label of an option that is choosing none of the others.
pub const NONE: &str = "None";
/// The type of an input that is a picture ([`Signature::image`]).
pub const IMAGE: &str = "image";

/// One input of a signature: the name of its type, as a model is told it, and what it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputField {
    /// The type as the model is told it: `str`, `dict`, `list[str]`.
    pub type_name: String,
    /// What the input is. May be empty.
    pub description: String,
}

impl InputField {
    /// An input's type, as the model is told it, and what the input is.
    pub fn new(type_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            description: description.into(),
        }
    }
}

/// One option of a choice: its label, and what choosing it means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opt {
    /// What the model is offered, and what the answer's value is when it is chosen.
    pub label: String,
    /// What the option means. An empty one is sent as null.
    pub description: String,
    /// Whether this is the option of choosing none of the others: it selects no tool, and no fit cuts it.
    pub none: bool,
}

impl Opt {
    /// An option and what it means.
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
            none: false,
        }
    }

    /// The option of choosing none of the others, labelled `None`. An output named after a
    /// tool does not call it when this is chosen, and the answer's plain value is then null:
    /// a tool reads it as an `Option` ([`crate::Call::arg`]).
    ///
    /// Beside bare labels or pairs, which are of another type, it is given by
    /// [`Output::or_none`] or [`Signature::choice_or_none`].
    pub fn none(description: impl Into<String>) -> Self {
        Self {
            label: NONE.into(),
            description: description.into(),
            none: true,
        }
    }
}

impl From<&str> for Opt {
    fn from(label: &str) -> Self {
        Opt::new(label, "")
    }
}

impl From<String> for Opt {
    fn from(label: String) -> Self {
        Opt::new(label, "")
    }
}

impl From<&String> for Opt {
    fn from(label: &String) -> Self {
        Opt::new(label.as_str(), "")
    }
}

impl<L: Into<String>, D: Into<String>> From<(L, D)> for Opt {
    fn from((label, description): (L, D)) -> Self {
        Opt::new(label, description)
    }
}

impl From<&Opt> for Opt {
    fn from(option: &Opt) -> Self {
        option.clone()
    }
}

/// The answers a question can have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Space {
    /// Yes or no, with what each means if it was said.
    Noul {
        /// What a yes means.
        yes: Option<String>,
        /// What a no means.
        no: Option<String>,
    },
    /// One of these options, in the order they are offered.
    Choice(Vec<Opt>),
    /// A level on this rubric, lowest first.
    Score(Vec<String>),
    /// One of the module's tools: the pointer, before a request has filled in its options.
    Tool,
}

/// One output of a signature: a question, the answers it can have, and what a module or a
/// fit has marked on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    /// The question, as the model is asked it.
    pub description: String,
    /// The answers it can have.
    pub space: Space,
    /// Whether the options are the module's tools: set by the module when it fills them in.
    pub points: bool,
    /// The options that stay on offer whatever a fit cuts: `finish`, and the names the module always offers.
    pub always: Vec<String>,
    /// Anything a fit wants to say about a question of its own, so that it knows it again.
    pub marks: IndexMap<String, Json>,
}

impl Output {
    fn of(description: impl Into<String>, space: Space) -> Self {
        Self {
            description: description.into(),
            space,
            points: false,
            always: Vec::new(),
            marks: IndexMap::new(),
        }
    }

    /// A yes or no question.
    pub fn noul(description: impl Into<String>) -> Self {
        Self::of(description, Space::Noul { yes: None, no: None })
    }

    /// A yes or no question that says what each answer means.
    pub fn noul_with(description: impl Into<String>, yes: Option<&str>, no: Option<&str>) -> Self {
        let space = Space::Noul {
            yes: yes.map(str::to_owned),
            no: no.map(str::to_owned),
        };
        Self::of(description, space)
    }

    /// A choice among options: bare labels, `(label, description)` pairs, or [`Opt`]s.
    pub fn choice<O: Into<Opt>>(description: impl Into<String>, options: impl IntoIterator<Item = O>) -> Self {
        Self::of(description, Space::Choice(options.into_iter().map(Into::into).collect()))
    }

    /// A score against levels, lowest first.
    pub fn score<L: Into<String>>(description: impl Into<String>, levels: impl IntoIterator<Item = L>) -> Self {
        Self::of(description, Space::Score(levels.into_iter().map(Into::into).collect()))
    }

    /// The pointer: a choice among the module's tools as they are when the request is made.
    pub fn tool(description: impl Into<String>) -> Self {
        Self::of(description, Space::Tool)
    }

    /// The same question over other options. What was marked on it stays.
    pub fn with_options<O: Into<Opt>>(mut self, options: impl IntoIterator<Item = O>) -> Self {
        self.space = Space::Choice(options.into_iter().map(Into::into).collect());
        self
    }

    /// The same choice with one more option, the last: choosing none of the others
    /// ([`Opt::none`]). Nothing changes for an output that is not a choice.
    ///
    /// ```
    /// use rdm::Output;
    ///
    /// let thermostat = Output::choice("How should the temperature change?", ["warmer", "cooler"]).or_none("The request is not about temperature.");
    /// let labels: Vec<&str> = thermostat.options().unwrap().iter().map(|option| option.label.as_str()).collect();
    /// assert_eq!(labels, ["warmer", "cooler", "None"]);
    /// ```
    pub fn or_none(mut self, description: impl Into<String>) -> Self {
        if let Space::Choice(options) = &mut self.space {
            options.push(Opt::none(description));
        }
        self
    }

    /// The same answers to another question.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Mark the question: a fit marks the questions it asks itself, so that it lets them through.
    pub fn mark(mut self, key: impl Into<String>, value: impl Into<Json>) -> Self {
        self.marks.insert(key.into(), value.into());
        self
    }

    /// What was marked under a key, if anything.
    pub fn marked(&self, key: &str) -> Option<&Json> {
        self.marks.get(key)
    }

    /// The kind of question this is. A pointer whose options are not filled in yet has none.
    pub fn kind(&self) -> Option<Kind> {
        match self.space {
            Space::Noul { .. } => Some(Kind::Noul),
            Space::Choice(_) => Some(Kind::Choice),
            Space::Score(_) => Some(Kind::Score),
            Space::Tool => None,
        }
    }

    /// The options of a choice; nothing for any other output.
    pub fn options(&self) -> Option<&[Opt]> {
        if let Space::Choice(options) = &self.space {
            Some(options)
        } else {
            None
        }
    }

    /// Whether the output points at one of the module's tools: typed as a tool, or offering the tools as options.
    pub fn is_pointer(&self) -> bool {
        self.points || self.space == Space::Tool
    }

    /// Whether an option is always offered: choosing none of them, and the names the module
    /// always offers, `finish` among them.
    pub fn stays(&self, option: &Opt) -> bool {
        option.none || self.always.contains(&option.label)
    }

    /// The answer space as it is sent: nothing for a plain noul, an object of option to
    /// description for a choice (null where there is none), a list of levels for a score.
    pub fn criteria(&self) -> Option<Json> {
        self.question().and_then(|question| question.criteria)
    }

    /// The question as it is put to a decision model. A pointer whose options are not filled in is not one yet.
    pub fn question(&self) -> Option<Question> {
        let instructions = self.description.as_str();
        match &self.space {
            Space::Noul { yes: None, no: None } => Some(Question::noul(instructions)),
            Space::Noul { yes, no } => Some(Question::noul_with(instructions, yes.as_deref(), no.as_deref())),
            Space::Choice(options) => Some(Question::choice(
                instructions,
                options
                    .iter()
                    .map(|option| (option.label.clone(), Some(option.description.clone()))),
            )),
            Space::Score(levels) => Some(Question::score(instructions, levels.iter().cloned())),
            Space::Tool => None,
        }
    }

    /// Why the output cannot be asked as it is, if it cannot: a choice has at least one option,
    /// and a score two to ten levels.
    fn unanswerable(&self) -> Option<String> {
        match &self.space {
            Space::Choice(options) if options.is_empty() => Some("is a choice with no options: a choice has at least one".to_owned()),
            Space::Score(levels) if !(2..=10).contains(&levels.len()) => Some(format!(
                "is a score with {} level(s): a score has two to ten levels, lowest first",
                levels.len()
            )),
            _ => None,
        }
    }

    /// The answer to a choice that has one option: that option, as certain. Nothing for any other output.
    pub fn only(&self) -> Option<Decision> {
        match &self.space {
            Space::Choice(options) if options.len() == 1 => Some(Decision::only(options[0].label.clone())),
            _ => None,
        }
    }

    /// Whether a decision chose the option of choosing none.
    pub fn chose_none(&self, decision: &Decision) -> bool {
        match (&self.space, decision) {
            (Space::Choice(options), Decision::Choice(choice)) => options.iter().any(|option| option.none && option.label == choice.value),
            _ => false,
        }
    }

    /// Whether an answer calls the tool the output is named after: a yes does, and so does
    /// any chosen option but the one of choosing none. A score is a judgment and calls nothing.
    pub fn selects(&self, decision: &Decision) -> bool {
        match decision {
            Decision::Noul(noul) => noul.value,
            Decision::Choice(_) => !self.chose_none(decision),
            Decision::Score(_) => false,
        }
    }

    /// An answer as a plain value: a bool, a label, a number, or null for choosing none.
    pub fn plain(&self, decision: &Decision) -> Json {
        if self.chose_none(decision) { Json::Null } else { decision.value() }
    }
}

/// Instructions, the inputs a request shows, and the questions it asks about them.
///
/// ```
/// use rdm::Signature;
///
/// let triage = Signature::new("Triage a message from a customer.")
///     .input("message")
///     .noul("refund", "Is the customer in `inputs.message` asking for a refund?")
///     .choice("team", "Which team should handle `inputs.message`?", ["billing", "technical support", "sales"])
///     .score("upset", "How upset is the customer in `inputs.message`?", ["calm", "annoyed", "angry"]);
///
/// assert_eq!(triage.input_fields(), "1. `message` (str):");
/// let request = triage.request(serde_json::json!({"message": "I was charged twice."}).as_object().unwrap()).unwrap();
/// assert_eq!(request.questions["team"].criteria, Some(serde_json::json!({"billing": null, "technical support": null, "sales": null})));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signature {
    instructions: String,
    inputs: IndexMap<String, InputField>,
    outputs: IndexMap<String, Output>,
    /// An output an edit named that the signature does not have. Asking it says so.
    unknown: Option<String>,
}

impl Signature {
    /// A signature with its instructions and no fields yet.
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            inputs: IndexMap::new(),
            outputs: IndexMap::new(),
            unknown: None,
        }
    }

    /// The string form: `"text, question -> look"`, `"task, house: dict -> next: Tool"`.
    ///
    /// An input's type is `str` unless it says otherwise. An output is a yes or no (`bool`, or
    /// nothing), a choice (`Literal['a', 'b']`), or the pointer (`Tool`). None has a description
    /// yet: give each one with [`Signature::with_output`] before the signature is asked.
    ///
    /// The instructions are the ones DSPy gives a signature written this way, which name its
    /// fields as they were written. Putting another output later leaves them as they are.
    ///
    /// ```
    /// use rdm::{Output, Signature};
    ///
    /// let ask = Signature::parse("listing -> category").unwrap().with_output(
    ///     "category",
    ///     Output::choice("Which category best matches the product in `inputs.listing`?", ["Furniture", "Kitchen"]),
    /// );
    /// assert_eq!(ask.outputs()["category"].options().unwrap().len(), 2);
    /// assert_eq!(ask.instructions(), "Given the fields `listing`, produce the fields `category`.");
    /// ```
    pub fn parse(text: &str) -> Result<Self> {
        let malformed = || {
            Error::Usage(format!(
                "A signature is written `inputs -> outputs`, each a list of names: got {text:?}."
            ))
        };
        let (inputs, outputs) = text.split_once("->").ok_or_else(malformed)?;
        if outputs.contains("->") {
            return Err(malformed());
        }
        let mut signature = Signature::new("");
        for field in fields(inputs) {
            let (name, type_name) = named(&field).ok_or_else(malformed)?;
            signature = signature.input_typed(name, type_name.unwrap_or("str"));
        }
        for field in fields(outputs) {
            let (name, type_name) = named(&field).ok_or_else(malformed)?;
            let output = match type_name.unwrap_or("bool") {
                "bool" | "str" | "Noul" | "noul" => Output::noul(""),
                "Tool" | "tool" | "dspy.Tool" => Output::tool(""),
                other => match other.strip_prefix("Literal[").and_then(|rest| rest.strip_suffix(']')) {
                    Some(labels) => Output::choice("", fields(labels).iter().map(|label| label.trim_matches(['\'', '"']).to_owned())),
                    None => {
                        return Err(Error::Usage(format!(
                            "Output `{name}` is typed `{other}`; an output is `bool`, `Literal[...]` or `Tool`."
                        )));
                    }
                },
            };
            signature = signature.with_output(name, output);
        }
        if signature.outputs.is_empty() {
            return Err(malformed());
        }
        // As DSPy words them for a signature given as a string: a model is shown the same state from either.
        let ticked = |names: Vec<&String>| names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ");
        signature.instructions = format!(
            "Given the fields {}, produce the fields {}.",
            ticked(signature.inputs.keys().collect()),
            ticked(signature.outputs.keys().collect())
        );
        Ok(signature)
    }

    /// An input of type `str`, with no description.
    pub fn input(self, name: impl Into<String>) -> Self {
        self.input_with(name, "str", "")
    }

    /// An input of another type, as the model is told it: `dict`, `list[str]`, `int`.
    pub fn input_typed(self, name: impl Into<String>, type_name: impl Into<String>) -> Self {
        self.input_with(name, type_name, "")
    }

    /// An input of type `str`, with what it is.
    pub fn input_described(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.input_with(name, "str", description)
    }

    /// An input that is a picture, with what it is: its value is a picture made with
    /// [`crate::image`], or null for none this time.
    ///
    /// A picture is shown to the model as itself, after the rest of the state, and the input
    /// shows where it is: `"<image: window>"` ([`Signature::state`]). Which inputs are
    /// pictures is the signature's to say: a value of any other input is shown as it is,
    /// whatever it looks like.
    ///
    /// ```
    /// use rdm::{Signature, image};
    /// use serde_json::json;
    ///
    /// let look = Signature::new("Look at the window in front.")
    ///     .input("task")
    ///     .image("window", "The window in front, as a picture.")
    ///     .noul("editor", "Is `inputs.window` a text editor?");
    /// let picture = image("image/png", b"\x89PNG");
    /// let request = look.request(json!({"task": "Save the file.", "window": picture}).as_object().unwrap()).unwrap();
    /// assert_eq!(request.state[0]["inputs"], json!({"task": "Save the file.", "window": "<image: window>"}));
    /// assert_eq!(request.state[1], picture);
    /// assert_eq!(request.images(), 1);
    ///
    /// // A value of a picture input that is no picture is the caller's mistake.
    /// let error = look.request(json!({"task": "Save.", "window": "a screenshot"}).as_object().unwrap()).unwrap_err();
    /// assert!(error.is_usage());
    /// ```
    pub fn image(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.input_with(name, IMAGE, description)
    }

    /// An input with its type and what it is.
    pub fn input_with(mut self, name: impl Into<String>, type_name: impl Into<String>, description: impl Into<String>) -> Self {
        self.inputs.insert(name.into(), InputField::new(type_name, description));
        self
    }

    /// A yes or no question. An output named after a tool calls it on a yes.
    pub fn noul(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.with_output(name, Output::noul(description))
    }

    /// A choice among options: bare labels, `(label, description)` pairs, or [`Opt`]s. An
    /// output named after a tool calls it with the option chosen.
    pub fn choice<O: Into<Opt>>(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        options: impl IntoIterator<Item = O>,
    ) -> Self {
        self.with_output(name, Output::choice(description, options))
    }

    /// A choice among options that may come to none of them: the options, and after them the
    /// one of choosing none ([`Opt::none`]), with what choosing it means. An output named after
    /// a tool does not call it when none is chosen.
    ///
    /// ```
    /// use rdm::Signature;
    ///
    /// let home = Signature::new("Carry out a request.").input("request").choice_or_none(
    ///     "thermostat",
    ///     "How should the temperature change?",
    ///     ["warmer", "cooler"],
    ///     "The request is not about temperature.",
    /// );
    /// assert!(home.output("thermostat").unwrap().options().unwrap()[2].none);
    /// ```
    pub fn choice_or_none<O: Into<Opt>>(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        options: impl IntoIterator<Item = O>,
        none: impl Into<String>,
    ) -> Self {
        self.with_output(name, Output::choice(description, options).or_none(none))
    }

    /// A score against levels, lowest first. A score calls nothing, whatever it is named.
    pub fn score<L: Into<String>>(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        levels: impl IntoIterator<Item = L>,
    ) -> Self {
        self.with_output(name, Output::score(description, levels))
    }

    /// The pointer: an output that points at one of the module's tools, turn after turn. A signature may have one.
    pub fn tool(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.with_output(name, Output::tool(description))
    }

    /// The same fields under other instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Put an output: in place of the one of that name, or after the others.
    pub fn with_output(mut self, name: impl Into<String>, output: Output) -> Self {
        self.outputs.insert(name.into(), output);
        self
    }

    /// Put an input: in place of the one of that name, or after the others.
    pub fn with_input(mut self, name: impl Into<String>, field: InputField) -> Self {
        self.inputs.insert(name.into(), field);
        self
    }

    /// The same question over other options, for an output the signature has. What is marked on it stays.
    ///
    /// A name the signature has no output of changes nothing, and is not let pass: the
    /// signature remembers it, and [`Signature::check`], and so every request made of it, is
    /// a usage error that names it.
    pub fn with_options<O: Into<Opt>>(mut self, name: &str, options: impl IntoIterator<Item = O>) -> Self {
        match self.outputs.get_mut(name) {
            Some(output) => output.space = Space::Choice(options.into_iter().map(Into::into).collect()),
            None => self.unknown = self.unknown.take().or_else(|| Some(name.to_owned())),
        }
        self
    }

    /// The signature without a field, input or output. A name it does not have leaves it as it is.
    pub fn without(mut self, name: &str) -> Self {
        self.inputs.shift_remove(name);
        self.outputs.shift_remove(name);
        self
    }

    /// The signature without an output. An input of the same name stays, and a name it has
    /// no output of leaves it as it is.
    pub fn without_output(mut self, name: &str) -> Self {
        self.outputs.shift_remove(name);
        self
    }

    /// What the request is for: said once, above the inputs.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// The inputs, in the order they were declared.
    pub fn inputs(&self) -> &IndexMap<String, InputField> {
        &self.inputs
    }

    /// The outputs, in the order they were declared.
    pub fn outputs(&self) -> &IndexMap<String, Output> {
        &self.outputs
    }

    /// The output of that name, if the signature has one.
    pub fn output(&self, name: &str) -> Option<&Output> {
        self.outputs.get(name)
    }

    /// An output, to change in place.
    pub fn output_mut(&mut self, name: &str) -> Option<&mut Output> {
        self.outputs.get_mut(name)
    }

    /// The outputs that point at one of the module's tools.
    pub fn pointers(&self) -> Vec<&str> {
        let pointing = self.outputs.iter().filter(|(_, output)| output.is_pointer());
        pointing.map(|(name, _)| name.as_str()).collect()
    }

    /// The output that points, if the signature has one.
    pub fn pointer(&self) -> Option<&str> {
        self.pointers().first().copied()
    }

    /// The names of every field that is one a call keeps for itself.
    pub(crate) fn reserved(&self) -> Vec<&str> {
        let names = self.inputs.keys().chain(self.outputs.keys());
        names.map(String::as_str).filter(|name| RESERVED.contains(name)).collect()
    }

    /// The inputs as a model is told of them: one numbered line each.
    ///
    /// ```
    /// use rdm::Signature;
    ///
    /// let operate = Signature::new("Operate a smart home, one step at a time.")
    ///     .input("task")
    ///     .input_typed("house", "dict")
    ///     .input_described("observation", "What the last step did.")
    ///     .tool("next", "What should happen next for `inputs.task`?");
    /// assert_eq!(
    ///     operate.input_fields(),
    ///     "1. `task` (str): \n2. `house` (dict): \n3. `observation` (str): What the last step did."
    /// );
    /// ```
    pub fn input_fields(&self) -> String {
        let lines = self.inputs.iter().enumerate().map(|(position, (name, field))| {
            let number = position + 1;
            format!("{number}. `{name}` ({}): {}", field.type_name, field.description)
        });
        lines.collect::<Vec<_>>().join("\n").trim().to_owned()
    }

    /// The names of the inputs that are pictures ([`Signature::image`]), in the order declared.
    pub fn images(&self) -> Vec<&str> {
        let pictures = self.inputs.iter().filter(|(_, field)| field.type_name == IMAGE);
        pictures.map(|(name, _)| name.as_str()).collect()
    }

    /// The state a request shows: the instructions, the inputs as a model is told of them,
    /// and their values. Values that are not inputs of the signature are left out.
    ///
    /// Where an input that is a picture has one, the state is an array: that object first, and
    /// after it each picture, in the order the inputs are declared. The input shows where its
    /// picture is, as `"<image: name>"`. An input that is a picture and is null has none this
    /// time, and shows null. A signature with no picture to show has the state it always had.
    pub fn state(&self, inputs: &Map<String, Json>) -> Json {
        let pictured =
            |name: &str, value: &Json| self.inputs.get(name).is_some_and(|field| field.type_name == IMAGE) && Image::of(value).is_some();
        let shown: Map<String, Json> = inputs
            .iter()
            .filter(|(name, _)| self.inputs.contains_key(*name))
            .map(|(name, value)| {
                let value = if pictured(name, value) {
                    Json::String(format!("<image: {name}>"))
                } else {
                    value.clone()
                };
                (name.clone(), value)
            })
            .collect();
        let state = json!({"instructions": self.instructions, "input_fields": self.input_fields(), "inputs": shown});
        let pictures: Vec<Json> = self
            .images()
            .into_iter()
            .filter_map(|name| inputs.get(name).filter(|value| pictured(name, value)).cloned())
            .collect();
        if pictures.is_empty() {
            state
        } else {
            Json::Array(std::iter::once(state).chain(pictures).collect())
        }
    }

    /// Whether every input that is a picture holds one, as [`crate::image`] makes it, or null.
    /// Anything else there is a usage error: a module checks this before a request is recorded
    /// or sent.
    pub fn check_images(&self, inputs: &Map<String, Json>) -> Result<()> {
        for name in self.images() {
            match inputs.get(name) {
                None | Some(Json::Null) => {}
                Some(value) if Image::of(value).is_some() => {}
                Some(value) => {
                    let what = match value {
                        Json::Object(_) => "an object that is not one",
                        Json::Array(_) => "a list",
                        Json::String(_) => "a string",
                        Json::Number(_) => "a number",
                        Json::Bool(_) | Json::Null => "true or false",
                    };
                    return Err(Error::Usage(format!(
                        "Input '{name}' is a picture, and was given {what}: a picture is made with `rdm::image(content_type, bytes)`, \
                         of {}, or is null for none.",
                        listed(systemone::CONTENT_TYPES)
                    )));
                }
            }
        }
        Ok(())
    }

    /// Whether the signature can be asked at all: every choice has at least one option, every
    /// score two to ten levels, and no edit named an output the signature does not have. A
    /// module checks every request before it is recorded, so nothing of the kind is sent or counted.
    ///
    /// ```
    /// use rdm::Signature;
    ///
    /// let nothing: [&str; 0] = [];
    /// let ask = Signature::new("Classify.").input("listing").choice("category", "Which category?", nothing);
    /// assert!(ask.check().unwrap_err().is_usage());
    /// ```
    pub fn check(&self) -> Result<()> {
        if let Some(name) = &self.unknown {
            return Err(Error::Usage(format!(
                "`with_options` was given '{name}', which is no output of the signature: it has {}.",
                listed(self.outputs.keys())
            )));
        }
        for (name, output) in &self.outputs {
            if let Some(why) = output.unanswerable() {
                return Err(Error::Usage(format!("Output '{name}' {why}.")));
            }
        }
        Ok(())
    }

    /// The questions, by output, as they are put to a decision model.
    pub fn questions(&self) -> Result<IndexMap<String, Question>> {
        self.check()?;
        let mut questions = IndexMap::new();
        for (name, output) in &self.outputs {
            if output.description.trim().is_empty() {
                return Err(Error::Usage(format!(
                    "Decision output '{name}' requires a description: it is the question asked."
                )));
            }
            let question = output.question().ok_or_else(|| {
                Error::Usage(format!(
                    "Output '{name}' points at a tool, and its options are the tools of the module that asks it: it cannot be asked without one."
                ))
            })?;
            questions.insert(name.clone(), question);
        }
        Ok(questions)
    }

    /// The whole request: one state, and every question at once.
    pub fn request(&self, inputs: &Map<String, Json>) -> Result<Request> {
        self.check_images(inputs)?;
        Ok(Request {
            model: String::new(),
            state: self.state(inputs),
            questions: self.questions()?,
        })
    }
}

/// The fields of one side of a written signature: split at commas that are not inside brackets or quotes.
/// A signature a language model completes is written the same way, and read with this too.
pub(crate) fn fields(text: &str) -> Vec<String> {
    let (mut found, mut field, mut depth, mut quote) = (Vec::new(), String::new(), 0usize, None);
    for character in text.chars() {
        match (quote, character) {
            (Some(open), _) if character == open => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(character),
            (None, '[' | '(') => depth += 1,
            (None, ']' | ')') => depth = depth.saturating_sub(1),
            (None, ',') if depth == 0 => {
                found.push(std::mem::take(&mut field));
                continue;
            }
            (None, _) => {}
        }
        field.push(character);
    }
    found.push(field);
    found
        .into_iter()
        .map(|field| field.trim().to_owned())
        .filter(|field| !field.is_empty())
        .collect()
}

/// A written field as its name and, if it gave one, its type.
pub(crate) fn named(field: &str) -> Option<(&str, Option<&str>)> {
    let (name, type_name) = match field.split_once(':') {
        Some((name, type_name)) => (name.trim(), Some(type_name.trim())),
        None => (field.trim(), None),
    };
    let plain = !name.is_empty() && name.chars().all(|character| character.is_alphanumeric() || character == '_');
    plain.then_some((name, type_name))
}
