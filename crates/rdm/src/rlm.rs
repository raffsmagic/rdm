//! Rlm: a language model in a loop over the sandbox.
//!
//! [`Generate`] is one request: a language model writes, and what it wrote is taken as it is.
//! An RLM is that model in a loop of its own, in a REPL. It writes a cell of Python, the cell
//! is run, it is shown what the cell printed, and it writes the next, until it submits its
//! outputs with `SUBMIT`. What it works on is not in its prompt: each input is a variable in
//! the REPL, and the model is told only what kind of thing it is and how long. It reads a long
//! text by slicing it, in code.
//!
//! ```text
//! a scratch REPL is made                 the functions of the one given, and what was written in it
//! until SUBMIT, or the turns run out:
//!     reasoning, code <- the language model   shown the variables and every step so far
//!     output <- the scratch runs the code     what it printed, or what it failed with
//! SUBMIT(...)                             ends the cell: the outputs, each as its type
//! the checks                              the caller's, and a failure is the step's output
//! ```
//!
//! It is `dspy.RLM` on this crate's own REPL ([`Repl`]) and its own signatures
//! ([`Generate`]): the same fields in each request, the same history of steps, the same
//! `SUBMIT`, `llm_query` and `llm_query_batched`, and one more request to read the outputs off
//! the history when the turns run out.
//!
//! # Who decides
//!
//! The language model chooses nothing that happens. An [`Rlm`] is a tool: it runs when a
//! decision model selects it, in a scratch REPL that is made for the run and gone after it.
//! What it submits is a candidate, put to checks that are the caller's and not the model's
//! ([`Rlm::check`]): a model saying it is done is not evidence that it is. And a function it
//! wrote ([`Rlm::writer`]) is a name on the map, which a decision model points at or does not.
//!
//! # A world where trying changes nothing
//!
//! Code that is tried is run, and the functions it calls do what they do. So what the scratch
//! is given decides what trying costs, and that is the caller's to cover:
//!
//! - stand-ins: the REPL an RLM is given ([`Rlm::repl`]) can hold other functions under the
//!   same names, which record that they were called and change nothing;
//! - a checkpoint ([`Rlm::checkpoint`]): taken when a run starts and put back when it ends,
//!   for a world that is data. In between the real functions run.
//!
//! With neither, the scratch has the functions it was given, and trying is acting.
//!
//! ```
//! use rdm::inputs;
//! use rdm::lm::{Lm, ScriptedLm, Type, input_of};
//! use rdm::rlm::Rlm;
//! use serde_json::json;
//!
//! // A language model that is a function: it looks first, and then it counts.
//! let lm = Lm::of(
//!     ScriptedLm::new(|_system, prompt| {
//!         let (reasoning, code) = match input_of(prompt, "iteration") {
//!             Some("1/5") => ("See how long the text is.", "print(len(text.splitlines()))"),
//!             _ => ("Count the lines that say heat.", "SUBMIT(len([line for line in text.splitlines() if 'heat' in line]))"),
//!         };
//!         json!({"reasoning": reasoning, "code": code}).to_string()
//!     }),
//!     "scripted",
//! );
//!
//! let count = Rlm::new("Count the lines of the text that are about heat.")
//!     .input("text")
//!     .output_typed("count", Type::Int)
//!     .lm(lm)
//!     .max_iters(5);
//! let ran = count.call(inputs! { "text" => "The heat is on.\nThe door is shut.\nNo heat upstairs." }).unwrap();
//! assert_eq!(ran.outputs["count"], 2);
//! assert_eq!((ran.steps.len(), ran.steps[0].output.as_str()), (2, "3"));
//! ```

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Map;
use systemone::{DecisionModel, Json, Usage};

use crate::call::{Ask, Call};
use crate::error::{Error, Result, listed, sorted};
use crate::inputs::Inputs;
use crate::lm::{Generate, Generated, Lm, OutputField, Type};
use crate::module::Rdm;
use crate::parallel::in_parallel;
use crate::prediction::Prediction;
use crate::repl::{ARGUMENTS, Draft, Functions, Repl, def_source, strip_code_fences, takes_of};
use crate::signature::{InputField, Signature};
use crate::tool::Tool;

/// The function that submits, as it is named in the REPL.
pub const SUBMIT: &str = "SUBMIT";

/// The names an RLM uses in its scratch: no input is a variable under one.
const USED: [&str; 4] = [SUBMIT, "llm_query", "llm_query_batched", "print"];

/// What a step's output is when the code printed nothing.
const NO_OUTPUT: &str = "(no output - did you forget to print?)";

/// How much of a variable the model is shown: this many characters, half from each end.
const PREVIEW: usize = 1000;

/// How many prompts of one `llm_query_batched` are asked at once.
const AT_ONCE: usize = 8;

/// What the model is told of where its code runs.
const ENVIRONMENT: &str = "Code runs in a sandboxed Python interpreter. State persists across executions: what one step defines, \
                           the next can use. There are no files, no network and no operating system. These modules can be \
                           imported: re, json, math, datetime, time, collections, itertools, functools, dataclasses, copy, \
                           random, base64, typing and unicodedata; no others. Check what `datetime.strptime` gives before \
                           relying on it.";

/// What a writer is told, after its own instructions.
const WRITING: &str = "What you are to produce is one Python function, which is kept and run later, when it is chosen from a \
                       list by its description. Define it in a step with `def`, call it there to try it, and when it does what is \
                       asked call `SUBMIT(name, description)`: the function of that name is taken as a step last defined it. The \
                       source must be one `def` and nothing beside it, with no decorator, and plain values as defaults. When \
                       the function is run, a parameter is given the request's input of the same name: give it a parameter only \
                       for what the request names, or give the parameter a default. What it returns is what is seen of it.";

/// A check as it is kept: `Err` ends the run, and `Ok(Err(why))` is what the model is told.
type Holds = dyn Fn(&Submitted<'_>) -> Result<std::result::Result<(), String>> + Send + Sync;
type Restores = Box<dyn FnOnce() + Send>;
type Checkpoint = dyn Fn() -> Restores + Send + Sync;
type Watches = dyn Fn(&Step) + Send + Sync;
type Tries = dyn Fn(&Tried<'_>) -> std::result::Result<(), String> + Send + Sync;
type Trying = dyn Fn(&str) + Send + Sync;

#[derive(Clone)]
struct Check {
    /// What the check is called, where it is the caller's.
    name: Option<String>,
    holds: Arc<Holds>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn add(spent: &mut Usage, more: Usage) {
    spent.input_tokens += more.input_tokens;
    spent.output_tokens += more.output_tokens;
}

/// One step of a run: what the model thought, the code it wrote, and what running it came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Step {
    /// What the model said of the step before it wrote it.
    pub reasoning: String,
    /// The code, without the fence a model puts round it.
    pub code: String,
    /// What the code printed, what it failed with, or why what it submitted was not taken.
    pub output: String,
}

impl Step {
    /// The step as the model is shown it in the history of a run.
    fn shown(&self, index: usize, most: usize) -> String {
        let reasoning = if self.reasoning.is_empty() {
            String::new()
        } else {
            format!("Reasoning: {}\n", self.reasoning)
        };
        format!(
            "=== Step {} ===\n{reasoning}Code:\n```python\n{}\n```\n{}",
            index + 1,
            self.code,
            cut(&self.output, most)
        )
    }
}

/// What a run came to: the outputs, each of its declared type, and how it got there.
#[derive(Clone, Debug, PartialEq)]
pub struct Ran {
    /// The outputs, by name, in the order the signature declares them.
    pub outputs: Map<String, Json>,
    /// Every step of the run, in order.
    pub steps: Vec<Step>,
    /// The tokens every request of the run used: the model's own, and those its code asked for.
    pub usage: Usage,
    /// Whether the turns ran out and the outputs were read off the history by one more request.
    pub extracted: bool,
}

impl Ran {
    /// An output, if the signature has one of that name.
    pub fn get(&self, name: &str) -> Option<&Json> {
        self.outputs.get(name)
    }

    /// An output that is text.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.outputs.get(name).and_then(Json::as_str)
    }

    /// An output as a type of the code's own. The outputs were checked against the signature,
    /// so a name it does not have, or a type the output was not declared as, is a usage error.
    pub fn read<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        let value = self.outputs.get(name).ok_or_else(|| {
            Error::Usage(format!(
                "No output is named '{name}': the signature has {}.",
                listed(self.outputs.keys())
            ))
        })?;
        serde_json::from_value(value.clone())
            .map_err(|error| Error::Usage(format!("The output '{name}' is read as a type it is not: {error}.")))
    }

    /// The tokens in and out together.
    pub fn tokens(&self) -> u64 {
        self.usage.total()
    }
}

/// Where a run's code is run: the scratch, the request it is run for, and its inputs.
#[derive(Clone, Copy)]
struct Scratch<'a> {
    repl: &'a Repl,
    call: Option<&'a Call>,
    inputs: &'a Map<String, Json>,
}

impl Scratch<'_> {
    /// Run code in the scratch, with the inputs as variables.
    fn run(&self, code: &str) -> Result<String> {
        self.repl.execute_for(self.call, code, self.inputs)
    }

    /// Run a step's code in the scratch: what it printed, and why it failed where it did.
    fn step(&self, code: &str) -> Result<(String, Option<String>)> {
        self.repl.execute_keeping_for(self.call, code, self.inputs)
    }

    /// Call a function of the scratch as the map would: its arguments by name, and what it
    /// returns printed.
    fn call(&self, name: &str, arguments: Map<String, Json>) -> Result<String> {
        if arguments.is_empty() {
            return self.run(&format!("print({name}())"));
        }
        let mut variables = self.inputs.clone();
        variables.insert(ARGUMENTS.to_owned(), Json::Object(arguments));
        self.repl
            .execute_for(self.call, &format!("print({name}(**{ARGUMENTS}))"), &variables)
    }
}

/// What a model submitted, as a check is given it: the outputs, each already of its type, and
/// the scratch they were made in.
pub struct Submitted<'a> {
    outputs: &'a Map<String, Json>,
    steps: &'a [Step],
    scratch: Scratch<'a>,
}

impl std::fmt::Debug for Submitted<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submitted")
            .field("outputs", self.outputs)
            .field("steps", &self.steps.len())
            .finish_non_exhaustive()
    }
}

impl Submitted<'_> {
    /// The outputs, by name.
    pub fn outputs(&self) -> &Map<String, Json> {
        self.outputs
    }

    /// An output, if the signature has one of that name.
    pub fn get(&self, name: &str) -> Option<&Json> {
        self.outputs.get(name)
    }

    /// An output that is text.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.outputs.get(name).and_then(Json::as_str)
    }

    /// The inputs of the run, by name: what the code has as variables.
    pub fn inputs(&self) -> &Map<String, Json> {
        self.scratch.inputs
    }

    /// The steps of the run so far, the one that submitted last.
    pub fn steps(&self) -> &[Step] {
        self.steps
    }

    /// Run code in the scratch the outputs were made in, with the inputs as variables, and
    /// return what it printed. What the model left there is there: a check can call it.
    pub fn run(&self, code: &str) -> std::result::Result<String, String> {
        self.scratch.run(code).map_err(|error| error.to_string())
    }
}

/// A function a writer submitted, as a check is given it: defined alone, in a REPL made for
/// the trial, and tried there if it could be.
pub struct Tried<'a> {
    /// The function's name.
    pub name: &'a str,
    /// The sentence a menu says of it.
    pub description: &'a str,
    /// Its source: one `def`.
    pub code: &'a str,
    /// What it returned when it was tried, printed. None where it was not tried: it takes a
    /// parameter that no input of the request names.
    pub printed: Option<&'a str>,
    scratch: Scratch<'a>,
}

impl std::fmt::Debug for Tried<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tried")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("code", &self.code)
            .field("printed", &self.printed)
            .finish_non_exhaustive()
    }
}

impl Tried<'_> {
    /// The inputs of the run, by name.
    pub fn inputs(&self) -> &Map<String, Json> {
        self.scratch.inputs
    }

    /// Run code in the REPL the function was tried in, and return what it printed: to call
    /// the function again, on what the check sets up for it.
    pub fn run(&self, code: &str) -> std::result::Result<String, String> {
        self.scratch.run(code).map_err(|error| error.to_string())
    }
}

/// What is put back when it is dropped: however a run ends, it ends here.
struct Restore(Option<Restores>);

impl Drop for Restore {
    fn drop(&mut self) {
        if let Some(restore) = self.0.take() {
            restore();
        }
    }
}

/// How many times the code of one run may ask a language model.
struct Budget {
    limit: usize,
    used: Mutex<usize>,
}

impl Budget {
    fn reserve(&self, asks: usize) -> Result<()> {
        let mut used = lock(&self.used);
        if *used + asks > self.limit {
            return Err(Error::Tool(format!(
                "LLM call limit exceeded: {} + {asks} > {}. Use Python code for aggregation instead of making more LLM calls.",
                *used, self.limit
            )));
        }
        *used += asks;
        Ok(())
    }
}

/// A language model in a loop over a REPL: a signature it completes by writing code, running
/// it, and submitting what the code came to.
///
/// It is built as a [`Generate`] is: instructions, inputs, typed outputs, and the language
/// model that works. Beside them it has the REPL its code runs in ([`Rlm::repl`]), its limits,
/// and the caller's checks. It is a tool when it is asked to be ([`Rlm::tool`]), and a writer
/// of functions ([`Rlm::writer`]).
///
/// Every run has a scratch REPL of its own: the functions of the one given, the functions
/// written in it so far, and nothing a run before left. The interpreter is not kept between
/// runs, since one that is has the first run's variables in it for the second.
#[derive(Clone)]
pub struct Rlm {
    lm: Option<Lm>,
    sub_lm: Option<Lm>,
    instructions: String,
    inputs: IndexMap<String, InputField>,
    outputs: IndexMap<String, OutputField>,
    /// The outputs `SUBMIT` does not take, since they are read off the steps: a writer's `code`.
    optional: Vec<String>,
    repl: Option<Repl>,
    max_iters: usize,
    max_llm_calls: usize,
    max_output_chars: usize,
    /// The paragraph after what is available, where it is not the default ([`Rlm::guidance`]).
    guidance: Option<String>,
    /// The numbered rules, where they are not the defaults ([`Rlm::rules`]).
    rules: Option<Vec<String>>,
    checks: Vec<Check>,
    checkpoint: Option<Arc<Checkpoint>>,
    watch: Option<Arc<Watches>>,
}

impl Rlm {
    /// An RLM with its instructions and no fields yet: what the task is, in the words a
    /// docstring would say it.
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            lm: None,
            sub_lm: None,
            instructions: instructions.into(),
            inputs: IndexMap::new(),
            outputs: IndexMap::new(),
            optional: Vec::new(),
            repl: None,
            max_iters: 20,
            max_llm_calls: 50,
            max_output_chars: 10_000,
            guidance: None,
            rules: None,
            checks: Vec::new(),
            checkpoint: None,
            watch: None,
        }
    }

    /// The string form, as [`Generate::parse`] reads it: `"text, question -> answer"`,
    /// `"text -> count: int"`.
    pub fn parse(text: &str) -> Result<Self> {
        let generate = Generate::parse(text)?;
        let mut rlm = Rlm::new(generate.instructions());
        rlm.inputs = generate.inputs().clone();
        rlm.outputs = generate.outputs().clone();
        Ok(rlm)
    }

    /// The language model that works: it writes each step.
    pub fn lm(mut self, lm: Lm) -> Self {
        self.lm = Some(lm);
        self
    }

    /// The language model the code asks with `llm_query`, where it is another: a cheaper one,
    /// as a rule. Unless this says otherwise it is the one that works.
    pub fn sub_lm(mut self, lm: Lm) -> Self {
        self.sub_lm = Some(lm);
        self
    }

    /// An input that is text.
    pub fn input(self, name: impl Into<String>) -> Self {
        self.input_with(name, "str", "")
    }

    /// An input of a named type: `dict`, `list[str]`.
    pub fn input_typed(self, name: impl Into<String>, type_name: impl Into<String>) -> Self {
        self.input_with(name, type_name, "")
    }

    /// An input that is text, and what it is.
    pub fn input_described(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.input_with(name, "str", description)
    }

    /// An input: its name, the name of its type, and what it is.
    pub fn input_with(mut self, name: impl Into<String>, type_name: impl Into<String>, description: impl Into<String>) -> Self {
        self.inputs.insert(name.into(), InputField::new(type_name, description));
        self
    }

    /// An output that is text.
    pub fn output(self, name: impl Into<String>) -> Self {
        self.output_with(name, Type::Str, "")
    }

    /// An output of a type.
    pub fn output_typed(self, name: impl Into<String>, kind: Type) -> Self {
        self.output_with(name, kind, "")
    }

    /// An output that is text, and what it is.
    pub fn output_described(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.output_with(name, Type::Str, description)
    }

    /// An output: its name, its type, and what it is.
    pub fn output_with(mut self, name: impl Into<String>, kind: Type, description: impl Into<String>) -> Self {
        self.outputs.insert(name.into(), OutputField::new(kind, description));
        self
    }

    /// The REPL the code runs in: its functions are the ones code may call, and the functions
    /// written in it so far can be called as well. A run works in a scratch made from it, so
    /// nothing a run defines is left in it. With none, the code has plain Python.
    ///
    /// What these functions do, they do when code that is being tried calls them. Give a
    /// REPL whose acting functions are stand-ins, or a [`Rlm::checkpoint`], where trying
    /// must not be acting.
    pub fn repl(mut self, repl: &Repl) -> Self {
        self.repl = Some(repl.clone());
        self
    }

    /// How many steps a run may take before its outputs are read off the history. 20.
    pub fn max_iters(mut self, steps: usize) -> Self {
        self.max_iters = steps;
        self
    }

    /// How many times the code of one run may ask a language model, with `llm_query` and
    /// `llm_query_batched`. 50. With 0 the code is not given either.
    pub fn max_llm_calls(mut self, asks: usize) -> Self {
        self.max_llm_calls = asks;
        self
    }

    /// How much of a step's output the model is shown: this many characters, half from each
    /// end, with how many were left out between. 10,000.
    pub fn max_output_chars(mut self, characters: usize) -> Self {
        self.max_output_chars = characters;
        self
    }

    /// The paragraph the model reads after what it has available, in place of the default
    /// ("IMPORTANT: This is ITERATIVE …"). With [`Rlm::rules`], what an optimizer rewrites of the
    /// instructions, as DSPy keeps the action's instructions for one; the task, the fields, the
    /// environment and the functions are still written from the RLM as it is.
    ///
    /// ```
    /// use rdm::rlm::Rlm;
    ///
    /// let rlm = Rlm::new("Count the words.").guidance("Look, then submit.").rules(["PRINT LITTLE"]);
    /// assert_eq!(rlm.instructions(), "Count the words.");
    /// ```
    pub fn guidance(mut self, text: impl Into<String>) -> Self {
        self.guidance = Some(text.into());
        self
    }

    /// The numbered rules of the instructions, in place of the defaults (`EXPLORE FIRST`, …): see
    /// [`Rlm::guidance`].
    pub fn rules(mut self, rules: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.rules = Some(rules.into_iter().map(Into::into).collect());
        self
    }

    /// A check on what is submitted: the caller's, and not the model's. It is given the
    /// outputs, each already of its type, and the scratch they were made in.
    ///
    /// What a check that fails says is the step's output, under the check's name, and the run
    /// goes on: the model reads it and writes again. Outputs read off the history when the
    /// turns ran out are checked too, and failing then is the run's failure.
    ///
    /// ```
    /// use rdm::inputs;
    /// use rdm::lm::{Lm, ScriptedLm, Type, input_of};
    /// use rdm::rlm::Rlm;
    /// use serde_json::json;
    ///
    /// // It guesses, is told why not, and then counts.
    /// let lm = Lm::of(
    ///     ScriptedLm::new(|_system, prompt| {
    ///         let refused = input_of(prompt, "repl_history").is_some_and(|history| history.contains("[Not accepted]"));
    ///         json!({"reasoning": "", "code": if refused { "SUBMIT(len(words))" } else { "SUBMIT(7)" }}).to_string()
    ///     }),
    ///     "scripted",
    /// );
    /// let count = Rlm::parse("words: list[str] -> count: int").unwrap().lm(lm).check("counted", |submitted| {
    ///     let said = submitted.run("print(len(words))")?;
    ///     if submitted.get("count").is_some_and(|count| count.to_string() == said) {
    ///         Ok(())
    ///     } else {
    ///         Err(format!("there are {said} words"))
    ///     }
    /// });
    /// let ran = count.call(inputs! { "words" => ["a", "b", "c"] }).unwrap();
    /// assert_eq!(ran.outputs["count"], 3);
    /// assert_eq!(ran.steps[0].output, "[Not accepted] counted: there are 3 words");
    /// ```
    pub fn check<F>(mut self, name: impl Into<String>, holds: F) -> Self
    where
        F: Fn(&Submitted<'_>) -> std::result::Result<(), String> + Send + Sync + 'static,
    {
        self.checks.push(Check {
            name: Some(name.into()),
            holds: Arc::new(move |submitted| Ok(holds(submitted))),
        });
        self
    }

    /// A checkpoint: `take` is called when a run starts, and what it returns is called when
    /// the run ends, however it ends. It is how a world that is data is put back as it was,
    /// so that code tried against the real functions leaves nothing changed.
    ///
    /// ```
    /// use rdm::lm::{Lm, ScriptedLm};
    /// use rdm::repl::Repl;
    /// use rdm::rlm::Rlm;
    /// use rdm::{Shared, inputs};
    /// use serde_json::json;
    ///
    /// let heat = Shared::new(21);
    /// let (down, now, kept) = (heat.clone(), heat.clone(), heat.clone());
    /// let repl = Repl::new()
    ///     .function("cooler", "Make it one degree cooler.", move |_| Ok(down.update(|heat| *heat -= 1)))
    ///     .function("heat", "How warm it is, in degrees.", move |_| Ok(now.get()));
    ///
    /// let lm = Lm::of(ScriptedLm::new(|_, _| json!({"reasoning": "", "code": "cooler()\ncooler()\nSUBMIT(heat())"}).to_string()), "scripted");
    /// let tried = Rlm::parse("task -> came_to: int").unwrap().lm(lm).repl(&repl).checkpoint(move || {
    ///     let before = kept.get();
    ///     let kept = kept.clone();
    ///     move || {
    ///         kept.set(before);
    ///     }
    /// });
    /// // It was 19 while the code ran, and it is 21 again.
    /// assert_eq!(tried.call(inputs! { "task" => "Try two degrees cooler." }).unwrap().outputs["came_to"], 19);
    /// assert_eq!(heat.get(), 21);
    /// ```
    pub fn checkpoint<F, R>(mut self, take: F) -> Self
    where
        F: Fn() -> R + Send + Sync + 'static,
        R: FnOnce() + Send + 'static,
    {
        self.checkpoint = Some(Arc::new(move || Box::new(take())));
        self
    }

    /// Something to be told each step as it ends: for a trace, or for a line on a screen.
    /// The steps are what a run returns as well ([`Ran::steps`]); a tool's observation is its
    /// outputs alone, and this is how the steps of a run that was a tool are kept.
    pub fn watch(mut self, watch: impl Fn(&Step) + Send + Sync + 'static) -> Self {
        self.watch = Some(Arc::new(watch));
        self
    }

    /// The instructions.
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// The inputs, by name.
    pub fn inputs(&self) -> &IndexMap<String, InputField> {
        &self.inputs
    }

    /// The outputs, by name.
    pub fn outputs(&self) -> &IndexMap<String, OutputField> {
        &self.outputs
    }

    /// Run for these inputs: the signature's, every one and no other.
    ///
    /// A language model that cannot be asked ends the run ([`Error::Model`]). One that took
    /// every turn and whose steps did not come to outputs that are accepted has failed as a
    /// tool fails ([`Error::Tool`]).
    pub fn call(&self, inputs: Inputs) -> Result<Ran> {
        let written = self.repl.as_ref().map(Repl::written).unwrap_or_default();
        self.run(None, &inputs.snapshot()?, written)
    }

    /// As [`Rlm::call`], from inside a tool: the tokens it takes are reported through the
    /// call, so they count against the run's limit, and a function of the REPL that asks the
    /// module again has the request to ask it for.
    pub fn call_in(&self, call: &Call, inputs: Inputs) -> Result<Ran> {
        let written = self.repl.as_ref().map(Repl::written).unwrap_or_default();
        self.run(Some(call), &inputs.snapshot()?, written)
    }

    /// As [`Rlm::call_in`], with the inputs taken from the request the tool was selected by:
    /// each by name from the request's answers, then from its inputs as the request showed them.
    pub fn call_from(&self, call: &Call) -> Result<Ran> {
        let written = self.repl.as_ref().map(Repl::written).unwrap_or_default();
        self.run(Some(call), &self.shown_by(call)?, written)
    }

    /// The RLM as a tool. Selected, its inputs are taken by name from the request's answers,
    /// then from its inputs, as a [`Generate`]'s are. Its observation is its outputs, an
    /// object by name: the steps are not in it ([`Rlm::watch`]).
    pub fn tool(&self, name: impl Into<String>, description: impl Into<String>) -> Tool {
        let rlm = self.clone();
        Tool::new(name, description, move |call| Ok(Json::Object(rlm.call_from(call)?.outputs)))
    }

    /// The RLM as a writer of functions for a REPL: what it submits is one function, which
    /// is defined in `repl` once it has been tried and has passed the caller's checks. Its
    /// outputs are a function's `name`, `description` and `code`, whatever they were, and it
    /// is told how a function is submitted.
    ///
    /// It is written and tried in the RLM's own REPL ([`Rlm::repl`]): `repl` itself, unless
    /// another was given, which is how a writer has stand-ins for the functions that act.
    pub fn writer(mut self, repl: &Repl) -> Writer {
        self.outputs = IndexMap::from([
            ("name".to_owned(), OutputField::new(Type::Str, "The function's name.")),
            (
                "description".to_owned(),
                OutputField::new(Type::Str, "One sentence for a menu: what choosing the function does."),
            ),
            (
                "code".to_owned(),
                OutputField::new(Type::Str, "The function's source: one `def` of that name and nothing beside it."),
            ),
        ]);
        self.optional = vec!["code".to_owned()];
        self.instructions = match self.instructions.trim() {
            "" => WRITING.to_owned(),
            own => format!("{own}\n\n{WRITING}"),
        };
        if self.repl.is_none() {
            self.repl = Some(repl.clone());
        }
        Writer {
            rlm: self,
            repl: repl.clone(),
            checks: Vec::new(),
            trying: None,
        }
    }

    /// The inputs a request gives: each by name from its answers, then from what it showed.
    fn shown_by(&self, call: &Call) -> Result<Map<String, Json>> {
        let (mut shown, mut missing) = (Map::new(), Vec::new());
        for field in self.inputs.keys() {
            match call.answer_value(field).or_else(|| call.shown().get(field).cloned()) {
                Some(value) => drop(shown.insert(field.clone(), value)),
                None => missing.push(field.clone()),
            }
        }
        if missing.is_empty() {
            return Ok(shown);
        }
        let who = call.tool().map_or("The RLM".to_owned(), |name| format!("Tool '{name}'"));
        Err(Error::Usage(format!(
            "{who} takes {}, which name no answer or input of the request. An input of a signature is taken by name from {}.",
            listed(&missing),
            sorted(call.names())
        )))
    }

    /// Whether the signature is one to run, for these inputs, in this REPL.
    fn fits(&self, shown: &Map<String, Json>, env: &Repl, written: &Functions) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(Error::Usage("An RLM has at least one output.".to_owned()));
        }
        if let Some(name) = self.outputs.keys().find(|name| self.inputs.contains_key(*name)) {
            return Err(Error::Usage(format!("'{name}' is both an input and an output of the signature.")));
        }
        if shown.len() != self.inputs.len() || shown.keys().any(|name| !self.inputs.contains_key(name)) {
            return Err(Error::Usage(format!(
                "The call has inputs {}; its signature takes {}.",
                sorted(shown.keys()),
                sorted(self.inputs.keys())
            )));
        }
        // Each input is a variable where the code runs, under its name.
        let functions = env.functions();
        let taken = |name: &&String| USED.contains(&name.as_str()) || functions.contains(name) || written.contains_key(*name);
        if let Some(name) = self.inputs.keys().find(taken) {
            return Err(Error::Usage(format!(
                "The input '{name}' has the name of a function in the REPL, where each input is a variable: give it another name."
            )));
        }
        Ok(())
    }

    /// One run, for a request where there is one. `written` are the functions the scratch
    /// starts with, beside those of the host.
    fn run(&self, call: Option<&Call>, shown: &Map<String, Json>, written: Functions) -> Result<Ran> {
        let lm = self
            .lm
            .as_ref()
            .ok_or_else(|| Error::Usage("The RLM has no language model to write its steps: give it one with `lm(...)`.".to_owned()))?;
        let env = self.repl.clone().unwrap_or_default();
        self.fits(shown, &env, &written)?;
        let act = self.action(lm, &env, &written);
        let (gave, asked_for) = (Arc::new(Mutex::new(None)), Arc::new(Mutex::new(Usage::default())));
        let repl = self.scratch(lm, &env, written, &gave, &asked_for);
        let scratch = Scratch {
            repl: &repl,
            call,
            inputs: shown,
        };
        let _restore = Restore(self.checkpoint.as_ref().map(|take| take()));
        let info = variables_info(&self.inputs, shown);
        let mut ran = Ran {
            outputs: Map::new(),
            steps: Vec::new(),
            usage: Usage::default(),
            extracted: false,
        };
        // What the code asked language models for is the run's, and the request's.
        let spend = |ran: &mut Ran| {
            let asked = std::mem::take(&mut *lock(&asked_for));
            if let Some(call) = call {
                call.spend(asked.total());
            }
            add(&mut ran.usage, asked);
        };

        for iteration in 0..self.max_iters {
            // A branch arbitration has stopped takes no more steps.
            if let Some(call) = call {
                call.go_on()?;
            }
            let mut inputs = Inputs::new();
            inputs.set("variables_info", info.as_str());
            inputs.set("repl_history", history(&ran.steps, self.max_output_chars));
            // The last step is told it is the last, as Alex Zhang's RLM tells it: an answer submitted is
            // better than one read off the history after the steps have run out.
            inputs.set(
                "iteration",
                match iteration + 1 == self.max_iters {
                    true => format!(
                        "{}/{} - this is the last step: SUBMIT your best answer now",
                        iteration + 1,
                        self.max_iters
                    ),
                    false => format!("{}/{}", iteration + 1, self.max_iters),
                },
            );
            let action = ask(&act, call, inputs, &mut ran.usage)?;
            let reasoning = action.text("reasoning").unwrap_or_default().trim().to_owned();
            let wrote = action.text("code").unwrap_or_default();
            lock(&gave).take();
            let (code, printed) = match strip_code_fences(wrote) {
                Ok(code) => {
                    let printed = scratch.step(&code);
                    (code, printed)
                }
                Err(why) => (wrote.to_owned(), Err(Error::Tool(why))),
            };
            spend(&mut ran);
            let printed = match printed {
                Ok((printed, None)) => printed,
                // Code that fails has failed, and the model reads what it printed before it did, then
                // why, as Alex Zhang's RLM shows it: a page read before the error is not read again.
                Ok((printed, Some(why))) if printed.is_empty() => format!("[Error] {why}"),
                Ok((printed, Some(why))) => format!("{printed}\n[Error] {why}"),
                Err(Error::Tool(why)) => format!("[Error] {why}"),
                // Anything else ends the run.
                Err(ended) => return Err(ended),
            };
            ran.steps.push(Step {
                reasoning,
                code,
                output: String::new(),
            });
            let submitted = lock(&gave).take();
            let accepted = match submitted {
                Some(submitted) => Some(self.accept(submitted, &ran.steps, scratch)?),
                None => None,
            };
            spend(&mut ran);
            let (output, outputs) = match accepted {
                None if printed.is_empty() => (NO_OUTPUT.to_owned(), None),
                None => (printed, None),
                Some(Ok(outputs)) => (format!("FINAL: {}", Json::Object(outputs.clone())), Some(outputs)),
                Some(Err(why)) if printed.is_empty() => (why, None),
                Some(Err(why)) => (format!("{printed}\n{why}"), None),
            };
            if let Some(step) = ran.steps.last_mut() {
                step.output = output;
                if let Some(watch) = &self.watch {
                    watch(step);
                }
            }
            if let Some(outputs) = outputs {
                ran.outputs = outputs;
                return Ok(ran);
            }
        }

        // The turns ran out: the outputs are read off the history, by one more request.
        let mut inputs = Inputs::new();
        inputs.set("variables_info", info.as_str());
        inputs.set("repl_history", history(&ran.steps, self.max_output_chars));
        let extracted = ask(&self.extract(lm), call, inputs, &mut ran.usage)?;
        let accepted = self.accept(extracted.outputs, &ran.steps, scratch)?;
        spend(&mut ran);
        match accepted {
            Ok(outputs) => {
                ran.outputs = outputs;
                ran.extracted = true;
                Ok(ran)
            }
            Err(why) => Err(Error::Tool(format!(
                "The language model did not submit outputs that were accepted in {} steps, and nor were the outputs read off \
                 its steps. {why}",
                self.max_iters
            ))),
        }
    }

    /// What was submitted, each output as its type and past every check, or why it is not taken.
    fn accept(
        &self,
        gave: Map<String, Json>,
        steps: &[Step],
        scratch: Scratch<'_>,
    ) -> Result<std::result::Result<Map<String, Json>, String>> {
        let (mut outputs, mut mistakes) = (Map::new(), Vec::new());
        for (name, field) in &self.outputs {
            match gave.get(name).cloned() {
                None | Some(Json::Null) if self.optional.contains(name) => drop(outputs.insert(name.clone(), Json::Null)),
                None => mistakes.push(format!("`{name}` is missing.")),
                Some(value) => match field.kind.fit(value) {
                    Ok(value) => drop(outputs.insert(name.clone(), value)),
                    Err(problem) => mistakes.push(format!("`{name}` is declared `{}`, and {problem}.", field.kind)),
                },
            }
        }
        if !mistakes.is_empty() {
            return Ok(Err(format!(
                "[Error] What was submitted is not the outputs. {}",
                mistakes.join(" ")
            )));
        }
        let submitted = Submitted {
            outputs: &outputs,
            steps,
            scratch,
        };
        for check in &self.checks {
            if let Err(why) = (check.holds)(&submitted)? {
                return Ok(Err(match &check.name {
                    Some(name) => format!("[Not accepted] {name}: {why}"),
                    None => format!("[Not accepted] {why}"),
                }));
            }
        }
        Ok(Ok(outputs))
    }

    /// The scratch a run works in: the functions of the REPL given, the functions written so
    /// far, and the three that are the RLM's own.
    fn scratch(
        &self,
        lm: &Lm,
        env: &Repl,
        written: Functions,
        gave: &Arc<Mutex<Option<Map<String, Json>>>>,
        asked_for: &Arc<Mutex<Usage>>,
    ) -> Repl {
        let names: Vec<String> = self.outputs.keys().cloned().collect();
        let gives = gave.clone();
        let scratch = env.spawn("scratch").with_written(written).ending(
            self.submitting(),
            "Submit the final outputs. The code ends here.",
            move |args| {
                let outputs = names.iter().filter_map(|name| Some((name.clone(), args.value(name)?.clone())));
                *lock(&gives) = Some(outputs.collect());
                Ok(())
            },
        );
        if self.max_llm_calls == 0 {
            return scratch;
        }
        let sub = self.sub_lm.clone().unwrap_or_else(|| lm.clone());
        let budget = Arc::new(Budget {
            limit: self.max_llm_calls,
            used: Mutex::new(0),
        });
        let (asks, counts, takes) = (sub.clone(), budget.clone(), asked_for.clone());
        let takes_all = asked_for.clone();
        // Each is given the request the code is run for, where there is one: what the language
        // model wrote is the run's `wrote` event, as a tool's is.
        scratch
            .asking("llm_query(prompt)", "Query a sub-LLM for semantic analysis.", move |args, call| {
                let prompt: String = args.get("prompt")?;
                if let Some(call) = call {
                    call.go_on()?;
                }
                counts.reserve(1)?;
                let asks = match call {
                    Some(call) => asks.under(call),
                    None => asks.clone(),
                };
                let (said, wrote) = asks.completing("", &prompt);
                if let (Some(call), Some(wrote)) = (call, wrote) {
                    call.wrote(wrote)?;
                }
                let said = said.map_err(queried)?;
                add(&mut lock(&takes), said.usage);
                Ok(said.text)
            })
            .asking(
                "llm_query_batched(prompts)",
                "Query multiple prompts concurrently.",
                move |args, call| {
                    let prompts: Vec<String> = args.get("prompts")?;
                    budget.reserve(prompts.len())?;
                    let asked = prompts.into_iter().map(|prompt| {
                        let lm = match call {
                            Some(call) => sub.under(call),
                            None => sub.clone(),
                        };
                        move || Ok(lm.completing("", &prompt))
                    });
                    let mut said = Vec::new();
                    // Kept in the order the prompts were given, whatever order they came back in.
                    for answered in in_parallel(AT_ONCE, asked.collect()) {
                        let (completed, wrote) = answered?;
                        if let (Some(call), Some(wrote)) = (call, wrote) {
                            call.wrote(wrote)?;
                        }
                        match completed.map_err(queried) {
                            Ok(completion) => {
                                add(&mut lock(&takes_all), completion.usage);
                                said.push(completion.text);
                            }
                            // One prompt that failed is told in its place, and the others stand.
                            Err(Error::Tool(error)) => said.push(format!("[ERROR] {error}")),
                            Err(ended) => return Err(ended),
                        }
                    }
                    Ok(said)
                },
            )
    }

    /// `SUBMIT` as it is declared: the outputs it takes, which are all but those read off the steps.
    fn submitting(&self) -> String {
        let takes = self.outputs.keys().filter(|name| !self.optional.contains(name));
        format!("{SUBMIT}({})", takes.cloned().collect::<Vec<_>>().join(", "))
    }

    /// The request for one step: the variables, the steps so far and which step this is in,
    /// and what the model thinks and the code it writes out.
    fn action(&self, lm: &Lm, env: &Repl, written: &Functions) -> Generate {
        Generate::new(self.instructions_for(env, written))
            .input_with("variables_info", "str", "Metadata about the variables available in the REPL")
            .input_with("repl_history", "str", "Previous REPL code executions and their outputs")
            .input_with("iteration", "str", "Current iteration number (1-indexed) out of max_iters")
            .output_described(
                "reasoning",
                "Think step-by-step: what do you know? What remains? Plan your next action.",
            )
            .output_described("code", "Python code to execute.")
            .lm(lm.clone())
    }

    /// The request that reads the outputs off the history, when the turns ran out.
    fn extract(&self, lm: &Lm) -> Generate {
        let task = match self.instructions.trim() {
            "" => String::new(),
            task => format!("The trajectory was generated with the following objective: \n{task}\n\n"),
        };
        let mut extract = Generate::new(format!(
            "{task}Based on the REPL trajectory, extract the final outputs now.\n\nReview your trajectory to see what \
             information you gathered and what values you computed, then provide the final outputs."
        ))
        .input_with("variables_info", "str", "Metadata about the variables available in the REPL")
        .input_with("repl_history", "str", "Your REPL interactions so far");
        for (name, field) in &self.outputs {
            extract = extract.output_with(name, field.kind.clone(), field.description.clone());
        }
        extract.lm(lm.clone())
    }

    /// What the model is told: the task, what a run is, what it has, and how to go about it.
    /// It is `dspy.RLM`'s, with what this REPL is said in place of what Deno is.
    fn instructions_for(&self, env: &Repl, written: &Functions) -> String {
        let ticked = |names: Vec<&String>| names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ");
        let inputs = ticked(self.inputs.keys().collect());
        let outputs = self.outputs.iter().map(|(name, field)| {
            let means = if field.description.is_empty() {
                String::new()
            } else {
                format!(": {}", field.description)
            };
            format!("- `{name}` ({}){means}", field.kind)
        });
        let outputs = outputs.collect::<Vec<_>>().join("\n");
        let asks = self.max_llm_calls > 0;
        // Where the decision model is there to ask (`deciding`), judgments are its, and the model is told so.
        let judges = env
            .describe()
            .lines()
            .any(|line| line.starts_with("decide(") || line.starts_with("choose("));
        let mut text = match self.instructions.trim() {
            "" => String::new(),
            task => format!("{task}\n\n"),
        };
        text.push_str(&format!(
            "You are tasked with producing the following outputs given the inputs {inputs}:\n{outputs}\n\n\
             You have access to a Python REPL environment. Write Python code and it will be executed. You will see the \
             output, then write more code based on what you learned. This is an iterative process.\n\n\
             Execution environment:\n{ENVIRONMENT}\n\n\
             Available:\n- Variables: {inputs} (your input data)\n"
        ));
        if asks {
            text.push_str(
                "- `llm_query(prompt)` - query a sub-LLM for semantic analysis\n\
                 - `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)\n",
            );
        }
        let guidance = self.guidance.as_deref().unwrap_or(
            "IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide \
             what to do next. Do NOT try to solve everything in one step.",
        );
        text.push_str(&format!(
            "- `print()` - ALWAYS print to see results\n- `{}` - submit final output when done\n\n{guidance}\n\n",
            self.submitting()
        ));
        let mut rules = vec![
            "EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.",
            "ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.",
            "VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.",
        ];
        if judges {
            rules.push(
                "USE decide AND choose FOR JUDGMENTS - Code finds WHERE things are; the decision model judges WHAT they are: \
                 whether something is so (`decide`), and which of several is meant (`choose`). Ask it rather than judging in \
                 your head.",
            );
        }
        if asks && judges {
            rules.push("USE llm_query FOR TEXT - to read or write text at length; leave judgments to decide and choose.");
        } else if asks {
            rules.push("USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.");
        }
        rules.push(
            "MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), \
             re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to \
             sanity-check, but avoid manual copying when variables can carry the exact value.",
        );
        rules.push(
            "SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, \
             run it in one step, review the result, then call SUBMIT in a later step.",
        );
        let rules: Vec<&str> = match &self.rules {
            Some(given) => given.iter().map(String::as_str).collect(),
            None => rules,
        };
        let numbered = rules.iter().enumerate().map(|(at, rule)| format!("{}. {rule}", at + 1));
        text.push_str(&numbered.collect::<Vec<_>>().join("\n"));
        text.push_str("\n\n");
        if asks {
            text.push_str(&format!("You have max {} sub-LLM calls. ", self.max_llm_calls));
        }
        text.push_str("When done, call SUBMIT() with your output.");
        // A function as `describe` has it: `switch(device, to): Set a device to on or off.`
        let listed = |line: &str| match line.split_once("): ") {
            Some((called, does)) => format!("- `{called})` - {does}"),
            None => format!("- {line}"),
        };
        let described = env.describe();
        if !described.is_empty() {
            let tools: Vec<String> = described.lines().map(listed).collect();
            text.push_str(&format!(
                "\n\nAdditional tools available (use these instead of standard library equivalents):\n{}",
                tools.join("\n")
            ));
        }
        let kept: Vec<String> = written
            .iter()
            .filter(|(_, entry)| entry.failed.is_none())
            .map(|(name, entry)| format!("- `{name}` - {}", entry.description))
            .collect();
        if !kept.is_empty() {
            text.push_str(&format!(
                "\n\nFunctions written before, which code may call as well:\n{}",
                kept.join("\n")
            ));
        }
        text
    }
}

/// The model's name, its fields and its limits. Never a key: nothing here holds one.
impl fmt::Debug for Rlm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rlm")
            .field("lm", &self.lm)
            .field("inputs", &self.inputs.keys().collect::<Vec<_>>())
            .field("outputs", &self.outputs.keys().collect::<Vec<_>>())
            .field("max_iters", &self.max_iters)
            .field("max_llm_calls", &self.max_llm_calls)
            .field(
                "checks",
                &self.checks.iter().filter_map(|check| check.name.as_ref()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// A language model the code asked that failed, as the code is told of it: it can do without,
/// and the run is not ended. A replay that was asked what it never was is the exception: the
/// run has gone another way than the one recorded, and ends.
fn queried(error: Error) -> Error {
    match error {
        Error::Model(systemone::Error::Unrecorded(why)) => Error::Model(systemone::Error::Unrecorded(why)),
        // A query stopped in the middle of its reply stops the branch, as the branch was.
        Error::Stopped(why) => Error::Stopped(why),
        other => Error::Tool(other.to_string()),
    }
}

/// One request of a run, with what it took added to the run's count. Inside a tool the
/// request reports its tokens through the call, a failed one's too.
fn ask(generate: &Generate, call: Option<&Call>, inputs: Inputs, spent: &mut Usage) -> Result<Generated> {
    let generated = match call {
        Some(call) => generate.call_in(call, inputs),
        None => generate.call(inputs),
    }?;
    add(spent, generated.usage);
    Ok(generated)
}

/// A number as a person writes it: `12,345`.
fn commas(number: usize) -> String {
    let digits = number.to_string();
    let mut written = String::new();
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at).is_multiple_of(3) {
            written.push(',');
        }
        written.push(digit);
    }
    written
}

/// The first and the last of a text that is longer than `most` characters, as two halves.
fn ends(text: &str, most: usize) -> Option<(String, String)> {
    let length = text.chars().count();
    if length <= most {
        return None;
    }
    let head = most / 2;
    let tail = most - head;
    Some((text.chars().take(head).collect(), text.chars().skip(length - tail).collect()))
}

/// A step's output as the model is shown it: how long it was, and its two ends where it is
/// longer than a step may show.
fn cut(output: &str, most: usize) -> String {
    let length = output.chars().count();
    match ends(output, most) {
        None => format!("Output ({} chars):\n{output}", commas(length)),
        Some((head, tail)) => format!(
            "Output ({} chars):\n{head}\n\n... ({} characters omitted) ...\n\n{tail}",
            commas(length),
            commas(length - most)
        ),
    }
}

/// The steps of a run so far, as the model is shown them.
fn history(steps: &[Step], most: usize) -> String {
    if steps.is_empty() {
        return "You have not interacted with the REPL environment yet.".to_owned();
    }
    let shown = steps.iter().enumerate().map(|(index, step)| step.shown(index, most));
    shown.collect::<Vec<_>>().join("\n")
}

/// What Python calls the type of a value.
fn type_of(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Number(number) if number.is_f64() => "float",
        Json::Number(_) => "int",
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// A value as text, for its length and its preview: a text as it is, and data as JSON.
fn as_text(value: &Json) -> String {
    match value {
        Json::String(text) => text.clone(),
        Json::Null => "None".to_owned(),
        Json::Bool(true) => "True".to_owned(),
        Json::Bool(false) => "False".to_owned(),
        Json::Number(number) => number.to_string(),
        data => serde_json::to_string_pretty(data).unwrap_or_default(),
    }
}

/// What the model is told of each variable: its name, its type, how long it is, and its ends.
/// The value itself is in the REPL and not in the prompt.
fn variables_info(fields: &IndexMap<String, InputField>, shown: &Map<String, Json>) -> String {
    let variable = |(name, field): (&String, &InputField)| {
        let value = shown.get(name).unwrap_or(&Json::Null);
        let text = as_text(value);
        let length = text.chars().count();
        let preview = match ends(&text, PREVIEW) {
            Some((head, tail)) => format!("{head}...{tail}"),
            None => text,
        };
        let mut lines = vec![
            format!("Variable: `{name}` (access it in your code)"),
            format!("Type: {}", type_of(value)),
        ];
        if !field.description.is_empty() {
            lines.push(format!("Description: {}", field.description));
        }
        lines.push(format!("Total length: {} characters", commas(length)));
        lines.push(format!("Preview:\n```\n{preview}\n```"));
        lines.join("\n")
    };
    fields.iter().map(variable).collect::<Vec<_>>().join("\n\n")
}

/// The arguments a function is tried with: each parameter by name from what there is to fill
/// it, or the parameters it must be given that nothing names.
fn arguments(code: &str, name: &str, fill: &Map<String, Json>) -> std::result::Result<Map<String, Json>, Vec<String>> {
    let (mut arguments, mut missing) = (Map::new(), Vec::new());
    for (parameter, required) in takes_of(code, name) {
        match fill.get(&parameter) {
            Some(value) => drop(arguments.insert(parameter, value.clone())),
            None if required => missing.push(parameter),
            None => {}
        }
    }
    if missing.is_empty() { Ok(arguments) } else { Err(missing) }
}

/// A writer of functions for a REPL: an [`Rlm`] that writes one, tries it, and submits it, and
/// the acceptance that stands between what it submits and the map. Made by [`Rlm::writer`].
///
/// When the model submits a candidate:
///
/// 1. its source is taken as a step last defined it (`SUBMIT(name, description)`: the model
///    does not write the source out again), and checked as [`Repl::define`] checks one, against
///    the REPL it is to be kept in and the map of the run. Every refusal there is given here,
///    where the model can still put it right;
/// 2. it is tried alone: in a REPL made for the trial, with the functions the scratch has and
///    nothing a step left in it, so a function that leans on a variable or a helper of the
///    scratch fails now and not when it is first pointed at. If every parameter it must be
///    given is named by an input of the request, it is run once, as the map would run it. A
///    function that fails is not kept;
/// 3. the caller's checks run ([`Writer::check`]), each with the candidate, what the trial
///    printed, and the REPL of the trial to run more in;
/// 4. only then is it defined in the real REPL.
///
/// Anything wrong at 1 to 3 is the step's output, and the model writes again.
///
/// ```
/// use rdm::lm::{Lm, ScriptedLm, input_of};
/// use rdm::repl::Repl;
/// use rdm::rlm::Rlm;
/// use rdm::{Shared, inputs};
/// use serde_json::json;
///
/// let heat = Shared::new(21);
/// let (down, now, kept) = (heat.clone(), heat.clone(), heat.clone());
/// let repl = Repl::new()
///     .function("cooler", "Make it one degree cooler.", move |_| Ok(down.update(|heat| *heat -= 1)))
///     .function("heat", "How warm it is, in degrees.", move |_| Ok(now.get()));
///
/// // It writes the function and tries it in one step, and submits it in the next.
/// let lm = Lm::of(
///     ScriptedLm::new(|_system, prompt| {
///         let code = match input_of(prompt, "iteration") {
///             Some("1/4") => "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return heat()\n\nprint(cool_to_18())",
///             _ => "SUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')",
///         };
///         json!({"reasoning": "", "code": code}).to_string()
///     }),
///     "scripted",
/// );
/// let writer = Rlm::new("Write one Python function that does what the task asks.")
///     .input("task")
///     .lm(lm)
///     .max_iters(4)
///     // The house is put back as it was: trying is not acting.
///     .checkpoint(move || {
///         let before = kept.get();
///         let kept = kept.clone();
///         move || {
///             kept.set(before);
///         }
///     })
///     .writer(&repl)
///     .check("comes to 18", |tried| if tried.printed == Some("18") { Ok(()) } else { Err("it did not come to 18".to_owned()) });
///
/// let wrote = writer.write(inputs! { "task" => "Bring the heat down to 18 degrees." }).unwrap();
/// assert_eq!(wrote.name, "cool_to_18");
/// assert_eq!(wrote.code, "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return heat()");
/// // It ran while it was tried, and the house is as it was. On the map, it has not been run.
/// assert_eq!(heat.get(), 21);
/// assert_eq!(repl.execute("print(cool_to_18())").unwrap(), "18");
/// ```
#[derive(Clone)]
pub struct Writer {
    rlm: Rlm,
    repl: Repl,
    checks: Vec<(String, Arc<Tries>)>,
    trying: Option<Arc<Trying>>,
}

/// What a writer wrote: the function, what defining it said, and the run that wrote it.
#[derive(Clone, Debug, PartialEq)]
pub struct Wrote {
    /// The function's name.
    pub name: String,
    /// The sentence a menu says of it.
    pub description: String,
    /// Its source: one `def`.
    pub code: String,
    /// What defining it returned: that it is on the list, and has not been run.
    pub said: String,
    /// The run that wrote it: its steps, and what it took.
    pub ran: Ran,
}

impl Writer {
    /// A check on a candidate: the caller's, and not the model's. It is given the function,
    /// what it printed when it was tried, and the scratch it is defined in, to run it again on
    /// whatever the check sets up. What a check that fails says is what the model reads next.
    pub fn check<F>(mut self, name: impl Into<String>, holds: F) -> Self
    where
        F: Fn(&Tried<'_>) -> std::result::Result<(), String> + Send + Sync + 'static,
    {
        self.checks.push((name.into(), Arc::new(holds)));
        self
    }

    /// Something to be told when a candidate is about to be tried, with its name. Where a
    /// stand-in keeps a record of what it was called for, this is where the record starts
    /// again: what the model did while it explored is not what the candidate does.
    pub fn trying(mut self, told: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.trying = Some(Arc::new(told));
        self
    }

    /// The RLM that writes, as it was set up.
    pub fn rlm(&self) -> &Rlm {
        &self.rlm
    }

    /// Write a function for these inputs, and define it in the REPL. For a writer that is a
    /// tool of a module, [`Writer::tool`] defines it for the run it is in.
    pub fn write(&self, inputs: Inputs) -> Result<Wrote> {
        let shown = inputs.snapshot()?;
        let (draft, ran) = self.draft(None, &shown, shown.clone())?;
        let said = self.repl.define(&draft.name, &draft.description, &draft.code)?;
        Ok(Wrote {
            name: draft.name,
            description: draft.description,
            code: draft.code,
            said,
            ran,
        })
    }

    /// The writer as a tool. Selected, its inputs are taken by name from the request's
    /// answers, then from its inputs. Its observation is what [`Repl::define_in`] said, or,
    /// where nothing was accepted, `Nothing was put on the list. {why}`: either is what the
    /// next request shows, so a decision model can point at it again.
    pub fn tool(&self, name: impl Into<String>, description: impl Into<String>) -> Tool {
        let writer = self.clone();
        Tool::new(name, description, move |call| {
            let shown = writer.rlm.shown_by(call)?;
            // What a parameter is filled from when the function is run: the answers, then the inputs.
            let mut fill = call.shown().clone();
            for name in call.names() {
                if let Some(answer) = call.answer_value(&name) {
                    fill.insert(name, answer);
                }
            }
            let nothing = |why: String| Ok(format!("Nothing was put on the list. {why}"));
            match writer.draft(Some(call), &shown, fill) {
                Ok((draft, _)) => match writer.repl.define_in(call, &draft.name, &draft.description, &draft.code) {
                    Err(Error::Tool(why)) => nothing(why),
                    done => done,
                },
                Err(Error::Tool(why)) => nothing(why),
                Err(ended) => Err(ended),
            }
        })
    }

    /// One writing: the function that was accepted, and the run that wrote it.
    fn draft(&self, call: Option<&Call>, shown: &Map<String, Json>, fill: Map<String, Json>) -> Result<(Draft, Ran)> {
        // Where it will be kept: the REPL the module that is running was given, where that is a copy.
        let target = call.map_or_else(|| self.repl.clone(), |call| self.repl.of(call));
        let taken = call.map(|call| target.others(call)).unwrap_or_default();
        let inputs: Vec<String> = match call {
            Some(call) => call.shown().keys().cloned().collect(),
            None => shown.keys().cloned().collect(),
        };
        let accepted = Arc::new(Mutex::new(None));
        let (keeps, vets, checks, trying) = (accepted.clone(), target.clone(), self.checks.clone(), self.trying.clone());
        let holds = move |submitted: &Submitted<'_>| -> Result<std::result::Result<(), String>> {
            let refused = |why: String| Ok(Err(why));
            let read = |name: &str| submitted.text(name).unwrap_or_default().trim().to_owned();
            let (name, description) = (read("name"), read("description"));
            if description.is_empty() {
                return refused("Give a description: one sentence for a menu, saying what choosing the function does.".to_owned());
            }
            // Its source as a step last defined it: the model does not write it out again to
            // submit it. Read off the steps when the turns ran out, it was given as an output.
            let given = Some(read("code")).filter(|code| !code.is_empty());
            let defined = || submitted.steps.iter().rev().find_map(|step| def_source(&step.code, &name));
            let Some(source) = given.or_else(defined) else {
                return refused(format!(
                    "No step defines a function `{name}` at its top level. Define it with `def {name}(...)` in a step, call \
                     it there to try it, and then SUBMIT."
                ));
            };
            let code = match vets.vetted(&name, &source, &taken, &inputs) {
                Ok(code) => code,
                Err(Error::Tool(why)) => return refused(why),
                Err(ended) => return Err(ended),
            };
            // Tried alone, as it will be where it is kept: in a REPL with the same functions
            // and nothing a step left behind, so that one that leans on a variable or a
            // helper of the scratch fails here and not when it is first pointed at.
            let alone = submitted.scratch.repl.spawn("trial").with_written(vets.written());
            match alone.define(&name, &description, &code) {
                Ok(_) => {}
                Err(Error::Tool(why)) => return refused(why),
                Err(ended) => return Err(ended),
            }
            let trial = Scratch {
                repl: &alone,
                ..submitted.scratch
            };
            if let Some(told) = &trying {
                told(&name);
            }
            let printed = match arguments(&code, &name, &fill) {
                // It takes what nothing here names: it cannot be tried as the map would run it.
                Err(_) => None,
                Ok(arguments) => match trial.call(&name, arguments) {
                    Ok(printed) => Some(printed),
                    Err(Error::Tool(why)) => return refused(format!("`{name}` failed when it was tried on its own. {why}")),
                    Err(ended) => return Err(ended),
                },
            };
            let tried = Tried {
                name: &name,
                description: &description,
                code: &code,
                printed: printed.as_deref(),
                scratch: trial,
            };
            for (label, holds) in &checks {
                if let Err(why) = holds(&tried) {
                    return refused(format!("{label}: {why}"));
                }
            }
            *lock(&keeps) = Some(Draft::new(name, description, code));
            Ok(Ok(()))
        };
        let mut rlm = self.rlm.clone();
        // Before any check of the RLM's own: those are given a function that was tried.
        rlm.checks.insert(
            0,
            Check {
                name: None,
                holds: Arc::new(holds),
            },
        );
        let ran = rlm.run(call, shown, target.written())?;
        let draft = lock(&accepted).take();
        let draft = draft.ok_or_else(|| Error::Tool("The run ended with no function accepted.".to_owned()))?;
        Ok((draft, ran))
    }
}

impl fmt::Debug for Writer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("rlm", &self.rlm)
            .field("repl", &self.repl)
            .field("checks", &self.checks.iter().map(|(name, _)| name).collect::<Vec<_>>())
            .finish()
    }
}

/// A REPL whose code can ask a decision model: the same REPL with `decide` and `choose` among
/// its functions.
///
/// - `decide(question, about='')` asks a yes or no and returns how likely a yes is, from 0 to 1;
/// - `choose(question, options, about='')` asks which of the options and returns the likeliest.
///
/// `about` is what the question is about, text or data, and the question refers to it as
/// `inputs.about`. They are functions like any other, so a function that was written can
/// call them when it runs: code loops and counts, which a decision model cannot, and the
/// decision model judges, which code cannot, and no language model is asked anything.
///
/// Where the code is run for a request, the question is a sub-call of that request: it is in
/// the run's trajectory, under its limits, and put to the module's own decision model. Where
/// there is no request ([`Repl::execute`], a bare [`Rlm::call`]), `model` is asked, and what
/// bounds the asking is how often code may call the functions it is given
/// ([`crate::repl::Limits::calls`]).
///
/// ```
/// use rdm::repl::Repl;
/// use rdm::rlm::deciding;
/// use rdm::systemone::{Answer, Scripted};
///
/// // A decision model that is a function: a yes for whatever mentions work.
/// let model = Scripted::new(|request| {
///     let about = request.state["inputs"]["about"].as_str().unwrap_or_default();
///     [("answer", Answer::noul(if about.contains("work") { 0.9 } else { 0.1 }))]
/// });
/// let repl = deciding(Repl::new(), model);
/// let code = "print([app for app in ['work mail', 'music'] if decide('Is `inputs.about` used for work?', app) > 0.5])";
/// assert_eq!(repl.execute(code).unwrap(), "['work mail']");
/// ```
pub fn deciding(repl: Repl, model: impl DecisionModel + 'static) -> Repl {
    let model: Arc<dyn DecisionModel> = Arc::new(model);
    let asked = |model: &Arc<dyn DecisionModel>, call: Option<&Call>, signature: Signature, about: Json| -> Result<Prediction> {
        let mut inputs = Inputs::new();
        inputs.set("about", about);
        match call {
            Some(call) => call.sub_with(Ask::new(inputs).signature(signature).max_depth(0)),
            None => Rdm::new(signature).model(model.clone()).call(inputs),
        }
    };
    let about = |about: Option<&Json>| {
        let about = about.cloned().unwrap_or_else(|| Json::String(String::new()));
        let kind = if about.is_string() { "str" } else { type_of(&about) };
        (about, kind)
    };
    let (decides, chooses) = (model.clone(), model);
    repl.asking(
        "decide(question, about='')",
        "Ask the decision model a yes-or-no question about `about`, text or data: returns how likely a yes is, from 0.0 \
         to 1.0. Write the question to be read on its own, and refer to what it is about as `inputs.about`.",
        move |args, call| {
            let question: String = args.get("question")?;
            let (about, kind) = about(args.value("about"));
            let signature = Signature::new("Answer a question about what is shown.")
                .input_with("about", kind, "What the question is about.")
                .noul("answer", question);
            Ok(asked(&decides, call, signature, about)?.probability("answer").unwrap_or(0.0))
        },
    )
    .asking(
        "choose(question, options, about='')",
        "Ask the decision model which of `options`, a list of texts, answers a question about `about`: returns the \
         likeliest of them. Refer to what the question is about as `inputs.about`.",
        move |args, call| {
            let (question, options): (String, Vec<String>) = (args.get("question")?, args.get("options")?);
            if options.is_empty() {
                return Err(Error::Tool("choose() needs at least one option".to_owned()));
            }
            let (about, kind) = about(args.value("about"));
            let signature = Signature::new("Answer a question about what is shown.")
                .input_with("about", kind, "What the question is about.")
                .choice("answer", question, options.clone());
            let answered = asked(&chooses, call, signature, about)?;
            Ok(answered.chosen("answer").map_or_else(|| options[0].clone(), str::to_owned))
        },
    )
}
