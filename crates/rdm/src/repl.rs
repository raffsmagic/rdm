//! Repl: a sandbox a decision model and a writer share, whose functions are names on the map.
//!
//! A decision model can only point, so a module can only do what is already on its map. A
//! language model can write. [`Repl`] is the environment the two share: a sandboxed Python
//! interpreter (Monty), where the functions it is given are functions written code may call.
//!
//! ```text
//! the decision model points at `write`         nothing on the map does what is still needed
//! the writer calls `define(name, ...)`         with the source of one function
//! the function is defined in the REPL          and its name is on the map from the next request
//! the decision model points at the name        and the REPL runs it
//! ```
//!
//! The language model never chooses what happens. It makes new things that can be pointed at,
//! with what a decision model cannot do: loops, arithmetic, conditions. A `Repl` never asks a
//! language model anything itself: [`Repl::define_in`] is a plain function, which a tool of
//! yours calls with what a language model wrote.
//!
//! ```
//! use rdm::repl::Repl;
//! use rdm::{Rdm, Shared, Signature, Tool, inputs};
//! use systemone::{Answer, Scripted};
//!
//! #[derive(serde::Serialize)]
//! struct House {
//!     heat: i64,
//! }
//! let house = Shared::new(House { heat: 21 });
//!
//! // The sandbox, and what written code may call in it.
//! let (down, now) = (house.clone(), house.clone());
//! let repl = Repl::new()
//!     .function("cooler", "Make it one degree cooler.", move |_| {
//!         Ok(down.update(|house| {
//!             house.heat -= 1;
//!             format!("heat: now {}", house.heat)
//!         }))
//!     })
//!     .function("heat", "How warm it is, in degrees.", move |_| Ok(now.lock().heat));
//!
//! // The writer is a tool like any other. Here it is a function; in a program it asks a language model.
//! let writing = repl.clone();
//! let write = Tool::new("write", "Have a new tool written in code: nothing listed does what is still needed.", move |call| {
//!     let code = "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return f'heat: now {heat()}'";
//!     writing.define_in(call, "cool_to_18", "Bring the heat down to 18 degrees.", code)
//! });
//!
//! let operate = Signature::new("Operate a smart home, one step at a time.")
//!     .input("task")
//!     .input_typed("house", "dict")
//!     .input_described("observation", "What the last step did.")
//!     .tool("next", "What should happen next for `inputs.task`?");
//! // A decision model that points at what was written once it is on the map, and stops when it is cool.
//! let model = Scripted::new(|request| {
//!     let labels = request.questions["next"].labels();
//!     let written = labels.iter().any(|label| label == "cool_to_18");
//!     let cool = request.state["inputs"]["house"]["heat"] == 18;
//!     let next = if cool { "finish" } else if written { "cool_to_18" } else { "write" };
//!     [("next", Answer::chose(next, labels.iter().map(String::as_str)))]
//! });
//!
//! let cool = Rdm::new(operate).tool(write).source(repl.clone()).model(model);
//! let result = cool.call(inputs! { "task" => "Bring the heat down to 18 degrees.", "house" => &house }).unwrap();
//! assert_eq!(result.pointed("next"), ["write", "cool_to_18", "finish"]);
//! assert_eq!(house.lock().heat, 18);
//! assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["cool_to_18"]);
//! ```
//!
//! A written function is a tool like any other: its parameters are filled by name from the
//! request's answers, then its inputs. So what a function should be told (how many, which one)
//! is a question of the request, answered by the decision model, and not a number written into
//! the function.
//!
//! Something found out and worth keeping is written the same way, as a function that returns
//! it: pointing at it reads it.
//!
//! What was written is [`Repl::written`], plain data: it serializes to each name with its
//! `description` and `code`, and [`Repl::with_written`] defines it again, in another program or
//! another day. A copy of a module has a REPL of its own with the same functions.

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use monty::{MontyRepl, ReplProgress, ReplStartError};
use monty_types::{
    CallArgs, CompileOptions, DateTimeSource, ExcType, ExtFunctionResult, MontyDate, MontyDateTime, MontyException, MontyObject,
    NameLookupResult, ObjectRef, OsFunctionCall, OsPolicy, PrintWriter, RandomStart, ResourceLimits, ResourceTracker,
    SOURCE_SCAN_THRESHOLD,
};
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::{Expr, Parameters, Stmt, StmtFunctionDef};
use ruff_python_parser::Mode;
use ruff_python_parser::lexer::lex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, json};
use systemone::Json;

use crate::call::Call;
use crate::error::{Error, Result, listed, sorted};
use crate::module::Run;
use crate::record::Event;
use crate::signature::FINISH;
use crate::tool::{Source, Tool};

/// The variable a written function's arguments are given to the REPL in.
pub const ARGUMENTS: &str = "_rdm_arguments";

/// Names the REPL uses itself: no function is written under one, and no input is a variable under one.
const USED: [&str; 2] = ["print", ARGUMENTS];

/// One function that was written: what choosing it does, and its source.
///
/// It serializes to `{"description": ..., "code": ...}`, with `"failed"` beside them once the
/// function has failed: the shape a saved map has.
///
/// ```
/// use rdm::repl::{Functions, Written};
///
/// let saved = r#"{"movie_time":{"description":"TV on, lights off.","code":"def movie_time():\n    return 'on'"}}"#;
/// let written: Functions = serde_json::from_str(saved).unwrap();
/// assert_eq!(written["movie_time"], Written::new("TV on, lights off.", "def movie_time():\n    return 'on'"));
/// assert_eq!(serde_json::to_string(&written).unwrap(), saved);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Written {
    /// One sentence for a menu: what choosing the function does.
    pub description: String,
    /// The function's source: one `def`.
    pub code: String,
    /// The error it failed with when it was run. A function that failed is off the map; its
    /// source is kept, marked, for whoever writes it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<String>,
}

impl Written {
    /// A function that has not failed.
    pub fn new(description: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            code: code.into(),
            failed: None,
        }
    }
}

/// Every function written, by name, in the order they were written: what [`Repl::written`]
/// returns and [`Repl::with_written`] takes. It is plain data, the `written` object of a saved map.
pub type Functions = IndexMap<String, Written>;

/// What a writer gives: a function's name, the sentence a menu says of it, and its source.
///
/// It reads from JSON of the same three keys, so what a language model was asked for as an
/// object is one ([`Draft::read`]). [`Repl::writer`] makes a tool of whatever gives one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Draft {
    /// The function's name.
    pub name: String,
    /// One sentence for a menu: what choosing it does.
    pub description: String,
    /// The function's source: one `def` of that name and nothing beside it.
    pub code: String,
}

impl Draft {
    /// A function as it was written.
    pub fn new(name: impl Into<String>, description: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            code: code.into(),
        }
    }

    /// A function read from what a writer gave as data: an object with a `name`, a
    /// `description` and a `code`, each a text, such as the outputs of a signature a language
    /// model completed. Whatever else is in it is left.
    ///
    /// What a language model gave is never how the module was used, so data that is not
    /// those three is a tool's failure ([`Error::Tool`]): inside a tool that writes, it is the
    /// tool's observation, and the run goes on.
    ///
    /// ```
    /// use rdm::repl::Draft;
    /// use serde_json::json;
    ///
    /// let gave = json!({"name": "hello", "description": "Say hello.", "code": "def hello():\n    return 'hello'", "reasoning": "It is short."});
    /// assert_eq!(Draft::read(&gave).unwrap(), Draft::new("hello", "Say hello.", "def hello():\n    return 'hello'"));
    ///
    /// let error = Draft::read(&json!({"name": "hello"})).unwrap_err();
    /// assert_eq!(error.to_string(), "What was given is not a function's name, description and code: missing field `description`.");
    /// ```
    pub fn read(gave: &impl Serialize) -> Result<Self> {
        let unfit =
            |error: serde_json::Error| Error::Tool(format!("What was given is not a function's name, description and code: {error}."));
        serde_json::from_value(serde_json::to_value(gave).map_err(unfit)?).map_err(unfit)
    }
}

/// What written code is held to, so that it cannot hang the program that runs it.
///
/// The interpreter runs in this process. None of these is a wall of a prison: they end a loop
/// that does not end, a recursion that does not return, a value too large to hold. Code that
/// reaches one has failed, as code that divides by zero has.
///
/// What they do not do is count what the code holds. Code that makes value after value, each
/// one within [`Limits::memory`], and keeps them all, has until [`Limits::time`] to fill the
/// memory of the process, and a process that cannot be given memory is ended. Nor is every
/// value measured before it is made: Monty 1.0 measures what `*`, `+` and `replace` would
/// make, and not what `str.join` would, so one join of large strings is larger than
/// [`Limits::memory`] allows, and large enough ends the process the same way. Either takes
/// code written to do it. What is not trusted that far belongs in a process of its own.
///
/// A source is held to something as well, before any of it is run: one nested too far, or
/// one chain too long (`x.a.a.a` a thousand times over), is refused as too deeply nested and
/// not read. Reading the deepest source that is not refused takes Monty about a megabyte of
/// stack, so a thread that uses a REPL wants the 2 MiB a thread of Rust has, or more.
///
/// ```
/// use std::time::Duration;
///
/// use rdm::repl::{Limits, Repl};
///
/// let repl = Repl::new().limits(Limits { time: Duration::from_millis(50), ..Limits::default() });
/// let error = repl.execute("while True:\n    pass").unwrap_err();
/// assert!(error.to_string().starts_with("TimeoutError"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// How long one run may execute. The time spent in the functions of the host is not
    /// counted: [`Limits::calls`] bounds those. 2 seconds.
    pub time: Duration,
    /// The most memory one value may take, in bytes: a string or a list made larger at a
    /// stroke by `*`, `+` or `replace` is refused. Not every way to make a value is measured
    /// (`str.join` is not, in Monty 1.0), and what grows a piece at a time is ended by the
    /// time. 64 MiB.
    pub memory: usize,
    /// How deep a function may call itself. 200.
    pub recursion: usize,
    /// How many times one run may call the functions of the host: what ends
    /// `while heat() > 18: cooler()` when the heat does not fall. Whatever else the code
    /// stops for is counted with them: a name it reads and never defined, a file it asks for
    /// and is refused. 1000.
    pub calls: usize,
    /// The most one run may print, in bytes. 1 MiB.
    pub output: usize,
    /// How long a call waits for the REPL while it runs something else. A REPL runs one thing
    /// at a time, and whatever asks while it is busy waits its turn; one that waited this long
    /// is refused, so that code waiting on itself is an error and not a program that hangs.
    /// 30 seconds.
    pub wait: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            time: Duration::from_secs(2),
            memory: 64 * 1024 * 1024,
            recursion: 200,
            calls: 1000,
            output: 1024 * 1024,
            wait: Duration::from_secs(30),
        }
    }
}

/// What written code called a function of the host with.
///
/// An argument is read by the name the function declared for it, whether the code gave it by
/// position or by keyword: for `"switch(device, to)"`, both `switch("tv", "off")` and
/// `switch(to="off", device="tv")` give `device` and `to`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args {
    function: String,
    named: Map<String, Json>,
    rest: Vec<Json>,
}

impl Args {
    /// An argument, read as a type. One that was not given, or is not of that type, is the
    /// function's failure: written code sees it as a `RuntimeError`.
    pub fn get<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        let value = self
            .named
            .get(name)
            .ok_or_else(|| Error::Tool(format!("{}() was not given `{name}`.", self.function)))?;
        serde_json::from_value(value.clone())
            .map_err(|error| Error::Tool(format!("{}() was given `{name}` as a type it is not: {error}.", self.function)))
    }

    /// An argument that may be left out: None where it was not given, or was given as `None`.
    pub fn opt<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>> {
        match self.named.get(name) {
            None | Some(Json::Null) => Ok(None),
            Some(_) => self.get(name).map(Some),
        }
    }

    /// An argument as it was given, if it was.
    pub fn value(&self, name: &str) -> Option<&Json> {
        self.named.get(name)
    }

    /// The arguments as a record keeps them: by the names declared, and what was given beyond
    /// them under `*`.
    fn written(&self) -> Json {
        let mut written = self.named.clone();
        if !self.rest.is_empty() {
            written.insert("*".to_owned(), Json::Array(self.rest.clone()));
        }
        Json::Object(written)
    }

    /// Every argument that has a name: those the function declared, and where it declared
    /// `**more`, the keywords beside them.
    pub fn named(&self) -> &Map<String, Json> {
        &self.named
    }

    /// The arguments given by position beyond those the function named, where it declared `*rest`.
    pub fn rest(&self) -> &[Json] {
        &self.rest
    }
}

type Does = dyn Fn(&Args) -> Result<Json> + Send + Sync;

/// Where written code's randomness comes from: given a count, that many bytes.
type Randomness = dyn Fn(usize) -> Result<Vec<u8>> + Send + Sync;

/// Where written code's clock is read: the instant, in seconds since the Unix epoch.
type Clock = dyn Fn() -> Result<f64> + Send + Sync;

/// What a REPL answers its code from besides its functions: its randomness and its clock.
#[derive(Clone)]
struct Outside {
    randomness: Arc<Randomness>,
    clock: Arc<Clock>,
}

impl Default for Outside {
    fn default() -> Self {
        Outside {
            randomness: Arc::new(entropy),
            clock: Arc::new(system_clock),
        }
    }
}

/// The clock a REPL has unless it is given another: the system's.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
fn system_clock() -> Result<f64> {
    let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    since
        .map(|since| since.as_secs_f64())
        .map_err(|_| Error::Tool("the system's clock is set before 1970".to_owned()))
}

/// The clock a REPL has unless it is given another, in a WebAssembly module with nothing under
/// it: none, since there is none to read there.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
fn system_clock() -> Result<f64> {
    Err(Error::Tool(
        "this WebAssembly module has no clock: a REPL here tells the time only by a clock it is given (`Repl::clock`)".to_owned(),
    ))
}

/// Bytes from the hex a `drew` event writes them in. None where it is not hex.
fn unhex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) || !hex.is_ascii() {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).ok())
        .collect()
}

/// The randomness a REPL has unless it is given another: the operating system's.
fn entropy(count: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; count];
    getrandom::fill(&mut bytes).map_err(|error| Error::Tool(format!("the operating system gave no randomness: {error}")))?;
    Ok(bytes)
}
type DoesFor = dyn Fn(&Args, Option<&Call>) -> Result<Json> + Send + Sync;

/// What a function of the host does when written code calls it.
#[derive(Clone)]
enum Doing {
    /// A closure over the arguments.
    Function(Arc<Does>),
    /// A closure over the arguments and the request the code is run for, where there is one.
    Asking(Arc<DoesFor>),
    /// A closure over the arguments that ends the code that called it: nothing after the
    /// call runs, and what the code printed until then is what it printed.
    Last(Arc<Does>),
    /// A tool of a module, run for the request that pointed at the written function.
    Tool(Tool),
}

/// A function that exists in the REPL for written code to call.
#[derive(Clone)]
struct Host {
    takes: Takes,
    /// How a writer is shown it: `switch(device, to)`.
    shown: String,
    description: String,
    doing: Doing,
}

/// What a function failed with, as the code that called it is to see it.
enum Raised {
    /// An exception at the call, which the code may catch.
    Inside(MontyException),
    /// The end of the run: the module was used wrongly, or a decision model failed.
    Outside(Error),
    /// Nothing failed: the function is one that ends the code that called it.
    Finished,
}

impl Host {
    /// Call it for code `repl` is running, for `request` where there is one. What a function
    /// that is the host's own answered is handed to the run's record.
    fn call(&self, repl: &str, name: &str, given: &CallArgs, request: Option<&Call>) -> std::result::Result<Json, Raised> {
        let positional: Vec<Json> = given.args().map(to_json).collect();
        let keyword: Map<String, Json> = given
            .kwargs()
            .map(|(key, value)| (key.as_str().unwrap_or_default().to_owned(), to_json(value)))
            .collect();
        let args = self
            .takes
            .bind(name, positional, keyword)
            .map_err(|error| Raised::Inside(MontyException::new(ExcType::TypeError, Some(error))))?;
        let doing = || match (&self.doing, request) {
            (Doing::Function(does) | Doing::Last(does), _) => does(&args),
            (Doing::Asking(does), request) => does(&args, request),
            (Doing::Tool(tool), Some(request)) => tool.call(request),
            (Doing::Tool(_), None) => Err(Error::Tool(format!(
                "`{name}` is a tool of a module: it runs for a request, and no request is running this code."
            ))),
        };
        let raise = |error: String| Raised::Inside(MontyException::new(ExcType::RuntimeError, Some(error)));
        // What a function leaves half done when it panics is its own, as a tool's is.
        let answered = match catch_unwind(AssertUnwindSafe(doing)) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(Error::Tool(error))) => Err(error),
            Ok(Err(other)) => return Err(Raised::Outside(other)),
            Err(panic) => Err(panicked(panic.as_ref())),
        };
        // A function that asks the module again, or ends the code, is the REPL's own doing, and
        // what it does is in the run's other events. The others answer from the world.
        if let (Some(request), Doing::Function(_) | Doing::Tool(_)) = (request, &self.doing) {
            let (value, error) = match &answered {
                Ok(value) => (Some(value.clone()), None),
                Err(error) => (None, Some(error.clone())),
            };
            let kept = request.emit(|run, index| Event::Answered {
                run,
                index,
                repl: repl.to_owned(),
                function: name.to_owned(),
                args: Some(args.written()),
                value,
                error,
            });
            kept.map_err(Raised::Outside)?;
        }
        match answered {
            Ok(_) if matches!(self.doing, Doing::Last(_)) => Err(Raised::Finished),
            Ok(value) => Ok(value),
            Err(error) => Err(raise(error)),
        }
    }
}

/// What a panic said.
fn panicked(panic: &(dyn Any + Send)) -> String {
    let text = panic.downcast_ref::<&str>().map(|text| (*text).to_owned());
    text.or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "the function panicked".to_owned())
}

/// A sandboxed Python REPL whose functions a module can point at.
///
/// Given to a module as a source ([`crate::Rdm::source`]), it offers every function written so
/// far: each request's map holds them as they are then. [`Repl::define`] writes one and
/// [`Repl::remove`] takes one away; both are plain functions, to call from a tool of your own.
/// When the decision model points at a written function, the REPL runs it. Its parameters are
/// filled as any tool's are, by name from the request's answers, then its inputs, and the
/// request's inputs are variables in the REPL as well: copies, so a function changes the world
/// through the functions it is given and not by changing a variable.
///
/// Writing only defines: the source is one `def` and nothing beside it, under a name that no
/// tool has. A function that fails when it is run is taken off the map, and the error is what
/// the next request shows as its observation, so that it can be written again.
///
/// Nothing of the machine is reached from written code: no file, no environment and no sleep.
/// The time it reads and what it draws at random come from the REPL ([`Repl::clock`],
/// [`Repl::randomness`]): the system's clock and the operating system's randomness unless the
/// REPL is given others. So the same code, given the same answers, the same time and the same
/// randomness, prints the same, here and in any other process, and a record of a run has all of them.
///
/// It is a handle: a clone is the same REPL, so a tool that writes and the module that points
/// hold one between them, and a function given to one is given to both. [`Repl::copy`] is
/// another REPL with the same functions.
///
/// ```
/// use rdm::repl::Repl;
/// use rdm::Shared;
///
/// let heat = Shared::new(21);
/// let (down, now) = (heat.clone(), heat.clone());
/// let repl = Repl::new()
///     .function("cooler", "Make it one degree cooler.", move |_| {
///         down.update(|heat| *heat -= 1);
///         Ok(())
///     })
///     .function("heat", "How warm it is, in degrees.", move |_| Ok(now.get()));
///
/// // Writing runs nothing.
/// let said = repl.define("cool_by", "Make it cooler by some degrees.", "def cool_by(degrees=3):\n    for _ in range(degrees):\n        cooler()\n    return heat()");
/// assert_eq!(
///     said.unwrap(),
///     "`cool_by` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Make it cooler by some degrees."
/// );
/// assert_eq!(heat.get(), 21);
///
/// assert_eq!(repl.execute("print(cool_by())").unwrap(), "18");
/// assert_eq!(repl.describe(), "cooler(): Make it one degree cooler.\nheat(): How warm it is, in degrees.");
/// ```
#[derive(Clone)]
pub struct Repl {
    inner: Arc<Inner>,
}

struct Inner {
    name: Mutex<String>,
    /// The functions of the host, by name. Replaced whole when one is added, so that code that
    /// is running keeps the ones it started with.
    hosts: Mutex<Arc<IndexMap<String, Host>>>,
    limits: Mutex<Limits>,
    /// What written code draws at random, and the clock it reads, come from here.
    outside: Mutex<Outside>,
    /// Every function written, by name, with its description and its source.
    written: Mutex<Functions>,
    gate: Mutex<Gate>,
    /// Told when the REPL has finished what it was running.
    freed: Condvar,
    /// Every copy made of this REPL.
    copies: Mutex<Vec<Weak<Inner>>>,
    /// What this REPL has in common with every copy made of it, and of them, and with no other.
    family: Arc<()>,
    /// The functions a tool took away that its run's map may hold still.
    taken_off: Mutex<Vec<TakenOff>>,
}

/// A function taken away from inside a tool. A run's map is made when a request is, so until
/// the run's next request it holds the name still, and the name is free all the same.
struct TakenOff {
    run: Weak<Run>,
    /// Where in the trajectory the request is whose tool took it away.
    index: usize,
    name: String,
}

/// Whose turn it is. The interpreter is not behind a lock while it runs: it is taken out, and
/// what is here says who has it, so that whatever asks meanwhile can tell waiting its turn
/// from waiting on itself.
#[derive(Default)]
struct Gate {
    /// The interpreter between runs. None before the first use, while something runs, and
    /// after a run it did not come back from whole: it is made again when it is next needed.
    idle: Option<MontyRepl>,
    running: Option<Running>,
    /// Counted up when the REPL is started again, so that an interpreter that was running
    /// then is not kept.
    started: u64,
}

/// What a REPL is running for.
struct Running {
    thread: ThreadId,
    /// The request that pointed at the function being run, where one did.
    request: Option<Pointed>,
}

/// A request that pointed at a written function: its run, its place in the trajectory, and
/// the function. A request the function's code asks for is recorded one level below it,
/// under that function's name.
struct Pointed {
    run: usize,
    index: usize,
    depth: usize,
    function: String,
}

impl Pointed {
    fn by(call: &Call) -> Option<Self> {
        Some(Self {
            run: Arc::as_ptr(&call.active.run) as usize,
            index: call.active.index,
            depth: call.active.depth,
            function: call.tool()?.to_owned(),
        })
    }
}

/// One turn on the interpreter. When it ends, however it ends, the REPL is free again.
struct Turn<'a> {
    inner: &'a Inner,
    repl: Option<MontyRepl>,
    started: u64,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut gate = self.inner.gate();
        if gate.started == self.started {
            gate.idle = self.repl.take();
        }
        gate.running = None;
        drop(gate);
        self.inner.freed.notify_all();
    }
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

impl Repl {
    /// A REPL called `repl`, with nothing in it but Python.
    pub fn new() -> Self {
        Self::made(
            "repl".to_owned(),
            Arc::default(),
            (Limits::default(), Outside::default()),
            Functions::new(),
            Arc::default(),
        )
    }

    fn made(
        name: String,
        hosts: Arc<IndexMap<String, Host>>,
        (limits, outside): (Limits, Outside),
        written: Functions,
        family: Arc<()>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                family,
                name: Mutex::new(name),
                hosts: Mutex::new(hosts),
                limits: Mutex::new(limits),
                outside: Mutex::new(outside),
                written: Mutex::new(written),
                gate: Mutex::default(),
                freed: Condvar::new(),
                copies: Mutex::default(),
                taken_off: Mutex::default(),
            }),
        }
    }

    /// The environment's name, where there are several: who a change in the trajectory is by.
    pub fn named(self, name: impl Into<String>) -> Self {
        *locked(&self.inner.name) = name.into();
        self
    }

    /// What the environment is called.
    pub fn name(&self) -> String {
        locked(&self.inner.name).clone()
    }

    /// A function that exists inside the REPL for written code to call: its name, what a
    /// writer is told of it, and what it does.
    ///
    /// The name may say what it takes, as Python declares it: `"switch(device, to)"`,
    /// `"wait(seconds=1)"`, `"log(*lines)"`. A bare name takes nothing. Written code is held
    /// to it, as Python holds a caller to a `def`: an argument too many, or a keyword the
    /// function does not have, is a `TypeError` in the code, which a writer can read and
    /// put right. The closure reads each argument by its name ([`Args::get`]).
    ///
    /// What it returns is what the code is given: any plain data. What it fails with is
    /// raised in the code as a `RuntimeError`, as is a panic; a function written to expect it
    /// may catch it, and otherwise the function has failed. Only a usage error or a decision
    /// model's failure ends the run, as it does anywhere.
    ///
    /// ```
    /// use rdm::repl::Repl;
    /// use rdm::Shared;
    ///
    /// let house = Shared::new(std::collections::BTreeMap::from([("tv".to_owned(), "off".to_owned())]));
    /// let devices = house.clone();
    /// let repl = Repl::new().function("switch(device, to)", "Set a device to on or off.", move |args| {
    ///     let (device, to): (String, String) = (args.get("device")?, args.get("to")?);
    ///     devices.lock().insert(device.clone(), to.clone());
    ///     Ok(format!("{device}: now {to}"))
    /// });
    ///
    /// assert_eq!(repl.execute("print(switch('tv', to='on'))").unwrap(), "tv: now on");
    /// assert_eq!(house.lock()["tv"], "on");
    /// assert!(repl.execute("switch('tv')").unwrap_err().to_string().starts_with("TypeError: switch() missing required argument: 'to'"));
    /// assert_eq!(repl.describe(), "switch(device, to): Set a device to on or off.");
    /// ```
    pub fn function<F, V>(self, declared: impl AsRef<str>, description: impl Into<String>, does: F) -> Self
    where
        F: Fn(&Args) -> Result<V> + Send + Sync + 'static,
        V: Serialize,
    {
        let (name, takes, shown) = declared_as(declared.as_ref());
        let does = move |args: &Args| Ok(serde_json::to_value(does(args)?)?);
        self.host(
            name,
            Host {
                takes,
                shown,
                description: description.into(),
                doing: Doing::Function(Arc::new(does)),
            },
        )
    }

    /// A function inside the REPL that is given the request as well: its arguments, and the
    /// [`Call`] the code is being run for, where there is one. It is declared and described as
    /// [`Repl::function`] has it.
    ///
    /// There is a request when a decision model pointed at a written function, and none when
    /// code is run with [`Repl::execute`]. With one, the function can ask the module again
    /// ([`Call::sub_with`]), and what it asks joins the run: its trajectory and its limits.
    ///
    /// ```
    /// use rdm::repl::Repl;
    ///
    /// let repl = Repl::new().asking("where()", "Whether the code is run for a request.", |_, call| {
    ///     Ok(if call.is_some() { "for a request" } else { "for nobody" })
    /// });
    /// assert_eq!(repl.execute("print(where())").unwrap(), "for nobody");
    /// ```
    pub fn asking<F, V>(self, declared: impl AsRef<str>, description: impl Into<String>, does: F) -> Self
    where
        F: Fn(&Args, Option<&Call>) -> Result<V> + Send + Sync + 'static,
        V: Serialize,
    {
        let (name, takes, shown) = declared_as(declared.as_ref());
        let does = move |args: &Args, call: Option<&Call>| Ok(serde_json::to_value(does(args, call)?)?);
        self.host(
            name,
            Host {
                takes,
                shown,
                description: description.into(),
                doing: Doing::Asking(Arc::new(does)),
            },
        )
    }

    /// A function inside the REPL that ends the code that calls it. It is declared and
    /// described as [`Repl::function`] has it, and what it does is done first.
    ///
    /// Nothing after the call runs: not the next line, not an `except` round it, and not a
    /// `finally`, since what ends the code is nothing the code can catch. What it printed until
    /// then is what it printed, and what it left in the REPL stays. A function that fails has
    /// failed as any other does, and the code goes on, or does not, as it was written to.
    ///
    /// ```
    /// use rdm::Shared;
    /// use rdm::repl::Repl;
    ///
    /// let kept = Shared::new(0);
    /// let keeps = kept.clone();
    /// let repl = Repl::new().ending("done(total)", "Give the total and stop.", move |args| {
    ///     keeps.set(args.get("total")?);
    ///     Ok(())
    /// });
    /// let printed = repl.execute("print('adding')\ntry:\n    done(1 + 2)\nfinally:\n    print('never')\nprint('nor this')");
    /// assert_eq!((printed.unwrap().as_str(), kept.get()), ("adding", 3));
    /// ```
    pub fn ending<F>(self, declared: impl AsRef<str>, description: impl Into<String>, does: F) -> Self
    where
        F: Fn(&Args) -> Result<()> + Send + Sync + 'static,
    {
        let (name, takes, shown) = declared_as(declared.as_ref());
        let does = move |args: &Args| does(args).map(|()| Json::Null);
        self.host(
            name,
            Host {
                takes,
                shown,
                description: description.into(),
                doing: Doing::Last(Arc::new(does)),
            },
        )
    }

    /// A tool of a module as a function inside the REPL, under the tool's name and with its
    /// description: written code can then do in a loop what the map does one step at a time.
    ///
    /// A tool is a function of a request and takes no arguments of its own, so written code
    /// calls it with none: `cooler()`. It runs for the request that pointed at the written
    /// function, as if that request had selected it: what it reads by name it reads from that
    /// request's answers and inputs, and a sub-call it makes is a sub-call of that request. A
    /// tool that fails is a `RuntimeError` in the code. Run with no request ([`Repl::execute`]),
    /// it fails. A tool whose name is not a Python name cannot be called, and a writer is not
    /// told of it ([`Repl::describe`]): give it one first ([`Tool::named`]).
    ///
    /// ```
    /// use rdm::repl::Repl;
    /// use rdm::{Rdm, Shared, Signature, Tool, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let heat = Shared::new(21);
    /// let down = heat.clone();
    /// let cooler = Tool::new("cooler", "Make it one degree cooler.", move |_| Ok(down.update(|heat| *heat -= 1)));
    ///
    /// let repl = Repl::new().tool(&cooler);
    /// repl.define("cool_three", "Make it three degrees cooler.", "def cool_three():\n    for _ in range(3):\n        cooler()").unwrap();
    ///
    /// let operate = Signature::new("Operate a heater.")
    ///     .input("task")
    ///     .input("observation")
    ///     .tool("next", "What should happen next for `inputs.task`?");
    /// let model = Scripted::new(|request| {
    ///     let labels = request.questions["next"].labels();
    ///     let next = if request.state["inputs"]["observation"] == "" { "cool_three" } else { "finish" };
    ///     [("next", Answer::chose(next, labels.iter().map(String::as_str)))]
    /// });
    /// Rdm::new(operate).tool(cooler).source(repl).model(model).call(inputs! { "task" => "Three cooler." }).unwrap();
    /// assert_eq!(heat.get(), 18);
    /// ```
    pub fn tool(self, tool: &Tool) -> Self {
        let name = tool.name().to_owned();
        self.host(
            name.clone(),
            Host {
                takes: Takes::default(),
                shown: format!("{name}()"),
                description: tool.description().to_owned(),
                doing: Doing::Tool(tool.clone()),
            },
        )
    }

    fn host(self, name: String, host: Host) -> Self {
        let mut hosts = locked(&self.inner.hosts);
        let mut all = (**hosts).clone();
        all.insert(name, host);
        *hosts = Arc::new(all);
        drop(hosts);
        self
    }

    /// What written code is held to. See [`Limits`].
    pub fn limits(self, limits: Limits) -> Self {
        *locked(&self.inner.limits) = limits;
        self.restart();
        self
    }

    /// Where what written code draws at random comes from: the first state of `random`, and
    /// what `os.urandom` returns. Given a count, it gives that many bytes. The operating
    /// system's unless this says otherwise.
    ///
    /// It is the one thing besides its functions that a REPL is answered from outside. A host
    /// that keeps a record of a run gives a source that records what it gave, and one that
    /// replays a run gives back what was recorded: the code then prints what it printed.
    /// The REPL is started again, so that `random` is seeded from it.
    ///
    /// ```
    /// use rdm::repl::Repl;
    ///
    /// let draw = "import random\nprint(random.randint(1, 1000), random.randint(1, 1000))";
    /// let given = |count: usize| Ok((0..count).map(|at| at as u8).collect());
    /// let (one, other) = (Repl::new().randomness(given), Repl::new().randomness(given));
    /// assert_eq!(one.execute(draw).unwrap(), other.execute(draw).unwrap());
    /// ```
    pub fn randomness(self, gives: impl Fn(usize) -> Result<Vec<u8>> + Send + Sync + 'static) -> Self {
        locked(&self.inner.outside).randomness = Arc::new(gives);
        self.restart();
        self
    }

    /// The clock written code reads: `time.time()`, `datetime.now()`, `date.today()` and the
    /// clocks `time` reckons from it. It gives the instant, in seconds since the Unix epoch, and
    /// is read once for each of them. The system's clock unless this says otherwise; in a
    /// WebAssembly module with nothing under it there is none, and the code is refused.
    ///
    /// A host that keeps a record of a run has each reading as a `clocked` event, and a replay
    /// reads the clock as it was then. A clock that gives an error is refused: the code sees a
    /// `PermissionError` that says what it said. So a REPL that is to tell no time is given
    /// `|| Err("there is no clock here".into())`.
    ///
    /// ```
    /// use rdm::repl::Repl;
    ///
    /// let repl = Repl::new().clock(|| Ok(1_700_000_000.25));
    /// let code = "import time, datetime\nprint(time.time(), datetime.datetime.now(), datetime.date.today())";
    /// assert_eq!(repl.execute(code).unwrap(), "1700000000.25 2023-11-14 22:13:20.250000 2023-11-14");
    /// let none = Repl::new().clock(|| Err("there is no clock here".into()));
    /// assert_eq!(none.execute("import time\ntime.time()").unwrap_err().to_string(), "PermissionError: there is no clock here");
    /// ```
    pub fn clock(self, reads: impl Fn() -> Result<f64> + Send + Sync + 'static) -> Self {
        locked(&self.inner.outside).clock = Arc::new(reads);
        self
    }

    /// Functions written before, as another REPL's [`Repl::written`] holds them, or a saved
    /// map does: each name with its `description` and `code`. They are defined when the REPL
    /// is first used, and are on the map from the first request.
    ///
    /// ```
    /// use rdm::repl::{Functions, Repl};
    ///
    /// let saved = r#"{"written": {"boiler": {"description": "What was found wrong with the boiler.", "code": "def boiler():\n    return 'The pilot light is out.'"}}}"#;
    /// let map: serde_json::Value = serde_json::from_str(saved).unwrap();
    /// let written: Functions = serde_json::from_value(map["written"].clone()).unwrap();
    ///
    /// let repl = Repl::new().with_written(written);
    /// assert_eq!(repl.execute("print(boiler())").unwrap(), "The pilot light is out.");
    /// // What it holds is the same plain data, to save again.
    /// assert_eq!(serde_json::to_value(repl.written()).unwrap(), map["written"]);
    /// ```
    pub fn with_written<N: Into<String>>(self, written: impl IntoIterator<Item = (N, Written)>) -> Self {
        self.load(written);
        self
    }

    /// Start again from these functions: afterwards the REPL holds them and nothing else that
    /// was written or kept in it.
    pub fn load<N: Into<String>>(&self, written: impl IntoIterator<Item = (N, Written)>) {
        *locked(&self.inner.written) = written.into_iter().map(|(name, entry)| (name.into(), entry)).collect();
        self.restart();
    }

    /// Start the interpreter again: what was written is defined in it when it is next used,
    /// and whatever else code left in it is gone.
    pub fn restart(&self) {
        let mut gate = self.inner.gate();
        gate.idle = None;
        gate.started += 1;
    }

    /// Every function written, by name, with its description and its source, and the error of
    /// one that failed. It is plain data that serializes to the shape a saved map has.
    pub fn written(&self) -> Functions {
        locked(&self.inner.written).clone()
    }

    /// What a writer is to be told it may call: each function of the host as it is called,
    /// with its description, one to a line. Empty when the REPL was given none.
    ///
    /// It is the part of a writer's instructions that only the REPL knows:
    /// `format!("Where the function runs, it may call:\n{}", repl.describe())`.
    pub fn describe(&self) -> String {
        let hosts = self.inner.hosts();
        let callable = hosts.iter().filter(|(name, _)| is_identifier(name));
        let lines: Vec<String> = callable.map(|(_, host)| format!("{}: {}", host.shown, host.description)).collect();
        lines.join("\n")
    }

    /// The names of the functions of the host, in the order they were given.
    pub fn functions(&self) -> Vec<String> {
        self.inner.hosts().keys().cloned().collect()
    }

    /// Define one function in the REPL and put its name on the list of tools. Nothing is run.
    ///
    /// `name` is the function's name, `description` one sentence for a menu (what choosing
    /// the function does), and `code` the function's source: one `def` of that name and
    /// nothing beside it. A parameter is given the request's answer or input of the same name.
    /// A fence round the source, as a language model writes code, is taken off.
    ///
    /// It returns what was done, in a sentence. It is refused, with the reason as a tool's
    /// failure ([`Error::Tool`]) and nothing changed, if the source is not one function of
    /// that name, the name is a function's of the host, or the function cannot be defined.
    ///
    /// This is `define` with no run: the only names taken are the REPL's own. A tool that
    /// writes calls [`Repl::define_in`], which knows the map of the run it is in.
    pub fn define(&self, name: &str, description: &str, code: &str) -> Result<String> {
        self.defining(None, name, description, code)
    }

    /// [`Repl::define`] from inside a tool: for the run the tool is in. A name that a tool on
    /// the run's map has is refused, as is the name of an input of the request, and the change
    /// is recorded in the trajectory, on the request whose tool made it:
    /// `{"by": "repl", "did": "define", "name": name}`.
    ///
    /// Where the module that is running is a copy ([`crate::Rdm::copy`]) of the one this REPL
    /// was given to, the function is written into the copy's own REPL, which is where the
    /// copy's next request looks. So it is for [`Repl::remove_in`], [`Repl::show_in`] and a
    /// writer. What is given no call ([`Repl::define`], [`Repl::execute`]) acts on the REPL
    /// it is called on.
    ///
    /// ```
    /// use rdm::repl::Repl;
    /// use rdm::{Error, Rdm, Signature, Tool, inputs};
    /// use serde_json::json;
    /// use systemone::{Answer, Scripted};
    ///
    /// let repl = Repl::new();
    /// let writing = repl.clone();
    /// let write = Tool::new("write", "Have a new tool written in code.", move |call| {
    ///     let (name, code): (String, String) = (call.arg("name")?, call.arg("code")?);
    ///     // What is refused is said, and the run goes on: whoever writes can write again.
    ///     match writing.define_in(call, &name, "Say hello.", &code) {
    ///         Err(Error::Tool(why)) => Ok(format!("Nothing was put on the list. {why}")),
    ///         done => done,
    ///     }
    /// });
    /// let operate = Signature::new("Greet.").input("name").input("code").input("observation").tool("next", "What next?");
    /// let model = Scripted::new(|request| {
    ///     let labels = request.questions["next"].labels();
    ///     let next = if request.state["inputs"]["observation"] == "" { "write" } else { "finish" };
    ///     [("next", Answer::chose(next, labels.iter().map(String::as_str)))]
    /// });
    /// let rdm = Rdm::new(operate).tool(write).source(repl.clone()).model(model);
    ///
    /// let result = rdm.call(inputs! { "name" => "hello", "code" => "def hello():\n    return 'hello'" }).unwrap();
    /// assert_eq!(result.trajectory[0].changes, Some(vec![json!({"by": "repl", "did": "define", "name": "hello"})]));
    ///
    /// let refused = rdm.call(inputs! { "name" => "write", "code" => "def write():\n    return 1" }).unwrap();
    /// assert_eq!(
    ///     refused.steps("next")[0].1,
    ///     "Nothing was put on the list. `write` is already the name of a tool. Give the function another name."
    /// );
    /// ```
    pub fn define_in(&self, call: &Call, name: &str, description: &str, code: &str) -> Result<String> {
        self.of(call).defining(Some(call), name, description, code)
    }

    /// The REPL of the run a tool is in. A tool holds the REPL it was made with, and a copy
    /// of its module ([`crate::Rdm::copy`]) has the same tool and a copy of the REPL: what
    /// the tool writes in a run of the copy belongs in the copy's, where the copy's next
    /// request will look for it. So this is the REPL the module that is running has for a
    /// source, where that is this one or a copy of it, and otherwise this one.
    pub(crate) fn of(&self, call: &Call) -> Repl {
        let sources = call.active.module.sources();
        let related = sources
            .iter()
            .filter_map(|source| source.as_any()?.downcast_ref::<Repl>())
            .filter(|repl| Arc::ptr_eq(&repl.inner.family, &self.inner.family));
        let (mut this, mut copy) = (false, None);
        for repl in related {
            this |= Arc::ptr_eq(&repl.inner, &self.inner);
            copy.get_or_insert(repl);
        }
        match copy {
            Some(copy) if !this => copy.clone(),
            _ => self.clone(),
        }
    }

    fn defining(&self, call: Option<&Call>, name: &str, description: &str, code: &str) -> Result<String> {
        let (name, description) = (name.trim(), description.trim());
        // The names on the map of the run this is written in, where there is one.
        let taken = call.map(|call| self.inner.others(call)).unwrap_or_default();
        // Each input is a variable where the function runs, under its name.
        let inputs: Vec<String> = call.map(|call| call.shown().keys().cloned().collect()).unwrap_or_default();
        let code = self.vetted(name, code, &taken, &inputs)?;
        let failed = |why: &dyn fmt::Display| Error::Tool(format!("What was written failed. {why}"));
        let mut turn = self.inner.enter(call, None).map_err(|busy| failed(&busy))?;
        match self.inner.feed(&mut turn, &code, &Map::new(), None) {
            Ok(_) => {}
            Err(Stop::Ended(error)) => return Err(error),
            Err(Stop::Failed(why)) => {
                // The function this name had is the one it has still.
                let before = locked(&self.inner.written).get(name).map(|entry| entry.code.clone());
                if let Some(before) = before {
                    let _ = self.inner.feed(&mut turn, &before, &Map::new(), None);
                }
                return Err(failed(&why));
            }
        }
        drop(turn);
        locked(&self.inner.written).insert(name.to_owned(), Written::new(description, code));
        locked(&self.inner.taken_off).retain(|gone| gone.name != name);
        if let Some(call) = call {
            call.changed(self.change("define", name));
        }
        // Said plainly, because a description of what a function does reads as a report that it was done.
        Ok(format!(
            "`{name}` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: {description}"
        ))
    }

    /// The names on the map of the run a tool is in that a function of this REPL cannot take.
    #[cfg(feature = "rlm")]
    pub(crate) fn others(&self, call: &Call) -> Vec<String> {
        self.inner.others(call)
    }

    /// A source as it would be defined, without the fence a language model puts round code,
    /// or what it is refused for before anything is run: `taken` are the names on the map
    /// that are not this REPL's own, and `inputs` those of the request's inputs.
    pub(crate) fn vetted(&self, name: &str, code: &str, taken: &[String], inputs: &[String]) -> Result<String> {
        let refused = |why: String| Err(Error::Tool(why));
        if !is_identifier(name) {
            return refused(format!("`{name}` cannot be the name of a function."));
        }
        let not_python = |why: String| Error::Tool(format!("What was written is not Python. {why}"));
        // The fence a language model puts round code.
        let code = strip_code_fences(code).map_err(not_python)?;
        let one = match check_def(&code) {
            Ok(function) => function == name,
            Err(Unfit::NotPython(why)) => return Err(not_python(why)),
            Err(Unfit::NotOneDef) => false,
        };
        if !one {
            // Writing only defines: no statement, decorator or default that would run when it is written.
            return refused(format!(
                "The source must be one `def {name}(...)` and nothing beside it, with no decorator, and plain values as \
                 defaults: writing a function defines it and runs nothing."
            ));
        }
        if USED.contains(&name) {
            return refused(format!("`{name}` is a name the REPL uses. Give the function another name."));
        }
        if name == FINISH || self.inner.hosts().contains_key(name) || taken.iter().any(|taken| taken == name) {
            return refused(format!("`{name}` is already the name of a tool. Give the function another name."));
        }
        if inputs.iter().any(|input| input == name) {
            return refused(format!(
                "`{name}` is already the name of an input of the request. Give the function another name."
            ));
        }
        Ok(code)
    }

    fn change(&self, did: &str, name: &str) -> Json {
        json!({"by": self.name(), "did": did, "name": name})
    }

    /// Take a written function off the list of tools, and forget its source. Returns what
    /// was done, in a sentence.
    pub fn remove(&self, name: &str) -> String {
        self.removing(None, name)
    }

    /// [`Repl::remove`] from inside a tool: the change is recorded in the trajectory, on the
    /// request whose tool made it. The name is free from then on: the same tool may write a
    /// function under it again at once, though the run's map holds the name until the run's
    /// next request.
    pub fn remove_in(&self, call: &Call, name: &str) -> String {
        self.of(call).removing(Some(call), name)
    }

    fn removing(&self, call: Option<&Call>, name: &str) -> String {
        let mut written = locked(&self.inner.written);
        if written.shift_remove(name).is_none() {
            return format!("There is no function `{name}`: the functions are {}.", listed(written.keys()));
        }
        drop(written);
        if let Some(call) = call {
            let mut taken_off = locked(&self.inner.taken_off);
            taken_off.retain(|gone| gone.run.strong_count() > 0);
            taken_off.push(TakenOff {
                run: Arc::downgrade(&call.active.run),
                index: call.active.index,
                name: name.to_owned(),
            });
            drop(taken_off);
            call.changed(self.change("remove", name));
        }
        format!("`{name}` is no longer on the list.")
    }

    /// Every function written so far, with its description and its source.
    ///
    /// For whoever writes: what there is already, so that nothing is written twice, and what
    /// failed, so that it can be written again.
    pub fn show(&self) -> String {
        self.showing(None)
    }

    /// [`Repl::show`] from inside a tool: with the other names on the run's map, which a
    /// function cannot take.
    pub fn show_in(&self, call: &Call) -> String {
        self.of(call).showing(Some(call))
    }

    fn showing(&self, call: Option<&Call>) -> String {
        let written = self.written();
        let entry = |(name, entry): (&String, &Written)| {
            let failed = entry
                .failed
                .as_ref()
                .map(|error| format!("\nIt failed and is off the list: {error}"))
                .unwrap_or_default();
            format!("{name}: {}{failed}\n{}", entry.description, entry.code)
        };
        let mut lines: Vec<String> = written.iter().map(entry).collect();
        let others = call.map(|call| self.inner.others(call)).unwrap_or_default();
        if !others.is_empty() {
            lines.push(format!(
                "Other names on the list, which a function cannot take: {}",
                others.join(", ")
            ));
        }
        if lines.is_empty() {
            return "No function has been written.".to_owned();
        }
        lines.join("\n\n")
    }

    /// Run code in the REPL and return what it printed.
    ///
    /// What the code leaves behind stays: a variable, a function. Code that fails is a
    /// tool's failure ([`Error::Tool`]) that says what it failed with, and so is code asked
    /// for by code this REPL is running, through a function of the host: a REPL runs one
    /// thing at a time.
    pub fn execute(&self, code: &str) -> Result<String> {
        self.execute_with(code, &Map::new())
    }

    /// [`Repl::execute`], with variables the code can read: each a name and plain data.
    pub fn execute_with(&self, code: &str, variables: &Map<String, Json>) -> Result<String> {
        self.execute_for(None, code, variables)
    }

    /// [`Repl::execute_with`], for a request: a function of the host that is a tool of a
    /// module, or that asks the module again, then has the request it does that for.
    pub(crate) fn execute_for(&self, call: Option<&Call>, code: &str, variables: &Map<String, Json>) -> Result<String> {
        let mut turn = self.inner.enter(call, None).map_err(Error::Tool)?;
        match self.inner.feed(&mut turn, code, variables, call) {
            Ok(printed) => Ok(printed.trim().to_owned()),
            Err(Stop::Failed(why)) => Err(Error::Tool(why)),
            Err(Stop::Ended(error)) => Err(error),
        }
    }

    /// [`Repl::execute_for`], keeping what the code printed before it failed: what it printed, and why
    /// it failed where it did. A run that is ended is an error, as there.
    pub(crate) fn execute_keeping_for(
        &self,
        call: Option<&Call>,
        code: &str,
        variables: &Map<String, Json>,
    ) -> Result<(String, Option<String>)> {
        let mut turn = self.inner.enter(call, None).map_err(Error::Tool)?;
        match self.inner.cell(&mut turn, code, variables, call) {
            (Ok(printed), _) => Ok((printed.trim().to_owned(), None)),
            (Err(Stop::Failed(why)), cell) => Ok((cell.printed.trim().to_owned(), Some(why))),
            (Err(Stop::Ended(error)), _) => Err(error),
        }
    }

    /// [`Repl::execute_with`], as data: what the code printed, and how it ended, whether it ran
    /// to its end or not. What a record of a run keeps of a piece of code, beside the code and
    /// what it was answered, and what a replay of the run sets beside it.
    ///
    /// A REPL that is running the code that asks, and a function of the host that ends the run,
    /// are errors here as they are for [`Repl::execute`].
    ///
    /// ```
    /// use rdm::repl::{Ending, Repl};
    ///
    /// let repl = Repl::new();
    /// let cell = repl.execute_cell("print('half')\nx = 1 / 0", &Default::default()).unwrap();
    /// assert_eq!(cell.printed, "half");
    /// assert_eq!(cell.ending, Ending::Raised { error: "ZeroDivisionError: division by zero".to_owned() });
    /// assert_eq!(repl.execute_cell("open('notes.txt')", &Default::default()).unwrap().ending, Ending::Raised {
    ///     error: "PermissionError: the REPL has no access to the operating system".to_owned()
    /// });
    /// ```
    pub fn execute_cell(&self, code: &str, variables: &Map<String, Json>) -> Result<Cell> {
        self.execute_cell_for(None, code, variables)
    }

    /// [`Repl::execute_cell`], for a request, as [`Repl::execute_for`] is.
    pub(crate) fn execute_cell_for(&self, call: Option<&Call>, code: &str, variables: &Map<String, Json>) -> Result<Cell> {
        let mut turn = self.inner.enter(call, None).map_err(Error::Tool)?;
        match self.inner.cell(&mut turn, code, variables, call) {
            (Err(Stop::Ended(error)), _) => Err(error),
            (_, cell) => Ok(cell),
        }
    }

    /// A tool that writes: `drafts` gives a function (a language model's, as a rule), and
    /// the tool defines it in this REPL, for the run it is in. What the tool returns is what
    /// [`Repl::define_in`] said, or where the function was refused,
    /// `Nothing was put on the list. {why}`: either is the next request's observation, so
    /// whoever writes can write again.
    ///
    /// ```
    /// use rdm::repl::{Draft, Repl};
    /// use rdm::{Rdm, Signature, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let repl = Repl::new();
    /// // In a program this asks a language model, with the request the call holds.
    /// let write = repl.writer("write", "Have a new tool written in code.", |_call| {
    ///     Ok(Draft::new("hello", "Say hello.", "```python\ndef hello():\n    return 'hello'\n```"))
    /// });
    ///
    /// let operate = Signature::new("Greet.")
    ///     .input("task")
    ///     .input("observation")
    ///     .tool("next", "What should happen next for `inputs.task`?");
    /// let model = Scripted::new(|request| {
    ///     let labels = request.questions["next"].labels();
    ///     let next = ["hello", "write"].into_iter().find(|name| labels.iter().any(|label| label == name)).unwrap_or("finish");
    ///     let next = if request.state["inputs"]["observation"] == "hello" { "finish" } else { next };
    ///     [("next", Answer::chose(next, labels.iter().map(String::as_str)))]
    /// });
    /// let result = Rdm::new(operate).tool(write).source(repl).model(model).call(inputs! { "task" => "Say hello." }).unwrap();
    /// assert_eq!(result.pointed("next"), ["write", "hello", "finish"]);
    /// assert_eq!(result.steps("next")[1].1, "hello");
    /// ```
    pub fn writer<F>(&self, name: impl Into<String>, description: impl Into<String>, drafts: F) -> Tool
    where
        F: Fn(&Call) -> Result<Draft> + Send + Sync + 'static,
    {
        let repl = self.clone();
        Tool::new(name, description, move |call| {
            let draft = drafts(call)?;
            match repl.define_in(call, &draft.name, &draft.description, &draft.code) {
                Err(Error::Tool(why)) => Ok(format!("Nothing was put on the list. {why}")),
                done => done,
            }
        })
    }

    /// A tool that writes, made of a tool that gives the function: what `gives` returns is read
    /// as a [`Draft`] (an object with a `name`, a `description` and a `code`) and defined in this
    /// REPL, as [`Repl::writer`] does it. The tool that writes has the name and the description
    /// of the one it is made of, and that one runs for the same request: it reads what it reads
    /// from that request, and the tokens it reports are that request's.
    ///
    /// So a signature a language model completes, made a tool, is a writer: with the `lm`
    /// helper, `repl.writer_from(&Generate::parse("task, house: dict -> name, description, code")?.lm(lm).tool("write", "Have a new tool written in code."))`.
    ///
    /// ```
    /// use rdm::repl::Repl;
    /// use rdm::{Rdm, Signature, Tool, inputs};
    /// use serde_json::json;
    /// use systemone::{Answer, Scripted};
    ///
    /// // Something that gives a function as data. In a program it asks a language model.
    /// let gives = Tool::new("write", "Have a new tool written in code.", |call| {
    ///     let task: String = call.arg("task")?;
    ///     Ok(json!({"name": "hello", "description": task, "code": "def hello():\n    return 'hello'"}))
    /// });
    /// let repl = Repl::new();
    /// let write = repl.writer_from(&gives);
    /// assert_eq!((write.name(), write.description()), ("write", "Have a new tool written in code."));
    ///
    /// let operate = Signature::new("Greet.")
    ///     .input("task")
    ///     .input("observation")
    ///     .tool("next", "What should happen next for `inputs.task`?");
    /// let model = Scripted::new(|request| {
    ///     let labels = request.questions["next"].labels();
    ///     let next = match request.state["inputs"]["observation"].as_str() {
    ///         Some("") => "write",
    ///         Some("hello") => "finish",
    ///         _ => "hello",
    ///     };
    ///     [("next", Answer::chose(next, labels.iter().map(String::as_str)))]
    /// });
    /// let result = Rdm::new(operate).tool(write).source(repl.clone()).model(model).call(inputs! { "task" => "Say hello." }).unwrap();
    /// assert_eq!(result.pointed("next"), ["write", "hello", "finish"]);
    /// assert_eq!(repl.written()["hello"].description, "Say hello.");
    /// ```
    pub fn writer_from(&self, gives: &Tool) -> Tool {
        let gives = gives.clone();
        let (name, description) = (gives.name().to_owned(), gives.description().to_owned());
        self.writer(name, description, move |call| Draft::read(&gives.call(call)?))
    }

    /// Another REPL with the same functions: those of the host, shared, and those written,
    /// its own from here on. Nothing else is shared: an interpreter is made for it, and has
    /// the sources defined in it, when the copy is first used.
    ///
    /// It is what a copy of a module is given ([`crate::Rdm::copy`]), so that a copy does not
    /// write into the original's: a writer made of this REPL writes into the copy when it runs
    /// for the copy of the module ([`Repl::define_in`]).
    pub fn copy(&self) -> Repl {
        let family = self.inner.family.clone();
        let copied = Repl::made(self.name(), self.inner.hosts(), self.inner.made_as(), self.written(), family);
        locked(&self.inner.copies).push(Arc::downgrade(&copied.inner));
        copied
    }

    /// The copies made of this REPL that are still held, the first made first: where a copy
    /// of a module is, its REPL is the last of these.
    pub fn copies(&self) -> Vec<Repl> {
        let mut copies = locked(&self.inner.copies);
        copies.retain(|copy| copy.strong_count() > 0);
        copies.iter().filter_map(Weak::upgrade).map(|inner| Repl { inner }).collect()
    }

    /// The same REPL, answered from a run's events in place of the world: each function of
    /// the host that is a closure over its arguments, or a tool, gives what it gave then
    /// (`answered`), what the code draws at random is what it drew then (`drew`), and the
    /// clock reads what it read then (`clocked`).
    /// Nothing the functions would have done is done.
    ///
    /// An answer is taken by the function and the arguments it is given, in the order they
    /// were given: the same code, given the same answers, asks the same again. A function asked
    /// what it was not asked then, or more often, and randomness drawn or a clock read past
    /// what was, end the run with [`systemone::Error::Unrecorded`]: the run has gone another way than
    /// the one recorded, and nothing is answered that was not. Its `ran` events, set beside
    /// those recorded, say where.
    ///
    /// A function that asks the module again (`decide`, `choose`, [`Repl::asking`]) or ends the
    /// code ([`Repl::ending`]) runs as it is: what it asks is a sub-call, which the module's
    /// replayed decision model answers ([`crate::record::replay`]), and a language model asked
    /// by the code answers from a [`crate::lm::ReplayLm`]. The written functions are this
    /// REPL's, as they are now.
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    ///
    /// use rdm::record::{Event, replay};
    /// use rdm::repl::Repl;
    /// use rdm::{Rdm, Shared, Signature, inputs};
    /// use systemone::{Answer, Scripted};
    ///
    /// let heat = Shared::new(21);
    /// let (now, down) = (heat.clone(), heat.clone());
    /// let repl = Repl::new()
    ///     .function("heat", "How warm it is.", move |_| Ok(now.get()))
    ///     .function("cooler", "One degree cooler.", move |_| Ok(down.update(|heat| *heat -= 1)));
    /// repl.define("cool_to", "Cool to 18 degrees.", "def cool_to():\n    while heat() > 18:\n        cooler()\n    return heat()").unwrap();
    /// let operate = Signature::new("Operate the house.").input("task").tool("next", "What next?");
    /// let model = Scripted::new(|request| {
    ///     let next = if request.state["inputs"]["observation"].is_null() { "cool_to" } else { "finish" };
    ///     [("next", Answer::chose(next, request.questions["next"].labels().iter().map(String::as_str)))]
    /// })
    /// .named("jev");
    /// let events = Arc::new(Mutex::new(Vec::new()));
    /// let kept = events.clone();
    /// let rdm = Rdm::new(operate.clone()).source(repl.clone()).model(model).record(move |event: &Event| {
    ///     kept.lock().unwrap().push(event.clone());
    ///     Ok(())
    /// });
    /// let live = rdm.call(inputs! { "task" => "Cool the house." }).unwrap();
    /// assert_eq!(heat.get(), 18);
    ///
    /// // The house is warm again, and the run is replayed: the code is answered as it was, and nothing is cooled.
    /// heat.set(25);
    /// let events = events.lock().unwrap().clone();
    /// let again = Rdm::new(operate).source(repl.replaying(&events)).model(replay("jev", &events));
    /// assert_eq!(again.call(inputs! { "task" => "Cool the house." }).unwrap().trajectory, live.trajectory);
    /// assert_eq!(heat.get(), 25);
    /// ```
    pub fn replaying<'a>(&self, events: impl IntoIterator<Item = &'a Event>) -> Repl {
        // Each answer by the function and what it was given; one kept without what it was
        // given, by the function alone. One not kept at all holds its place, and is refused.
        let (mut answers, mut drawn, mut clocked) = (
            HashMap::<(String, Option<String>), VecDeque<_>>::new(),
            VecDeque::new(),
            VecDeque::new(),
        );
        for event in events {
            match event {
                Event::Answered {
                    function,
                    args,
                    value,
                    error,
                    ..
                } => {
                    let answered = match (value, error) {
                        (Some(value), _) => Some(Ok(value.clone())),
                        (None, Some(error)) => Some(Err(error.clone())),
                        (None, None) => None,
                    };
                    let given = args.as_ref().map(Json::to_string);
                    answers.entry((function.clone(), given)).or_default().push_back(answered);
                }
                Event::Drew { bytes, .. } => drawn.push_back(unhex(bytes)),
                Event::Clocked { seconds, .. } => clocked.push_back(*seconds),
                _ => {}
            }
        }
        let answers = Arc::new(Mutex::new(answers));
        let mut hosts = (*self.inner.hosts()).clone();
        for (name, host) in &mut hosts {
            if !matches!(host.doing, Doing::Function(_) | Doing::Tool(_)) {
                continue;
            }
            let (name, answers) = (name.clone(), answers.clone());
            host.doing = Doing::Function(Arc::new(move |args: &Args| {
                let given = args.written().to_string();
                let answered = {
                    let mut answers = locked(&answers);
                    let mut next = |given: Option<String>| answers.get_mut(&(name.clone(), given)).and_then(VecDeque::pop_front);
                    next(Some(given.clone())).or_else(|| next(None))
                };
                let unrecorded = |why: String| Err(Error::Model(systemone::Error::Unrecorded(why)));
                match answered {
                    Some(Some(Ok(value))) => Ok(value),
                    Some(Some(Err(error))) => Err(Error::Tool(error)),
                    Some(None) => unrecorded(format!(
                        "What `{name}` answered {given} was not kept in the run replayed, so it cannot be answered again."
                    )),
                    None => unrecorded(format!(
                        "`{name}` was not answered {given} in the run replayed, or not this many times: the run has gone another way."
                    )),
                }
            }));
        }
        let (drawn, clocked) = (Mutex::new(drawn), Mutex::new(clocked));
        let replayed = move |count: usize| match locked(&drawn).pop_front() {
            Some(Some(bytes)) if bytes.len() == count => Ok(bytes),
            _ => Err(Error::Model(systemone::Error::Unrecorded(format!(
                "{count} bytes of randomness were not drawn here in the run replayed: the run has gone another way."
            )))),
        };
        Repl::made(
            self.name(),
            Arc::new(hosts),
            (
                self.inner.limits(),
                Outside {
                    randomness: Arc::new(replayed),
                    clock: Arc::new(move || {
                        locked(&clocked).pop_front().ok_or_else(|| {
                            Error::Model(systemone::Error::Unrecorded(
                                "The clock was not read here in the run replayed: the run has gone another way.".to_owned(),
                            ))
                        })
                    }),
                },
            ),
            self.written(),
            Arc::default(),
        )
    }

    /// A new environment: another REPL with the same functions of the host and the same
    /// limits, isolated from this one. A function written in one does not exist in another.
    ///
    /// To route to it, put a module that runs in it on a map under its name:
    /// `Tool::module("heating", "Anything about the temperature.", &Rdm::new(signature).source(heating))`.
    pub fn spawn(&self, name: impl Into<String>) -> Repl {
        Repl::made(
            name.into(),
            self.inner.hosts(),
            self.inner.made_as(),
            Functions::new(),
            Arc::default(),
        )
    }

    /// Run a written function for the request that pointed at it.
    ///
    /// It is given the parameters it takes, by name from the request's answers, then its
    /// inputs. One that names neither is an observation, since a language model named it and
    /// can write the function again.
    fn run(&self, name: &str, call: &Call) -> Result<String> {
        let entry = locked(&self.inner.written).get(name).cloned();
        let Some(entry) = entry else {
            let written = locked(&self.inner.written);
            return Err(Error::Tool(format!(
                "There is no function `{name}`: the functions are {}.",
                listed(written.keys())
            )));
        };
        let (mut arguments, mut missing) = (Map::new(), Vec::new());
        for (parameter, required) in takes_of(&entry.code, name) {
            if let Some(answer) = call.answer_value(&parameter) {
                arguments.insert(parameter, answer);
            } else if let Some(input) = call.shown().get(&parameter) {
                arguments.insert(parameter, input.clone());
            } else if required {
                missing.push(parameter);
            }
        }
        if !missing.is_empty() {
            return Ok(format!(
                "`{name}` was not run: it takes {}, which name no answer or input of the request. A parameter is filled by name from {}.",
                listed(&missing),
                sorted(call.names())
            ));
        }
        // The request's inputs are variables where the function runs. A function keeps its
        // name: an input that has the name of one is not a variable.
        let hosts = self.inner.hosts();
        let functions = locked(&self.inner.written);
        let free = |name: &String| !hosts.contains_key(name) && !functions.contains_key(name) && !USED.contains(&name.as_str());
        let mut variables: Map<String, Json> = call
            .shown()
            .iter()
            .filter(|(name, _)| free(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        drop(functions);
        let code = if arguments.is_empty() {
            format!("print({name}())")
        } else {
            variables.insert(ARGUMENTS.to_owned(), Json::Object(arguments));
            format!("print({name}(**{ARGUMENTS}))")
        };
        let mut turn = self.inner.enter(Some(call), Pointed::by(call)).map_err(Error::Tool)?;
        let repl = self.inner.session(&mut turn);
        let ran = match self.inner.failed(name) {
            // It could not be defined when the REPL started, and that is what it failed with.
            Some(why) => {
                turn.repl = Some(repl);
                Err(Stop::Failed(why))
            }
            None => self.inner.fed(&mut turn, repl, &code, &variables, Some(call)),
        };
        drop(turn);
        match ran {
            Ok(printed) => Ok(printed.trim().to_owned()),
            Err(Stop::Ended(error)) => Err(error),
            Err(Stop::Failed(why)) => {
                // A function that fails would be pointed at again: it comes off the map, and the error is
                // what is observed. Its source is kept, marked, for whoever writes it again.
                let mut written = locked(&self.inner.written);
                if let Some(entry) = written.get_mut(name) {
                    entry.failed = Some(why.clone());
                    drop(written);
                    call.changed(self.change("fail", name));
                }
                Ok(format!("`{name}` failed and was taken off the list. {why}"))
            }
        }
    }
}

impl fmt::Debug for Repl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Repl")
            .field("name", &self.name())
            .field("functions", &self.functions())
            .field("written", &locked(&self.inner.written).keys().collect::<Vec<_>>())
            .finish()
    }
}

/// What goes on a map: a tool for every function written that has not failed.
impl Source for Repl {
    fn offers(&self) -> Vec<Tool> {
        let written = locked(&self.inner.written);
        let working = written.iter().filter(|(_, entry)| entry.failed.is_none());
        let runs = |(name, entry): (&String, &Written)| {
            let (repl, function) = (self.clone(), name.clone());
            Tool::new(name.clone(), entry.description.clone(), move |call| repl.run(&function, call))
        };
        working.map(runs).collect()
    }

    fn name(&self) -> String {
        "Repl".to_owned()
    }

    /// A copy of a module has a REPL of its own: see [`Repl::copy`].
    fn copy(&self) -> Option<Arc<dyn Source>> {
        Some(Arc::new(Repl::copy(self)))
    }

    /// So that a tool that holds a REPL finds the one the module running was given.
    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// Why code did not run to its end.
enum Stop {
    /// It failed, with this.
    Failed(String),
    /// The run it was part of is over: a function of the host said so.
    Ended(Error),
}

impl Inner {
    fn gate(&self) -> MutexGuard<'_, Gate> {
        locked(&self.gate)
    }

    fn hosts(&self) -> Arc<IndexMap<String, Host>> {
        locked(&self.hosts).clone()
    }

    fn limits(&self) -> Limits {
        locked(&self.limits).clone()
    }

    fn outside(&self) -> Outside {
        locked(&self.outside).clone()
    }

    /// What a copy of this REPL, or another made from it, is made with besides its functions.
    fn made_as(&self) -> (Limits, Outside) {
        (self.limits(), self.outside())
    }

    fn failed(&self, name: &str) -> Option<String> {
        locked(&self.written).get(name).and_then(|entry| entry.failed.clone())
    }

    /// The names on the map of the run a tool is in that are not this REPL's: those a
    /// function cannot take. A function of this REPL's is not one, nor is one a tool of this
    /// same request took away, which the map holds until the run's next request.
    fn others(&self, call: &Call) -> Vec<String> {
        let written = locked(&self.written);
        let taken_off = locked(&self.taken_off);
        let gone = |name: &String| {
            let here = |gone: &&TakenOff| gone.index == call.active.index && gone.name == *name;
            taken_off
                .iter()
                .filter(here)
                .any(|gone| gone.run.upgrade().is_some_and(|run| Arc::ptr_eq(&run, &call.active.run)))
        };
        let names = call.tools().names();
        names
            .into_iter()
            .filter(|name| !written.contains_key(name) && !gone(name))
            .collect()
    }

    /// Wait for the REPL and take it. Refused, with the reason, for code the REPL is itself
    /// running: it would wait for itself. `call` is the request that asks, where one does, and
    /// `pointed` the request the REPL is then running a function for.
    fn enter(&self, call: Option<&Call>, pointed: Option<Pointed>) -> std::result::Result<Turn<'_>, String> {
        let name = || locked(&self.name).clone();
        let wait = self.limits().wait;
        // The clock is read by a call that has to wait, and by no other: where there is one
        // thread nothing waits, and a WebAssembly module with no host under it has no clock.
        let mut until = None;
        let mut gate = self.gate();
        while let Some(running) = &gate.running {
            // Code this REPL is running led here: on the same thread, or through a request the function it runs asked for.
            if running.thread == thread::current().id() || under(call, running.request.as_ref()) {
                return Err(format!(
                    "The REPL `{}` is running the code that asked for this, and runs one thing at a time: \
                     a function written here cannot be run from inside one that is running.",
                    name()
                ));
            }
            let left = until
                .get_or_insert_with(|| Instant::now() + wait)
                .saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!(
                    "The REPL `{}` runs one thing at a time, and was still running something else after {wait:?}.",
                    name()
                ));
            }
            gate = self.freed.wait_timeout(gate, left).unwrap_or_else(PoisonError::into_inner).0;
        }
        gate.running = Some(Running {
            thread: thread::current().id(),
            request: pointed,
        });
        Ok(Turn {
            inner: self,
            repl: gate.idle.take(),
            started: gate.started,
        })
    }

    /// The interpreter of a turn, made if there is none, with whatever was written before.
    fn session(&self, turn: &mut Turn<'_>) -> MontyRepl {
        match turn.repl.take() {
            Some(repl) => repl,
            None => self.start(),
        }
    }

    /// A new interpreter with every function written defined in it, those marked as failed
    /// too: another function may call one.
    fn start(&self) -> MontyRepl {
        let (hosts, limits, outside) = (self.hosts(), self.limits(), self.outside());
        let sources: Vec<(String, String)> = locked(&self.written)
            .iter()
            .map(|(name, entry)| (name.clone(), entry.code.clone()))
            .collect();
        // A source that cannot be defined is left out, and marked: it has failed.
        let mut left_out = HashSet::new();
        'again: loop {
            let mut repl = fresh(&limits);
            for (name, code) in &sources {
                if left_out.contains(name) {
                    continue;
                }
                let mut fed = feed(repl, code, &Map::new(), &hosts, None, (&limits, &outside, ""));
                if let Err(stop) = fed.out() {
                    let why = match stop {
                        Stop::Failed(why) => why,
                        Stop::Ended(error) => error.to_string(),
                    };
                    left_out.insert(name.clone());
                    if let Some(entry) = locked(&self.written).get_mut(name) {
                        entry.failed.get_or_insert(why);
                    }
                }
                match fed.repl {
                    Some(kept) => repl = kept,
                    // What it did to the interpreter is not known: the others are defined in a new one.
                    None => continue 'again,
                }
            }
            return repl;
        }
    }

    /// Run code on the interpreter of a turn.
    fn feed(
        &self,
        turn: &mut Turn<'_>,
        code: &str,
        variables: &Map<String, Json>,
        call: Option<&Call>,
    ) -> std::result::Result<String, Stop> {
        self.cell(turn, code, variables, call).0
    }

    /// [`Inner::feed`], with what the code printed and how it ended, whether it ran to its end or not.
    fn cell(
        &self,
        turn: &mut Turn<'_>,
        code: &str,
        variables: &Map<String, Json>,
        call: Option<&Call>,
    ) -> (std::result::Result<String, Stop>, Cell) {
        let repl = self.session(turn);
        self.fed_cell(turn, repl, code, variables, call)
    }

    fn fed(
        &self,
        turn: &mut Turn<'_>,
        repl: MontyRepl,
        code: &str,
        variables: &Map<String, Json>,
        call: Option<&Call>,
    ) -> std::result::Result<String, Stop> {
        self.fed_cell(turn, repl, code, variables, call).0
    }

    fn fed_cell(
        &self,
        turn: &mut Turn<'_>,
        repl: MontyRepl,
        code: &str,
        variables: &Map<String, Json>,
        call: Option<&Call>,
    ) -> (std::result::Result<String, Stop>, Cell) {
        let name = locked(&self.name).clone();
        let mut fed = feed(repl, code, variables, &self.hosts(), call, (&self.limits(), &self.outside(), &name));
        turn.repl = fed.repl.take();
        let cell = Cell {
            printed: fed.printed.trim().to_owned(),
            ending: fed.ending.clone(),
        };
        // What came of code run for a request is the run's, beside what it was answered.
        let kept = call.map_or(Ok(()), |call| {
            call.emit(|run, index| Event::Ran {
                run,
                index,
                repl: name,
                code: Some(code.to_owned()),
                printed: Some(cell.printed.clone()),
                ending: cell.ending.clone(),
            })
        });
        match kept {
            Ok(()) => (fed.out(), cell),
            Err(error) => (Err(Stop::Ended(error)), cell),
        }
    }
}

/// Whether a request was asked for by the function a REPL is running: by a request that
/// function's code made, through a tool it called, or by one under such a request.
///
/// A request another tool made is not, though the same request selected that tool and it is
/// running beside the function: it waits its turn, as anything does that the REPL is not
/// waiting for.
fn under(call: Option<&Call>, pointed: Option<&Pointed>) -> bool {
    let (Some(call), Some(pointed)) = (call, pointed) else {
        return false;
    };
    if Arc::as_ptr(&call.active.run) as usize != pointed.run {
        return false;
    }
    let state = call.active.run.lock();
    let mut at = call.active.index;
    while let Some(entry) = state.trajectory.get(at) {
        match entry.parent {
            // One level below the request that pointed: what made it is the tool it is recorded under.
            Some(parent) if parent == pointed.index => {
                return entry.depth > pointed.depth && entry.tool_name.as_deref() == Some(pointed.function.as_str());
            }
            Some(parent) => at = parent,
            None => return false,
        }
    }
    false
}

/// What feeding code to an interpreter came to: the interpreter, unless it has to be made
/// again, what the code printed or why it stopped, and how it ended.
struct Fed {
    repl: Option<MontyRepl>,
    printed: String,
    stopped: Option<Stop>,
    ending: Ending,
}

impl Fed {
    /// What the code printed, or why it stopped.
    fn out(&mut self) -> std::result::Result<String, Stop> {
        let printed = std::mem::take(&mut self.printed);
        self.stopped.take().map_or(Ok(printed), Err)
    }
}

/// How code that was run came to its end: kept with the events of a run, where it is defined.
pub use crate::record::Ending;

/// What a piece of code came to: what it printed, and how it ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// What it printed, until it ended.
    pub printed: String,
    /// How it ended.
    #[serde(flatten)]
    pub ending: Ending,
}

/// What a REPL lets written code know of the machine: its clock and its randomness, each asked
/// of the host and answered from the REPL's own source, so that a record has every reading.
fn policy() -> OsPolicy {
    OsPolicy {
        datetime: DateTimeSource::CallHost,
        random_start: RandomStart::CallHost,
        ..OsPolicy::default()
    }
}

fn fresh(limits: &Limits) -> MontyRepl {
    let limits = ResourceLimits::default()
        .max_feed_duration(limits.time)
        // Without an allocator that counts, this is not a count of what is held: it refuses
        // one value that would be larger, where Monty measures it beforehand (it does not for
        // `str.join`). It has to be set for that much.
        .max_memory(limits.memory)
        .max_recursion_depth(limits.recursion)
        .max_suspensions(limits.calls);
    MontyRepl::new("repl.py", ResourceTracker::new(limits), CompileOptions::default()).with_os_policy(policy())
}

/// Run code to its end, answering each time it stops for the host.
///
/// Monty never calls out. When code calls or reads a name it never defined, it stops and
/// says what it wants; the host answers and it goes on. Nothing is registered beforehand: the
/// functions of the host are the names this loop answers for.
fn feed(
    repl: MontyRepl,
    code: &str,
    variables: &Map<String, Json>,
    hosts: &IndexMap<String, Host>,
    call: Option<&Call>,
    (limits, outside, repl_name): (&Limits, &Outside, &str),
) -> Fed {
    // Monty reads the source into a tree and frees it, on this stack: one too deep is not given to it.
    if !shallow(code) {
        return Fed {
            repl: Some(repl),
            printed: String::new(),
            stopped: Some(Stop::Failed(TOO_DEEP.to_owned())),
            ending: Ending::Raised {
                error: TOO_DEEP.to_owned(),
            },
        };
    }
    let variables: Vec<(String, MontyObject)> = variables.iter().map(|(name, value)| (name.clone(), from_json(value))).collect();
    let mut printed = String::new();
    let mut stops = 0usize;
    let mut ended = None;
    let mut finished = false;
    let over = || {
        MontyException::runtime_error(format!(
            "the code called the functions it is given more than {} times",
            limits.calls
        ))
    };

    let mut progress = repl.feed_start(code, variables, PrintWriter::CollectString(&mut printed, Some(limits.output)));
    let outcome = loop {
        let step = match progress {
            Ok(step) => step,
            Err(error) => break Err(error),
        };
        let print = PrintWriter::CollectString(&mut printed, Some(limits.output));
        // Each time the code stops for the host is counted, whatever it stopped for. What ends
        // the code for stopping too often cannot be caught by it.
        stops += 1;
        let spent = stops > limits.calls;
        progress = match step {
            ReplProgress::Complete { repl, .. } => break Ok(repl),
            ReplProgress::NameLookup(lookup) if spent => lookup.abort(over(), print),
            ReplProgress::FunctionCall(asked) if spent => asked.abort(over(), print),
            ReplProgress::OsCall(asked) if spent => asked.abort(over(), print),
            // The code read a name it never defined (`f = cooler`). The name of a function of
            // the host is that function; any other is a NameError in the code.
            ReplProgress::NameLookup(lookup) => {
                if hosts.contains_key(&lookup.name) {
                    let function = MontyObject::function(lookup.name.clone(), None);
                    lookup.resume(NameLookupResult::Value(function), print)
                } else {
                    lookup.resume(NameLookupResult::Undefined, print)
                }
            }
            // The code called a name it never defined. Its arguments are here, and the answer is what it returns.
            ReplProgress::FunctionCall(asked) => {
                let answer = match hosts.get(&asked.function_name) {
                    None => Ok(ExtFunctionResult::NotFound(asked.function_name.clone())),
                    Some(host) => match host.call(repl_name, &asked.function_name, &asked.args, call) {
                        Ok(value) => Ok(ExtFunctionResult::Return(from_json(&value))),
                        Err(Raised::Inside(exception)) => Ok(ExtFunctionResult::Error(exception)),
                        Err(Raised::Outside(error)) => Err(Some(error)),
                        Err(Raised::Finished) => Err(None),
                    },
                };
                match answer {
                    Ok(answer) => asked.resume(answer, print),
                    Err(error) => {
                        // Either way the code stops here, and cannot catch what stops it.
                        finished = error.is_none();
                        ended = error;
                        asked.abort(MontyException::runtime_error("the run has ended"), print)
                    }
                }
            }
            ReplProgress::OsCall(asked) => match &asked.function_call {
                // Randomness: `random` seeding itself, or `os.urandom`. It is the REPL's to give.
                OsFunctionCall::Urandom(wanted) => match drawn(&*outside.randomness, wanted.size, limits, (call, repl_name)) {
                    Ok(answer) => asked.resume(answer, print),
                    Err(error) => {
                        ended = Some(error);
                        asked.abort(MontyException::runtime_error("the run has ended"), print)
                    }
                },
                // The clock: `time.time()` and every clock `time` reckons from it, `datetime.now()`, `date.today()`.
                OsFunctionCall::Time(_) | OsFunctionCall::DateTimeNow(_) | OsFunctionCall::DateToday => {
                    match read(&*outside.clock, &asked.function_call, (call, repl_name)) {
                        Ok(answer) => asked.resume(answer, print),
                        Err(error) => {
                            ended = Some(error);
                            asked.abort(MontyException::runtime_error("the run has ended"), print)
                        }
                    }
                }
                // `open()`, `os.getenv`, `time.sleep`: nothing of the machine is reached from here.
                _ => {
                    let refused = MontyException::new(
                        ExcType::PermissionError,
                        Some("the REPL has no access to the operating system".to_owned()),
                    );
                    asked.resume(ExtFunctionResult::Error(refused), print)
                }
            },
            // `await` on something of the host: no function here is one to wait for.
            ReplProgress::ResolveFutures(waiting) => {
                waiting.abort(MontyException::runtime_error("the REPL has no functions to await"), print)
            }
        };
    };

    let (repl, stopped, ending) = match (outcome, ended) {
        (Ok(repl), None) => (Some(repl), None, Ending::Ran),
        (Ok(repl), Some(error)) => ended_by(Some(repl), error),
        // The code was ended by a function that ends it: it printed what it printed, and
        // the interpreter is as that function left it.
        (Err(error), None) if finished => {
            let ReplStartError { repl, .. } = *error;
            (Some(repl), None, Ending::Finished)
        }
        (Err(error), ended) => {
            let ReplStartError { repl, error } = *error;
            // A limit on time or memory stops the interpreter in the middle of something, and
            // what it holds afterwards is not to be trusted: it is made again.
            let lost = matches!(error.exc_type(), ExcType::TimeoutError | ExcType::MemoryError);
            let repl = (!lost).then_some(repl);
            match ended {
                Some(ended) => ended_by(repl, ended),
                None => {
                    let why = said(&error, lost);
                    let ending = match error.exc_type() {
                        ExcType::TimeoutError => Ending::TimedOut,
                        ExcType::MemoryError => Ending::OutOfMemory { error: why.clone() },
                        _ => Ending::Raised { error: why.clone() },
                    };
                    (repl, Some(Stop::Failed(why)), ending)
                }
            }
        }
    };
    Fed {
        repl,
        printed,
        stopped,
        ending,
    }
}

/// The clock read for the code, from the REPL's clock, and answered as the call asked for it:
/// the instant in seconds, or that instant as a `datetime` or a `date`. A clock that fails is
/// refused, as the machine is.
fn read(clock: &Clock, asked: &OsFunctionCall, (call, repl): (Option<&Call>, &str)) -> std::result::Result<ExtFunctionResult, Error> {
    let refuse = |why: String| Ok(ExtFunctionResult::Error(MontyException::new(ExcType::PermissionError, Some(why))));
    let seconds = match clock() {
        Ok(seconds) if seconds.is_finite() => seconds,
        Ok(_) => return refuse("the clock gave no instant".to_owned()),
        Err(Error::Tool(why)) => return refuse(why),
        Err(other) => return Err(other),
    };
    if let Some(call) = call {
        call.emit(|run, index| Event::Clocked {
            run,
            index,
            repl: repl.to_owned(),
            seconds,
        })?;
    }
    let raise = || {
        let why = "the clock's instant is not a date Python can hold".to_owned();
        Ok(ExtFunctionResult::Error(MontyException::new(ExcType::OverflowError, Some(why))))
    };
    let answer = match asked {
        OsFunctionCall::DateTimeNow(zone) => {
            let offset = zone.as_ref().map_or(0, |zone| zone.offset_seconds);
            let Some((year, month, day, second, microsecond)) = civil(seconds, offset) else {
                return raise();
            };
            MontyObject::datetime(MontyDateTime {
                year,
                month,
                day,
                hour: (second / 3600) as u8,
                minute: (second / 60 % 60) as u8,
                second: (second % 60) as u8,
                microsecond,
                offset_seconds: zone.as_ref().map(|zone| zone.offset_seconds),
                timezone_name: zone.as_ref().and_then(|zone| zone.name.clone()),
            })
        }
        OsFunctionCall::DateToday => {
            let Some((year, month, day, ..)) = civil(seconds, 0) else {
                return raise();
            };
            MontyObject::date(MontyDate { year, month, day })
        }
        _ => MontyObject::float(seconds),
    };
    Ok(ExtFunctionResult::Return(answer))
}

/// An instant, in seconds since the Unix epoch, as the wall clock `offset` seconds east of UTC
/// shows it: the year, month and day, the second of the day and the microsecond. None for one
/// outside the years 1 to 9999. The REPL's zone is UTC.
fn civil(seconds: f64, offset: i32) -> Option<(i32, u8, u8, u32, u32)> {
    let whole = seconds.floor();
    let microsecond = ((seconds - whole) * 1e6).round().min(999_999.0) as u32;
    // Years 1 to 9999 are well inside what an i64 of seconds holds.
    if !(-62_135_596_800.0..=253_402_300_799.0).contains(&whole) {
        return None;
    }
    let local = whole as i64 + i64::from(offset);
    let (days, second) = (local.div_euclid(86_400), local.rem_euclid(86_400) as u32);
    // Days since 1970-01-01 as a date of the proleptic Gregorian calendar (Hinnant's `civil_from_days`).
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let of_era = shifted.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let day_of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_from_march + 2) / 5 + 1) as u8;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    } as u8;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (1..=9999).contains(&year).then_some((year as i32, month, day, second, microsecond))
}

/// Code that a function of the host ended the run under: the interpreter, the stop, and the ending.
fn ended_by(repl: Option<MontyRepl>, error: Error) -> (Option<MontyRepl>, Option<Stop>, Ending) {
    let ending = Ending::Ended { error: error.to_string() };
    (repl, Some(Stop::Ended(error)), ending)
}

/// `count` bytes of randomness for the code, from the REPL's source. Too many to hold is a
/// `MemoryError` the code can catch, and a source that fails is one too. A source that gives
/// another count ends the run: it is the host's mistake, and not the code's.
fn drawn(
    randomness: &Randomness,
    count: u64,
    limits: &Limits,
    (call, repl): (Option<&Call>, &str),
) -> std::result::Result<ExtFunctionResult, Error> {
    let raise = |kind: ExcType, why: String| Ok(ExtFunctionResult::Error(MontyException::new(kind, Some(why))));
    let Some(count) = usize::try_from(count).ok().filter(|count| *count <= limits.memory) else {
        return raise(
            ExcType::MemoryError,
            format!("{count} bytes of randomness are more than the REPL may hold"),
        );
    };
    match randomness(count) {
        Ok(bytes) if bytes.len() == count => {
            if let Some(call) = call {
                let hex = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
                call.emit(|run, index| Event::Drew {
                    run,
                    index,
                    repl: repl.to_owned(),
                    bytes: hex,
                })?;
            }
            Ok(ExtFunctionResult::Return(MontyObject::bytes(bytes)))
        }
        Ok(bytes) => Err(Error::Usage(format!(
            "The REPL's source of randomness gave {} bytes where {count} were asked for.",
            bytes.len()
        ))),
        Err(Error::Tool(why)) => raise(ExcType::RuntimeError, why),
        Err(other) => Err(other),
    }
}

/// What code failed with, in a line: the error, and where in a function it was.
fn said(error: &MontyException, lost: bool) -> String {
    let mut said = error.summary();
    match error.traceback().last() {
        Some(frame) if frame.hide_frame_name => said.push_str(&format!(" (line {})", frame.start.line)),
        Some(frame) => {
            // The line of a function, which is a line of its source. Not that of the code that called it.
            if let Some(function) = frame.frame_name.as_ref().filter(|name| !name.starts_with('<')) {
                said.push_str(&format!(" (line {} of `{function}`)", frame.start.line));
            }
        }
        None => {}
    }
    if lost {
        said.push_str(". The REPL was started again: it holds the functions that were written and nothing else.");
    }
    said
}

/// A Python value as plain data. Tuples and sets are lists; a key that is not a string is its
/// `str()`; an integer too large and whatever plain data has no word for are their `repr()`.
fn to_json(value: ObjectRef<'_>) -> Json {
    if let Some(flag) = value.as_bool() {
        return Json::Bool(flag);
    }
    if let Some(integer) = value.as_int() {
        return Json::Number(integer.into());
    }
    match value.type_name() {
        "NoneType" => return Json::Null,
        // `as_float` answers for an integer too, so the type is looked at first.
        "float" => {
            let float = value.as_float().and_then(Number::from_f64);
            return float.map_or_else(|| Json::String(value.py_repr()), Json::Number);
        }
        _ => {}
    }
    if let Some(text) = value.as_str() {
        return Json::String(text.to_owned());
    }
    if let Some(pairs) = value.pairs() {
        return Json::Object(pairs.into_iter().map(|(key, value)| (key.to_string(), to_json(value))).collect());
    }
    if let Some(items) = value.items() {
        return Json::Array(items.into_iter().map(to_json).collect());
    }
    Json::String(value.py_repr())
}

/// Plain data as a Python value: an object is a dict with its keys in the order given, an array a list.
fn from_json(value: &Json) -> MontyObject {
    match value {
        Json::Null => MontyObject::none(),
        Json::Bool(flag) => MontyObject::bool(*flag),
        Json::Number(number) => match (number.as_i64(), number.as_f64()) {
            (Some(integer), _) => MontyObject::int(integer),
            (None, Some(float)) => MontyObject::float(float),
            (None, None) => MontyObject::string(number.to_string()),
        },
        Json::String(text) => MontyObject::string(text.clone()),
        Json::Array(items) => MontyObject::list(items.iter().map(from_json)),
        Json::Object(pairs) => MontyObject::dict(
            pairs
                .iter()
                .map(|(key, value)| (MontyObject::string(key.clone()), from_json(value))),
        ),
    }
}

/// What a function takes: the parameters it names, and whether it takes more than it names.
#[derive(Clone, Debug, Default, PartialEq)]
struct Takes {
    /// Those that may be given by position, in order, each with whether it must be given.
    positional: Vec<(String, bool)>,
    /// Those that are given by keyword only.
    keyword: Vec<(String, bool)>,
    /// `*rest`: any number more by position.
    rest: bool,
    /// `**more`: any keyword.
    more: bool,
}

impl Takes {
    fn of(parameters: &Parameters) -> Self {
        let named =
            |parameter: &ruff_python_ast::ParameterWithDefault| (parameter.parameter.name.as_str().to_owned(), parameter.default.is_none());
        Self {
            positional: parameters.posonlyargs.iter().chain(&parameters.args).map(named).collect(),
            keyword: parameters.kwonlyargs.iter().map(named).collect(),
            rest: parameters.vararg.is_some(),
            more: parameters.kwarg.is_some(),
        }
    }

    /// Every parameter that has a name, with whether it must be given.
    fn named(&self) -> impl Iterator<Item = &(String, bool)> {
        self.positional.iter().chain(&self.keyword)
    }

    /// The arguments code gave, each under the name the function has for it, as Python
    /// would bind them. What Python would refuse is refused, in its words.
    fn bind(&self, function: &str, positional: Vec<Json>, keyword: Map<String, Json>) -> std::result::Result<Args, String> {
        let plural = |count: usize| if count == 1 { "" } else { "s" };
        if positional.len() > self.positional.len() && !self.rest {
            let (takes, given) = (self.positional.len(), positional.len());
            let were = if given == 1 { "was" } else { "were" };
            return Err(format!(
                "{function}() takes {takes} positional argument{} but {given} {were} given",
                plural(takes)
            ));
        }
        let (mut named, mut rest) = (Map::new(), Vec::new());
        for (place, value) in positional.into_iter().enumerate() {
            match self.positional.get(place) {
                Some((name, _)) => drop(named.insert(name.clone(), value)),
                None => rest.push(value),
            }
        }
        for (name, value) in keyword {
            if named.contains_key(&name) {
                return Err(format!("{function}() got multiple values for argument '{name}'"));
            }
            if !self.more && !self.named().any(|(known, _)| *known == name) {
                return Err(format!("{function}() got an unexpected keyword argument '{name}'"));
            }
            named.insert(name, value);
        }
        let missing: Vec<String> = self
            .named()
            .filter(|(name, required)| *required && !named.contains_key(name))
            .map(|(name, _)| format!("'{name}'"))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{function}() missing required argument{}: {}",
                plural(missing.len()),
                missing.join(", ")
            ));
        }
        Ok(Args {
            function: function.to_owned(),
            named,
            rest,
        })
    }
}

/// A function of the host as it was declared: its name, what it takes, and how a writer is shown it.
fn declared_as(declared: &str) -> (String, Takes, String) {
    let declared = declared.trim();
    let shown = if declared.contains('(') {
        declared.to_owned()
    } else {
        format!("{declared}()")
    };
    match parsed(&format!("def {shown}:\n    pass")).as_deref() {
        Ok([Stmt::FunctionDef(function)]) => (function.name.as_str().to_owned(), Takes::of(&function.parameters), shown),
        // Not a declaration Python has: the name is still a function, given whatever it is called with.
        _ => {
            let name = declared.split('(').next().unwrap_or(declared).trim().to_owned();
            let anything = Takes {
                rest: true,
                more: true,
                ..Takes::default()
            };
            (name, anything, shown)
        }
    }
}

/// Why a source is not a function to define.
#[derive(Debug, PartialEq)]
enum Unfit {
    /// It does not parse. Holds what the parser said, and the line.
    NotPython(String),
    /// It parses, and is not one plain `def`: there is a statement beside it, a decorator, an
    /// `async`, or a default that is not a plain value and so would run when the `def` does.
    NotOneDef,
}

/// How many links of one chain a source may have, each on the one before it: what it has
/// between two commas, with every bracket open round them. It is five times as deep as Monty
/// lets anything be, so nothing is refused here that would have run.
const DEEPEST_SOURCE: usize = 1000;

/// Whether what a parser makes of a source is shallow enough to be walked on a stack.
///
/// A source becomes a tree, and whatever walks the tree goes as deep on the stack as the tree
/// is, were it only to free it: a tree too deep ends the process, and no limit catches that.
/// A parser goes deep itself where a source is nested, and Monty's scan bounds that. It does
/// not go deep for a chain, which nests nothing: `x.a.a.a`, `1 + 1 + 1`, `f()()()` and
/// `x[0][0][0]` are each read in a loop, into a tree with a level for every link.
fn shallow(code: &str) -> bool {
    // A tree is no deeper than its source is long.
    code.len() <= DEEPEST_SOURCE || links(code, DEEPEST_SOURCE).is_some()
}

/// How deep what a parser makes of a source can be at most, counted from the tokens alone
/// with no tree made. None once it is past `most`.
///
/// A token that is not a name or a value is a link, of the bracket it is in. What is on either
/// side of a comma, or of the end of a line, is one thing beside another, and neither is under
/// the other: the count starts again there. What a bracket closed over lies under the links
/// around it, so the deepest of those is counted with them, and a block lies under each block
/// it is in. The count is never less than the depth of what a statement holds, and for code as
/// it is written it is not far above it.
fn links(code: &str, most: usize) -> Option<usize> {
    /// What is open at one bracket: the links since its last comma, the deepest of what
    /// was closed among them, the `lambda`s whose `:` has not come (a comma before it is
    /// between parameters, which the body is under), and whether it is an f-string, whose
    /// parts lie side by side.
    #[derive(Default)]
    struct Level {
        links: usize,
        below: usize,
        lambdas: usize,
        string: bool,
    }
    let mut levels = vec![Level::default()];
    // The blocks open, the links of every bracket open, and the deepest any token was.
    let (mut blocks, mut open, mut deepest) = (0usize, 0usize, 0usize);
    let mut lexer = lex(code, Mode::Module);
    loop {
        let kind = lexer.next_token();
        let top = levels.len() - 1;
        match kind {
            TokenKind::EndOfFile => return Some(deepest),
            TokenKind::Name
            | TokenKind::Int
            | TokenKind::Float
            | TokenKind::Complex
            | TokenKind::String
            | TokenKind::FStringMiddle
            | TokenKind::TStringMiddle
            | TokenKind::True
            | TokenKind::False
            | TokenKind::None
            | TokenKind::Ellipsis
            | TokenKind::Comment
            | TokenKind::NonLogicalNewline => {}
            TokenKind::Indent => blocks += 1,
            TokenKind::Dedent => blocks = blocks.saturating_sub(1),
            // A line has ended, and with it whatever was open on it.
            TokenKind::Newline => {
                levels.truncate(1);
                levels[0] = Level::default();
                open = 0;
            }
            TokenKind::Comma | TokenKind::Semi if levels[top].lambdas == 0 => {
                open -= levels[top].links;
                levels[top].links = 0;
                levels[top].below = 0;
            }
            TokenKind::Rpar | TokenKind::Rsqb | TokenKind::Rbrace | TokenKind::FStringEnd | TokenKind::TStringEnd if top > 0 => {
                let closed = levels.pop().unwrap_or_default();
                open -= closed.links;
                let under = &mut levels[top - 1];
                // A bracket is a level, and so is what it holds: a call's arguments, a part of an f-string.
                under.below = under.below.max(closed.links + closed.below + 1 + usize::from(under.string));
            }
            opens => {
                let string = matches!(opens, TokenKind::FStringStart | TokenKind::TStringStart);
                // What an f-string holds are its parts, one beside another: nothing there is a link.
                if !string && !levels[top].string {
                    levels[top].links += 1;
                    open += 1;
                }
                match opens {
                    TokenKind::Lambda => levels[top].lambdas += 1,
                    TokenKind::Colon => levels[top].lambdas = levels[top].lambdas.saturating_sub(1),
                    TokenKind::Lpar | TokenKind::Lsqb | TokenKind::Lbrace | TokenKind::FStringStart | TokenKind::TStringStart => {
                        levels.push(Level {
                            string,
                            ..Level::default()
                        });
                    }
                    _ => {}
                }
            }
        }
        deepest = deepest.max(blocks + open + levels.last().map_or(0, |level| level.below));
        if deepest > most {
            return None;
        }
    }
}

/// What code is refused with where it is too deep to be read: what Monty says of code that is
/// nested too far.
const TOO_DEEP: &str = "SyntaxError: Source is too deeply nested";

/// The statements of a source, if it is Python. Nothing is run: this only parses.
fn parsed(code: &str) -> std::result::Result<Vec<Stmt>, String> {
    // The parser goes as deep on the stack as the source is nested. Monty guards its own use
    // of it with this scan, which bounds how far brackets and blocks nest and not how long a
    // chain is: that is counted beside it, so that neither is overflowed on.
    if !monty::source_within_nesting_bound(code, SOURCE_SCAN_THRESHOLD) || !shallow(code) {
        return Err("The source is too deeply nested.".to_owned());
    }
    match ruff_python_parser::parse_module(code) {
        // Moved, not copied: a copy would be one more walk of the tree.
        Ok(module) => Ok(module.into_syntax().body.into_iter().collect()),
        Err(error) => {
            let start = usize::from(error.location.start()).min(code.len());
            let line = code.as_bytes()[..start].iter().filter(|byte| **byte == b'\n').count() + 1;
            Err(format!("{} (line {line})", error.error))
        }
    }
}

/// The name of the function a source defines, if it is one `def` and nothing beside it, with
/// no decorator and only plain values as defaults: defining it then runs nothing.
fn check_def(code: &str) -> std::result::Result<String, Unfit> {
    let body = parsed(code).map_err(Unfit::NotPython)?;
    let [Stmt::FunctionDef(function)] = body.as_slice() else {
        return Err(Unfit::NotOneDef);
    };
    let parameters = &function.parameters;
    let mut defaults = parameters
        .posonlyargs
        .iter()
        .chain(&parameters.args)
        .chain(&parameters.kwonlyargs)
        .filter_map(|parameter| parameter.default.as_deref());
    if function.is_async || !function.decorator_list.is_empty() || !defaults.all(plain) {
        return Err(Unfit::NotOneDef);
    }
    Ok(function.name.as_str().to_owned())
}

/// What Python's `ast.Constant` is: a number, a string, bytes, `True`, `False`, `None` or
/// `...`. As there, `-1` is not one (it is a minus applied to `1`), nor is an f-string.
fn plain(default: &Expr) -> bool {
    matches!(
        default,
        Expr::NumberLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::NoneLiteral(_)
            | Expr::EllipsisLiteral(_)
    )
}

/// The parameters of the function a source defines under a name, each with whether it must
/// be given. `*rest` and `**more` are left out: nothing is given to them.
pub(crate) fn takes_of(code: &str, name: &str) -> Vec<(String, bool)> {
    let body = parsed(code).unwrap_or_default();
    let defined = body.iter().filter_map(|statement| match statement {
        Stmt::FunctionDef(function) if function.name.as_str() == name => Some(function),
        _ => None,
    });
    let function: Option<&StmtFunctionDef> = defined.last();
    function.map_or_else(Vec::new, |function| Takes::of(&function.parameters).named().cloned().collect())
}

/// The source of the function a cell defines under a name, at its top level: the last, where
/// it defines the name more than once. None where the cell is not Python, or defines no such
/// function there.
#[cfg(feature = "rlm")]
pub(crate) fn def_source(code: &str, name: &str) -> Option<String> {
    let body = parsed(code).ok()?;
    let defined = body.iter().filter_map(|statement| match statement {
        Stmt::FunctionDef(function) if function.name.as_str() == name => Some(function),
        _ => None,
    });
    let function: &StmtFunctionDef = defined.last()?;
    let (start, end) = (usize::from(function.range.start()), usize::from(function.range.end()));
    code.get(start..end).map(|source| source.trim_end().to_owned())
}

/// Whether a name can be a function's in Python.
pub(crate) fn is_identifier(name: &str) -> bool {
    let mut letters = name.chars();
    letters.next().is_some_and(|first| first == '_' || first.is_alphabetic())
        && letters.all(|letter| letter == '_' || letter.is_alphanumeric())
}

/// The Python inside a markdown fence, or the code as it is where there is none.
pub(crate) fn strip_code_fences(code: &str) -> std::result::Result<String, String> {
    let code = code.trim();
    if !code.contains("```") {
        return Ok(code.to_owned());
    }
    // A fence round a fence.
    let mut lines: Vec<&str> = code.lines().collect();
    while lines.len() >= 2 && lines[0].trim() == "```" && lines[lines.len() - 1].trim() == "```" {
        lines.pop();
        lines.remove(0);
    }
    let code = lines.join("\n");
    let code = code.trim();
    // The first fence that opens: what is before it is not code.
    let Some(start) = code.find("```") else {
        return Ok(code.to_owned());
    };
    let Some((language, rest)) = code[start + 3..].split_once('\n') else {
        return Ok(code.to_owned());
    };
    let language = language.split_whitespace().next().unwrap_or_default().to_lowercase();
    if !["python", "py", "python3", "py3", ""].contains(&language.as_str()) {
        return Err(format!(
            "Expected Python code but got ```{language} fence. Write Python code, not {language}."
        ));
    }
    let inside = rest.find("```").map_or(rest, |end| &rest[..end]);
    Ok(inside.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::AnyNodeRef;
    use ruff_python_ast::visitor::source_order::{SourceOrderVisitor, TraversalSignal, walk_module};
    use ruff_python_parser::{ParseOptions, parse_unchecked};

    use super::*;

    /// How deep the tree of a source is, as the parser makes it, whether or not the source
    /// is Python: the parser makes a tree of what it could read.
    fn depth(code: &str) -> usize {
        #[derive(Default)]
        struct Deepest {
            now: usize,
            deepest: usize,
        }
        impl<'a> SourceOrderVisitor<'a> for Deepest {
            fn enter_node(&mut self, _: AnyNodeRef<'a>) -> TraversalSignal {
                self.now += 1;
                self.deepest = self.deepest.max(self.now);
                TraversalSignal::Traverse
            }

            fn leave_node(&mut self, _: AnyNodeRef<'a>) {
                self.now -= 1;
            }
        }
        let parsed = parse_unchecked(code, ParseOptions::from(Mode::Module));
        let mut deepest = Deepest::default();
        walk_module(&mut deepest, parsed.syntax());
        deepest.deepest
    }

    /// Where an expression can stand in a statement, and in another expression.
    const STATEMENTS: [&str; 34] = [
        "HOLE",
        "x = HOLE",
        "x = y = HOLE",
        "x: int = HOLE",
        "x += HOLE",
        "x, y = 1, HOLE",
        "x = 1; y = HOLE",
        "del x, y; z = HOLE",
        "assert 1, HOLE",
        "raise X from HOLE",
        "global a, b\nx = HOLE",
        "import a, b\nx = HOLE",
        "from a import b, c\nHOLE",
        "x = [\n 1, 2]\ny = HOLE",
        "x = (1 +\n HOLE)",
        "x = 1 + \\\n HOLE",
        "type X = HOLE",
        "def f(a, b=HOLE):\n    pass",
        "def f(a: HOLE, b) -> HOLE:\n    pass",
        "@HOLE\ndef f():\n    pass",
        "def f():\n    return HOLE",
        "def f():\n    if 1:\n        for q in r:\n            return 1, HOLE",
        "if HOLE:\n    pass\nelif 1:\n    pass",
        "if 1:\n    pass\nelif HOLE:\n    pass\nelse:\n    pass",
        "for a, b in HOLE:\n    pass",
        "while HOLE:\n    x = 1",
        "with HOLE as a, 2 as b:\n    pass",
        "with 1 as a, HOLE as b:\n    pass",
        "class C(A, HOLE):\n    pass",
        "try:\n    pass\nexcept HOLE:\n    pass",
        "match HOLE:\n    case _:\n        pass",
        "match x:\n    case 1:\n        HOLE",
        "match x:\n    case HOLE:\n        pass",
        "x = (1, HOLE",
    ];
    const EXPRESSIONS: [&str; 46] = [
        "HOLE",
        "(HOLE)",
        "[HOLE]",
        "{HOLE}",
        "{1: HOLE}",
        "{HOLE: 1}",
        "{1: 2, **HOLE}",
        "(1, HOLE)",
        "1, HOLE",
        "*HOLE, 1",
        "f(1, HOLE)",
        "f(k=HOLE)",
        "f(*HOLE)",
        "f(1, **HOLE)",
        "x.y(HOLE).z",
        "x[HOLE]",
        "x[1:HOLE]",
        "x[HOLE, 2]",
        "lambda a, b: HOLE",
        "lambda a, b: 1, HOLE",
        "lambda a, b=(HOLE): a",
        "lambda a, b=HOLE, c=1: a",
        "lambda a, b: lambda c, d: HOLE",
        "1 if c else HOLE",
        "HOLE if c else 2",
        "1 if HOLE else 2",
        "not HOLE",
        "-HOLE",
        "2 ** HOLE",
        "1 + (HOLE) + 2 + 3",
        "1 and HOLE or 2",
        "HOLE < 1 < 2",
        "(yield HOLE)",
        "(await HOLE)",
        "(x := HOLE)",
        "[y for y in z if HOLE]",
        "[HOLE for a, b in z]",
        "[y for a, b in HOLE]",
        "{k: HOLE for k, v in z}",
        "f'{HOLE}'",
        "f'{x:{HOLE}}'",
        "f'a{1}b{HOLE}c'",
        "f'{1}' f'{HOLE}'",
        "(HOLE).b.b.b.b.b.b.b.b.b.b",
        "(HOLE)(1)(2)(3)[0][0]",
        "[1, 2,\n HOLE,\n 3]",
    ];

    /// Every way a source is deep: `links` of each, one on another.
    fn chains(links: usize) -> Vec<String> {
        let after = [".a", " + 1", "()", "[0]", "(1, 2)", " | x", " @ x", " if c else x"];
        let before = ["-", "not ", "await ", "2 ** ", "lambda: ", "lambda a, b: ", "1 if c else ", "x if "];
        let mut chains: Vec<String> = after.iter().map(|link| format!("x{}", link.repeat(links))).collect();
        chains.extend(before.iter().map(|link| format!("{}x", link.repeat(links))));
        for (opens, closes) in [("(", ")"), ("[", "]"), ("{", "}"), ("f(", ")"), ("x[", "]"), ("(1, ", ")")] {
            chains.push(format!("{}x{}", opens.repeat(links), closes.repeat(links)));
        }
        chains.push(format!("{}x{}", "f'{".repeat(links), "}'".repeat(links)));
        chains.push(format!("{}x{}", "f'{x:".repeat(links), "}'".repeat(links)));
        chains
    }

    #[test]
    #[cfg_attr(
        all(target_family = "wasm", not(target_feature = "atomics")),
        ignore = "no thread can be started here"
    )]
    fn the_count_of_links_is_never_less_than_the_depth_of_the_tree() {
        // The parser goes deep where a source is nested, so this runs on a stack that lets it.
        let checked = thread::Builder::new().stack_size(256 * 1024 * 1024).spawn(|| {
            let mut sources = 0;
            for chain in chains(150) {
                for expression in EXPRESSIONS {
                    for statement in STATEMENTS {
                        let code = statement.replace("HOLE", &expression.replace("HOLE", &chain));
                        let (counted, deep) = (links(&code, usize::MAX).unwrap_or_default(), depth(&code));
                        // A level of the tree that is no token's is one of a few around a token that is counted.
                        assert!(deep <= counted + 16, "{deep} deep, and {counted} counted:\n{code}");
                        sources += 1;
                    }
                }
            }
            assert_eq!(sources, 24 * 46 * 34);
        });
        checked.unwrap().join().unwrap();
    }

    #[test]
    fn what_lies_side_by_side_is_not_counted_as_deep() {
        let beside = |one: &str, between: &str| vec![one; 5000].join(between);
        for code in [
            format!("x = [{}]", beside("-a.b(c)[0] + f'{d.e}'", ", ")),
            format!("x = {{{}}}", beside("'k': (1, [2, {'a': -3}])", ", ")),
            format!("f({})", beside("k=v.w", ", ")),
            beside("x = x.a.b + 1", "\n"),
            beside("x = x.a.b + 1", "; "),
            format!("x = ({})", beside("f'{a.b}: {c!r:>{w}}\\n'", "\n ")),
            format!("x = f'{}'", beside("{a.b}", " ")),
            format!("def f():\n{}", beside("    if a.b:\n        return c.d(e)", "\n")),
            format!("x = lambda a, b: [{}]", beside("a.b", ", ")),
        ] {
            let counted = links(&code, usize::MAX).unwrap_or_default();
            assert!(counted <= 16, "{counted} counted:\n{}", &code[..200]);
            assert!(shallow(&code));
        }
    }
}
