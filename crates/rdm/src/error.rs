//! What can go wrong in a run.

use std::fmt;

/// What a call of a module, or a tool, can fail with.
///
/// Two of the first three end the run wherever they happen, a tool's sub-calls included: a module
/// used in a way it cannot be, and a decision model that failed. The third is a tool's own
/// failure, which is not the end of anything: it becomes that tool's observation. A tool that
/// panics has failed in the same way, and what the panic said is the observation. The fourth
/// is no failure: a branch that arbitration stopped ends with it, and the run goes on.
///
/// Which of them a mistake is depends on whose it is. What the code got wrong is a usage
/// error: a name that is no answer and no input, an input or the state read as a type it is
/// not, a question with no options. What the model answered is never one: an answer read as
/// a type it is not is the tool's failure, and the run goes on.
///
/// ```
/// use rdm::{Error, Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let wait = Signature::new("Wait for a door.").input("door").noul("knock", "Is `inputs.door` shut?");
/// let model = || Scripted::new(|_| [("knock", Answer::noul(0.9))]);
///
/// // A tool that fails is an observation, and the run goes on.
/// let jammed = Tool::new("knock", "Knock.", |_| -> rdm::Result<()> { Err("the door is jammed".into()) });
/// let result = Rdm::new(wait.clone()).tool(jammed).model(model()).call(inputs! { "door" => "front" }).unwrap();
/// assert_eq!(result.observations["knock"], "Execution error in knock: the door is jammed");
///
/// // A tool that reads a name the request does not have is a module used wrongly: the call fails.
/// let misspelt = Tool::new("knock", "Knock.", |call| call.arg::<String>("dor"));
/// let error = Rdm::new(wait).tool(misspelt).model(model()).call(inputs! { "door" => "front" }).unwrap_err();
/// assert!(matches!(error, Error::Usage(_)));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The module was used in a way it cannot be: inputs that are not the signature's, a tool
    /// that reads a name the request does not have, a choice with no options, a limit below one.
    Usage(String),
    /// The decision model failed, or answered with something that is not evidence.
    Model(systemone::Error),
    /// A tool failed. The module turns it into the observation `Execution error in {name}: {error}`.
    Tool(String),
    /// What was asked for is no longer wanted: the branch it was asked for was stopped by
    /// arbitration ([`crate::Arbiter`]). Nothing more is begun for it, a request, a written
    /// reply or a tool, and what it did before it was stopped stands. It ends the branch, and
    /// not the run: the request the branch was of goes on without it.
    Stopped(String),
}

impl Error {
    /// A module used wrongly.
    pub fn usage(message: impl fmt::Display) -> Self {
        Error::Usage(message.to_string())
    }

    /// A tool's own failure: what the next request will be shown.
    pub fn tool(message: impl fmt::Display) -> Self {
        Error::Tool(message.to_string())
    }

    /// Whether this is a module used wrongly.
    pub fn is_usage(&self) -> bool {
        matches!(self, Error::Usage(_))
    }

    /// Whether this is a branch that was stopped, and not a failure.
    pub fn is_stopped(&self) -> bool {
        matches!(self, Error::Stopped(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Usage(message) | Error::Tool(message) | Error::Stopped(message) => f.write_str(message),
            Error::Model(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    /// A model's failure is shown as the model's own (`Display`), so what caused it comes next,
    /// not the model's failure again: a report that walks the causes says it once.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Model(error) => std::error::Error::source(error),
            Error::Usage(_) | Error::Tool(_) | Error::Stopped(_) => None,
        }
    }
}

/// A decision model's failure is the run's.
impl From<systemone::Error> for Error {
    fn from(error: systemone::Error) -> Self {
        Error::Model(error)
    }
}

/// A message is a tool's failure: `Err("the lock is jammed".into())`.
impl From<String> for Error {
    fn from(message: String) -> Self {
        Error::Tool(message)
    }
}

/// A message is a tool's failure: `Err("the lock is jammed".into())`.
impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Error::Tool(message.to_owned())
    }
}

/// Reading or writing JSON inside a tool is that tool's failure.
impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Tool(error.to_string())
    }
}

/// A file or a socket inside a tool is that tool's failure.
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Tool(error.to_string())
    }
}

/// Any other error is a tool's failure, unless it is one of ours that was boxed on the way.
impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        match error.downcast::<Error>() {
            Ok(ours) => *ours,
            Err(other) => match other.downcast::<systemone::Error>() {
                Ok(model) => Error::Model(*model),
                Err(other) => Error::Tool(other.to_string()),
            },
        }
    }
}

/// What a call returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Names as Python prints a list of them, which is how the messages of the reference list them.
pub(crate) fn listed<S: AsRef<str>>(names: impl IntoIterator<Item = S>) -> String {
    let quoted: Vec<String> = names.into_iter().map(|name| format!("'{}'", name.as_ref())).collect();
    format!("[{}]", quoted.join(", "))
}

/// As [`listed`], in alphabetical order.
pub(crate) fn sorted<S: AsRef<str>>(names: impl IntoIterator<Item = S>) -> String {
    let mut names: Vec<String> = names.into_iter().map(|name| name.as_ref().to_owned()).collect();
    names.sort();
    names.dedup();
    listed(names)
}
