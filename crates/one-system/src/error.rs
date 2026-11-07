//! The errors a gateway raises itself.
//!
//! Each is a [`systemone::Error::Api`] with a status, a code and a fixed message, the same
//! whether the gateway is called in the process or over HTTP. A message never holds anything
//! from a request, a response or a credential.

use std::fmt;

use systemone::Error;

pub(crate) fn api(status: u16, code: &str, message: &str) -> Error {
    Error::Api {
        status,
        code: code.into(),
        message: message.into(),
    }
}

/// A gateway that cannot be built, or a registry that cannot be loaded.
///
/// The message is one of a fixed set and says which rule was broken. It never holds a value
/// from the file or from the environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub(crate) String);

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// What is wrong.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for Error {
    fn from(error: ConfigError) -> Self {
        Error::Invalid(error.0)
    }
}
