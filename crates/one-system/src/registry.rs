//! The registry file: a gateway written down as JSON.
//!
//! The format is the Go gateway's `backends.json`, and the files that ship with it load here
//! unchanged:
//!
//! ```json
//! {
//!   "name": "privacy-demo",
//!   "selector": "local",
//!   "fallback": "local",
//!   "selection": {
//!     "questions_file": "examples/routing.questions.json",
//!     "escalate_to": "hosted",
//!     "rules": [{"question": "domain", "choice": "math_or_logic", "above": 0.5}]
//!   },
//!   "backends": [
//!     {"id": "local", "base_url": "http://127.0.0.1:8091", "model": "laya-english",
//!      "api_key_env": "LOCAL_API_KEY", "description": "short simple English text",
//!      "limits": {"max_characters": 1200, "max_non_ascii_letter_fraction": 0.02},
//!      "capabilities": {"question_types": ["choice", "score", "noul"], "max_questions": 16}},
//!     {"id": "hosted", "base_url": "https://api.typesafe.ai", "model": "jev-latest",
//!      "api_key_env": "TYPESAFE_API_KEY", "description": "other languages, long documents"}
//!   ]
//! }
//! ```
//!
//! A backend that reads pictures says so in its capabilities, by OpenRouter's name for it:
//! `"input_modalities": ["text", "image"]`. One that does not say is text only, and a request
//! with pictures is never sent to it.

use std::io::Read;
use std::net::{IpAddr, Ipv6Addr};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::Deserialize;
use systemone::{Json, Kind, Question};

use crate::client::Client;
use crate::error::ConfigError;
use crate::gateway::{Backend, Gateway, GatewayBuilder, Rule, Selection, valid_id};
use crate::schema::valid_question;
use crate::shape::{Capabilities, Limits};

/// The most a registry file, or a file of routing questions, may be: 1 MiB.
pub const MAX_CONFIG_BYTES: usize = 1 << 20;

// The file as it is written. Everything is optional here so that what is missing is reported
// by the rule it breaks, and a member nobody knows is an error at every level.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRegistry {
    name: Option<String>,
    selector: Option<String>,
    fallback: Option<String>,
    escalation_confidence: Option<f64>,
    selection: Option<RawSelection>,
    backends: Option<Vec<RawBackend>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSelection {
    questions_file: Option<String>,
    escalate_to: Option<String>,
    rules: Option<Vec<RawRule>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRule {
    question: Option<String>,
    choice: Option<String>,
    above: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBackend {
    id: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    api_key_env: Option<String>,
    description: Option<String>,
    limits: Option<RawLimits>,
    capabilities: Option<RawCapabilities>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLimits {
    max_characters: Option<i64>,
    max_non_ascii_letter_fraction: Option<f64>,
    max_questions: Option<i64>,
    max_criteria: Option<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCapabilities {
    question_types: Option<Vec<String>>,
    max_questions: Option<i64>,
    min_criteria: Option<i64>,
    max_criteria: Option<i64>,
    structured_state: Option<bool>,
    input_modalities: Option<Vec<String>>,
}

/// One backend as a registry file describes it: a System One server at a URL.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The name the gateway knows it by.
    pub id: String,
    /// Where it is, with its scheme in lower case and no `/` at the end. Plain `http` is for
    /// this machine only.
    pub base_url: String,
    /// The `model` its requests name.
    pub model: String,
    /// The name of the environment variable that holds its key. Never the key.
    pub api_key_env: String,
    /// What it is good for, in words a selector can choose by.
    pub description: String,
    /// What it would rather not be sent.
    pub limits: Option<Limits>,
    /// What it is able to answer. `None` is unknown, and passes everything.
    pub capabilities: Option<Capabilities>,
}

/// A gateway as a file describes it, read and checked, with no key in it and nothing yet
/// connected.
///
/// ```
/// use one_system::Registry;
///
/// let registry = Registry::from_json(
///     r#"{"name": "local-demo", "selector": "local", "backends": [
///         {"id": "local", "base_url": "http://127.0.0.1:8080", "model": "laya-english",
///          "api_key_env": "LOCAL_API_KEY", "description": "Laya, on this machine."}]}"#,
///     |_questions_file| Err(std::io::ErrorKind::NotFound.into()),
/// )?;
/// assert_eq!(registry.backends[0].base_url, "http://127.0.0.1:8080");
///
/// // The keys come from wherever the caller says, by the names the file gives.
/// let gateway = registry.gateway_with(|variable| (variable == "LOCAL_API_KEY").then(|| "sk-local".to_string()))?;
/// assert_eq!(gateway.selector(), "local");
/// # Ok::<(), one_system::ConfigError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Registry {
    /// The gateway's own name: the `model` that asks it to choose.
    pub name: String,
    /// The backend that decides between several.
    pub selector: String,
    /// The backend a request stays on, or goes to when the selector fails.
    pub fallback: Option<String>,
    /// How sure the selector must be for a request to leave the fallback.
    pub escalation_confidence: Option<f64>,
    /// Routing by questions about the task, with the questions file already read.
    pub selection: Option<Selection>,
    /// The backends, in the order the file gives them.
    pub backends: Vec<Entry>,
}

fn wrong<T>(message: &str) -> Result<T, ConfigError> {
    Err(ConfigError::new(message))
}

/// Read a file of at most 1 MiB as text.
fn read_bounded(path: &Path) -> std::io::Result<String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_CONFIG_BYTES {
        return Err(std::io::Error::other("the file is larger than 1 MiB"));
    }
    String::from_utf8(bytes).map_err(std::io::Error::other)
}

/// Whether a key can be sent as a bearer token.
pub(crate) fn valid_key(key: &str) -> bool {
    !key.is_empty() && !key.contains([' ', '\t', '\r', '\n'])
}

/// A positive whole number, or the rule it breaks.
fn positive(value: Option<i64>, least: i64, message: &'static str) -> Result<Option<u64>, ConfigError> {
    match value {
        Some(value) if value < least => wrong(message),
        Some(value) => Ok(Some(value as u64)),
        None => Ok(None),
    }
}

impl RawCapabilities {
    fn read(self) -> Result<Capabilities, ConfigError> {
        let mut question_types = Vec::new();
        for kind in self.question_types.unwrap_or_default() {
            let kind = match kind.as_str() {
                "choice" => Kind::Choice,
                "score" => Kind::Score,
                "noul" => Kind::Noul,
                _ => return wrong("capabilities require unique native question types"),
            };
            question_types.push(kind);
        }
        let limit = |value| positive(value, 1, "capability limits must be positive integers");
        let capabilities = Capabilities {
            question_types,
            max_questions: limit(self.max_questions)?,
            min_criteria: limit(self.min_criteria)?,
            max_criteria: limit(self.max_criteria)?,
            structured_state: self.structured_state,
            input_modalities: self.input_modalities,
        };
        match capabilities.fault() {
            Some(fault) => wrong(fault),
            None => Ok(capabilities),
        }
    }
}

impl RawLimits {
    fn read(self) -> Result<Limits, ConfigError> {
        let limits = Limits {
            max_characters: positive(self.max_characters, 1, "a backend max_characters must be a positive integer")?,
            max_non_ascii_letter_fraction: self.max_non_ascii_letter_fraction,
            max_questions: positive(self.max_questions, 1, "a backend max_questions must be a positive integer")?,
            max_criteria: positive(self.max_criteria, 2, "a backend max_criteria must be at least 2")?,
        };
        match limits.fault() {
            Some(fault) => wrong(fault),
            None => Ok(limits),
        }
    }
}

/// A base URL as it is used: `http` or `https`, a host, perhaps a port and a path, and
/// nothing else. Plain `http` is allowed only to this machine, since a key travels with every
/// request.
///
/// It is held to what Go's `url.Parse` holds one to, so that a URL the Go gateway refuses at
/// startup is refused here when the file is read, and not found out at the first request:
/// the scheme in any case (and kept in lower case), a port that is a number, a bracket that
/// closes round an IPv6 address, and no percent sign that is not an escape.
fn base_url(given: &str) -> Result<String, ConfigError> {
    const SHAPE: &str = "backend base_url must use HTTP(S) without credentials, query, or fragment";
    let Some((scheme, rest)) = given.split_once("://") else {
        return wrong(SHAPE);
    };
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "https" && scheme != "http" {
        return wrong(SHAPE);
    }
    let authority = rest.split('/').next().unwrap_or_default();
    let path = &rest[authority.len()..];
    if authority.is_empty()
        || authority.contains(['@', '%'])
        || rest.contains(['?', '#'])
        || rest.chars().any(|character| character.is_whitespace() || character.is_control())
    {
        return wrong(SHAPE);
    }
    // Every percent sign in the path begins an escape: two hexadecimal digits.
    let escapes_whole = path.split('%').skip(1).all(|after| {
        let digits = after.as_bytes();
        digits.len() >= 2 && digits[..2].iter().all(u8::is_ascii_hexdigit)
    });
    if !escapes_whole {
        return wrong(SHAPE);
    }
    // The host and its port: `[::1]:8080`, `127.0.0.1:8080`, `localhost`.
    let bracketed = authority.starts_with('[');
    let (host, port) = match authority.strip_prefix('[') {
        Some(inside) => {
            let Some((host, after)) = inside.split_once(']') else {
                return wrong(SHAPE);
            };
            let port = match after.strip_prefix(':') {
                Some(port) => port,
                None if after.is_empty() => "",
                None => return wrong(SHAPE),
            };
            if host.parse::<Ipv6Addr>().is_err() {
                return wrong(SHAPE);
            }
            (host, port)
        }
        None => authority.rsplit_once(':').unwrap_or((authority, "")),
    };
    if host.is_empty() || (!bracketed && host.contains(':')) || !port.bytes().all(|digit| digit.is_ascii_digit()) {
        return wrong(SHAPE);
    }
    // This machine, however its address is written: `::ffff:127.0.0.1` is 127.0.0.1.
    let loopback = host == "localhost" || host.parse::<IpAddr>().is_ok_and(|address| address.to_canonical().is_loopback());
    if scheme == "http" && !loopback {
        return wrong("backend base_url requires HTTPS except for loopback hosts");
    }
    // Written again from its parts: the scheme in lower case, no colon that no port follows,
    // and no `/` at the end.
    let host = if bracketed { format!("[{host}]") } else { host.to_string() };
    let port = if port.is_empty() { String::new() } else { format!(":{port}") };
    Ok(format!("{scheme}://{host}{port}{}", path.trim_end_matches('/')))
}

/// The questions a selection asks, from the text of its file.
fn routing_questions(text: &str) -> Result<IndexMap<String, Question>, ConfigError> {
    let Ok(Json::Object(members)) = serde_json::from_str::<Json>(text) else {
        return wrong("selection questions_file must contain a SystemOne questions object");
    };
    if members.is_empty() {
        return wrong("selection questions_file must contain a SystemOne questions object");
    }
    let mut questions = IndexMap::new();
    for (name, definition) in members {
        match serde_json::from_value::<Question>(definition) {
            Ok(question) if valid_question(&question) => questions.insert(name, question),
            _ => return wrong("selection questions_file must contain valid SystemOne questions"),
        };
    }
    Ok(questions)
}

impl Registry {
    /// Read a registry from its JSON. `read` is given the `questions_file` a selection names
    /// and returns that file's text; a registry with no selection never calls it.
    ///
    /// Everything the Go gateway checks at startup is checked here, and reported with the
    /// same words: a member nobody knows at any level, names that are not plain identifiers,
    /// a selector or a fallback that is not a backend, a base URL that is not `http(s)` or is
    /// plain `http` to another machine, limits that are not positive, rules that name no
    /// question. Whether the keys are set is checked when the gateway is built.
    ///
    /// One thing is refused here that the Go gateway reads: a member written twice in one
    /// object. Go keeps the last of the two. A file that names two fallbacks says nothing
    /// certain about where a state stays, so here it is no file.
    pub fn from_json(source: &str, read: impl Fn(&str) -> std::io::Result<String>) -> Result<Self, ConfigError> {
        let mut reader = serde_json::Deserializer::from_str(source);
        let Ok(raw) = RawRegistry::deserialize(&mut reader) else {
            return wrong("backend configuration must contain name, selector, and backends");
        };
        if reader.end().is_err() {
            return wrong("backend configuration must contain one JSON document");
        }
        let name = raw.name.unwrap_or_default();
        if !valid_id(&name) {
            return wrong("configuration name must contain only ASCII letters, digits, hyphens, or underscores");
        }
        let listed = raw.backends.unwrap_or_default();
        if listed.is_empty() {
            return wrong("backend configuration requires at least one backend");
        }
        let mut backends: Vec<Entry> = Vec::new();
        for backend in listed {
            let entry = Entry {
                id: backend.id.unwrap_or_default(),
                base_url: backend.base_url.unwrap_or_default(),
                model: backend.model.unwrap_or_default(),
                api_key_env: backend.api_key_env.unwrap_or_default(),
                description: backend.description.unwrap_or_default(),
                limits: None,
                capabilities: None,
            };
            if entry.id == name {
                return wrong("backend ID must differ from the configuration name");
            }
            let capabilities = backend.capabilities.map(RawCapabilities::read).transpose()?;
            if entry.id.is_empty() || entry.model.is_empty() || entry.description.is_empty() || entry.api_key_env.is_empty() {
                return wrong("every backend requires id, model, description, and api_key_env");
            }
            if !valid_id(&entry.id) {
                return wrong("backend IDs must contain only ASCII letters, digits, hyphens, or underscores");
            }
            if backends.iter().any(|other| other.id == entry.id) {
                return wrong("backend IDs must be unique");
            }
            let base_url = base_url(&entry.base_url)?;
            let limits = backend.limits.map(RawLimits::read).transpose()?;
            backends.push(Entry {
                base_url,
                limits,
                capabilities,
                ..entry
            });
        }
        let known = |id: &str| backends.iter().any(|backend| backend.id == id);
        let selector = raw.selector.unwrap_or_default();
        if !known(&selector) {
            return wrong("selector must name a configured backend ID");
        }
        let fallback = raw.fallback.filter(|id| !id.is_empty());
        if fallback.as_ref().is_some_and(|id| !known(id)) {
            return wrong("fallback must name a configured backend ID");
        }
        if let Some(confidence) = raw.escalation_confidence {
            if fallback.is_none() {
                return wrong("escalation_confidence requires a fallback backend ID");
            }
            if !(0.0..=1.0).contains(&confidence) {
                return wrong("escalation_confidence must fall between 0 and 1");
            }
        }
        let selection = match raw.selection {
            None => None,
            Some(selection) => {
                let escalate_to = selection.escalate_to.unwrap_or_default();
                let rules = selection.rules.unwrap_or_default();
                if fallback.is_none() {
                    return wrong("selection requires a fallback backend ID");
                }
                if !known(&escalate_to) {
                    return wrong("selection escalate_to must name a configured backend ID");
                }
                if Some(&escalate_to) == fallback.as_ref() {
                    return wrong("selection escalate_to must differ from the fallback backend");
                }
                if rules.is_empty() {
                    return wrong("selection requires at least one rule");
                }
                let Ok(text) = read(&selection.questions_file.unwrap_or_default()) else {
                    return wrong("could not read selection questions_file");
                };
                if text.len() > MAX_CONFIG_BYTES {
                    return wrong("could not read selection questions_file");
                }
                let questions = routing_questions(&text)?;
                let mut read_rules = Vec::new();
                for rule in rules {
                    let question = rule.question.unwrap_or_default();
                    let Some(asked) = questions.get(&question) else {
                        return wrong("every selection rule must name a question from questions_file");
                    };
                    let choice = rule.choice.filter(|option| !option.is_empty());
                    if asked.kind == Kind::Choice && !choice.as_ref().is_some_and(|option| asked.labels().contains(option)) {
                        return wrong("every Choice selection rule must name a configured criterion");
                    }
                    let above = rule.above.unwrap_or(0.0);
                    if !(0.0..=1.0).contains(&above) {
                        return wrong("every selection rule threshold must fall between 0 and 1");
                    }
                    read_rules.push(Rule { question, choice, above });
                }
                Some(Selection {
                    questions,
                    escalate_to,
                    rules: read_rules,
                })
            }
        };
        Ok(Self {
            name,
            selector,
            fallback,
            escalation_confidence: raw.escalation_confidence,
            selection,
            backends,
        })
    }

    /// Read a registry file.
    ///
    /// A `questions_file` is looked for from the working directory, as the Go gateway does,
    /// and failing that from the directory the registry is in and each directory above it, so
    /// that `examples/privacy.backends.json` finds `examples/routing.questions.json` from
    /// wherever it is loaded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let Ok(source) = read_bounded(path) else {
            return wrong("could not open ONE_SYSTEM_CONFIG");
        };
        Self::from_json(&source, |questions_file| {
            let named = Path::new(questions_file);
            let mut places: Vec<PathBuf> = vec![named.to_path_buf()];
            if named.is_relative() {
                // The registry's own directory, then each one above it.
                let home = path.parent().map(Path::to_path_buf).unwrap_or_default();
                let home = std::path::absolute(&home).unwrap_or(home);
                places.extend(home.ancestors().map(|directory| directory.join(named)));
            }
            match places.iter().find(|place| place.is_file()) {
                Some(place) => read_bounded(place),
                None => Err(std::io::ErrorKind::NotFound.into()),
            }
        })
    }

    /// A gateway builder with a [`Client`] for every backend, their keys read now from the
    /// environment variables the file names. A variable that is unset or holds no usable key
    /// is an error, whichever backend it is for.
    pub fn builder(&self) -> Result<GatewayBuilder, ConfigError> {
        self.builder_with(|variable| std::env::var(variable).ok())
    }

    /// As [`Registry::builder`], with the keys from a function of the variable's name.
    pub fn builder_with(&self, key: impl Fn(&str) -> Option<String>) -> Result<GatewayBuilder, ConfigError> {
        let mut builder = Gateway::builder(self.name.clone()).selector(self.selector.clone());
        for entry in &self.backends {
            let Some(key) = key(&entry.api_key_env).filter(|key| valid_key(key)) else {
                return wrong("a backend api_key_env must reference a nonempty bearer key without whitespace");
            };
            // One request, once: a gateway's clients do not retry.
            let client = Client::new(entry.base_url.clone()).model(entry.model.clone()).key(key);
            let mut backend = Backend::new(entry.id.clone(), client).description(entry.description.clone());
            backend.limits = entry.limits.clone();
            backend.capabilities = entry.capabilities.clone();
            builder = builder.backend(backend);
        }
        if let Some(fallback) = &self.fallback {
            builder = builder.fallback(fallback.clone());
        }
        if let Some(confidence) = self.escalation_confidence {
            builder = builder.escalation_confidence(confidence);
        }
        if let Some(selection) = &self.selection {
            builder = builder.selection(selection.clone());
        }
        Ok(builder)
    }

    /// The gateway, with keys from the environment.
    pub fn gateway(&self) -> Result<Gateway, ConfigError> {
        self.builder()?.build()
    }

    /// The gateway, with keys from a function of the variable's name.
    pub fn gateway_with(&self, key: impl Fn(&str) -> Option<String>) -> Result<Gateway, ConfigError> {
        self.builder_with(key)?.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_urls() {
        assert_eq!(base_url("https://api.typesafe.ai").unwrap(), "https://api.typesafe.ai");
        assert_eq!(base_url("https://openrouter.ai/api//").unwrap(), "https://openrouter.ai/api");
        assert_eq!(base_url("http://127.0.0.1:8091/").unwrap(), "http://127.0.0.1:8091");
        assert_eq!(base_url("http://localhost:8080").unwrap(), "http://localhost:8080");
        assert_eq!(base_url("http://[::1]:8080").unwrap(), "http://[::1]:8080");
        assert_eq!(
            base_url("https://example.com/tenant%2Fteam/%E2%98%83").unwrap(),
            "https://example.com/tenant%2Fteam/%E2%98%83"
        );
        for bad in [
            "",
            "ftp://example.com",
            "https://",
            "https://user:pass@example.com",
            "https://example.com?x=1",
            "https://example.com/#top",
            "example.com",
            "https://exa mple.com",
        ] {
            assert!(base_url(bad).unwrap_err().message().contains("HTTP(S)"), "{bad}");
        }
        for remote in ["http://example.com", "http://10.0.0.5:8080", "http://localhost.evil.example"] {
            assert!(base_url(remote).unwrap_err().message().contains("HTTPS except"), "{remote}");
        }
    }

    #[test]
    fn base_urls_at_the_edges() {
        // Read as Go reads them, and written again plainly.
        assert_eq!(base_url("HTTPS://API.typesafe.ai/V1/").unwrap(), "https://API.typesafe.ai/V1");
        assert_eq!(base_url("hTTp://localhost:8080").unwrap(), "http://localhost:8080");
        assert_eq!(base_url("http://localhost:").unwrap(), "http://localhost");
        assert_eq!(base_url("http://[::1]:").unwrap(), "http://[::1]");
        assert_eq!(
            base_url("http://[::ffff:127.0.0.1]:8091/x").unwrap(),
            "http://[::ffff:127.0.0.1]:8091/x"
        );
        assert_eq!(base_url("http://[0:0:0:0:0:0:0:1]:1").unwrap(), "http://[0:0:0:0:0:0:0:1]:1");
        assert_eq!(
            base_url("https://example.com/a%2Fb/%e2%98%83/").unwrap(),
            "https://example.com/a%2Fb/%e2%98%83"
        );
        for bad in [
            "http://localhost:notaport",
            "http://localhost:80:80",
            "https://example.com:8o",
            "https://[::1",
            "https://[::1]8080",
            "https://[::1]]",
            "https://[not-an-address]",
            "https://::1",
            "https://:8080",
            "http://127.0.0.1:8091/%zz",
            "https://example.com/%",
            "https://example.com/%2",
            "https://example.com/ok%2Fthen%g0",
            "https://example%2ecom",
            "https://example.com/a\u{7}b",
            "https+http://example.com",
            "://example.com",
        ] {
            assert!(base_url(bad).unwrap_err().message().contains("HTTP(S)"), "{bad}");
        }
        for remote in ["http://[::ffff:10.0.0.5]", "http://[::2]:80", "HTTP://example.com"] {
            assert!(base_url(remote).unwrap_err().message().contains("HTTPS except"), "{remote}");
        }
    }
}
