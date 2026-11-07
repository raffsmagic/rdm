//! The two models of an example, chosen by the environment: a decision model, and for the
//! examples that write, a language model. With nothing set and no key, both are stand-ins that
//! answer by the rules the example gives them, with no network.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use one_system::{Client, Gateway, Retry};
use rdm::Json;
use rdm::lm::{Lm, ScriptedLm};
use rdm::systemone::{self, Answer, DecisionModel, Kind, Question, Request, Response, Scripted};

use super::Failure;

/// The name the stand-in decision model answers under.
pub const STAND_IN: &str = "jev-stand-in";
/// The name the stand-in language model answers under.
pub const STAND_IN_LM: &str = "lm-stand-in";
/// The language model the site's hosted mode asks, through OpenRouter.
pub const HOSTED_LM: &str = "openrouter:typesafe/jev-router";
/// The System One server `local` means: llama.cpp serving Laya, where it listens by default.
pub const LOCAL: &str = "http://127.0.0.1:8080";

/// How likely a stand-in's yes is, its no, and the option it chooses: the numbers of the
/// site's own mock (`test/notebook.mjs`). The options not chosen share what is left.
pub const YES: f64 = 0.93;
pub const NO: f64 = 0.04;
pub const CHOSEN: f64 = 0.95;

/// What a stand-in decision model says to one question.
#[derive(Clone, Debug, PartialEq)]
pub enum Say {
    /// A yes or a no, as likely as the site's mock gives it: 0.93 or 0.04.
    Yes(bool),
    /// How likely a yes is.
    P(f64),
    /// The option chosen, at 0.95, the others sharing the rest.
    Pick(String),
    /// How likely each option is. An option not named is at 0.
    Odds(Vec<(String, f64)>),
    /// The level scored, as certain.
    Level(usize),
}

impl From<bool> for Say {
    fn from(yes: bool) -> Self {
        Say::Yes(yes)
    }
}

impl From<f64> for Say {
    fn from(probability: f64) -> Self {
        Say::P(probability)
    }
}

impl From<&str> for Say {
    fn from(option: &str) -> Self {
        Say::Pick(option.to_owned())
    }
}

impl From<String> for Say {
    fn from(option: String) -> Self {
        Say::Pick(option)
    }
}

impl From<&String> for Say {
    fn from(option: &String) -> Self {
        Say::Pick(option.clone())
    }
}

impl<L: Into<String>> From<Vec<(L, f64)>> for Say {
    fn from(odds: Vec<(L, f64)>) -> Self {
        Say::Odds(odds.into_iter().map(|(option, probability)| (option.into(), probability)).collect())
    }
}

/// What was said, as the probabilities a System One model returns for that kind of question.
fn evidence(name: &str, question: &Question, said: Say) -> Result<Answer, systemone::Error> {
    let labels = question.labels();
    let known = |option: &str| match labels.iter().any(|label| label == option) {
        true => Ok(()),
        false => Err(systemone::Error::Evidence(format!(
            "The stand-in answered '{name}' with '{option}', which is not one of its options: {labels:?}."
        ))),
    };
    match (question.kind, said) {
        (Kind::Noul, Say::Yes(yes)) => Ok(Answer::noul(if yes { YES } else { NO })),
        (Kind::Noul, Say::P(probability)) => Ok(Answer::noul(probability)),
        (Kind::Choice, Say::Pick(option)) => {
            known(&option)?;
            let rest = (1.0 - CHOSEN) / labels.len().saturating_sub(1).max(1) as f64;
            let odds = labels
                .iter()
                .map(|label| (label.clone(), if *label == option { CHOSEN } else { rest }));
            Ok(Answer::choice(odds))
        }
        (Kind::Choice, Say::Odds(odds)) => {
            for (option, _) in &odds {
                known(option)?;
            }
            let of = |label: &String| {
                odds.iter()
                    .find(|(option, _)| option == label)
                    .map_or(0.0, |(_, probability)| *probability)
            };
            Ok(Answer::choice(labels.iter().map(|label| (label.clone(), of(label)))))
        }
        (Kind::Score, Say::Level(level)) => Ok(Answer::score((0..labels.len()).map(|at| if at == level { 1.0 } else { 0.0 }))),
        (kind, said) => Err(systemone::Error::Evidence(format!(
            "The stand-in answered '{name}', a {} question, with {said:?}.",
            kind.as_str()
        ))),
    }
}

/// A stand-in decision model: `rules(inputs, name, question)` gives each question's answer,
/// where `inputs` is what the request showed, `name` the output asked about, and `question`
/// what was asked (its options are `question.labels()`).
pub fn standin<S: Into<Say>>(rules: impl Fn(&Json, &str, &Question) -> S + Send + Sync + 'static) -> Scripted {
    let answers = move |request: &Request| {
        let inputs = &request.state["inputs"];
        let each = request
            .questions
            .iter()
            .map(|(name, question)| Ok((name.clone(), evidence(name, question, rules(inputs, name, question).into())?)));
        each.collect()
    };
    Scripted::fallible(answers).named(STAND_IN)
}

/// Answers that are given in turn, by task: the plans of the stand-in the Python tests its
/// reference page with (`tests/predict/test_rdm_docs.py`). Each task has the names to point at,
/// one for each time it is asked; when they have run out, or for a task it has no plan for,
/// it says `finish`.
pub struct Plans(Mutex<HashMap<String, VecDeque<String>>>);

impl Plans {
    pub fn new<const N: usize>(plans: [(&str, &[&str]); N]) -> Self {
        let plans = plans
            .into_iter()
            .map(|(task, names)| (task.to_owned(), names.iter().map(|name| name.to_string()).collect()));
        Plans(Mutex::new(plans.collect()))
    }

    /// The next name of the plan for a task.
    pub fn next(&self, task: &str) -> String {
        let mut plans = self.0.lock().expect("the plans");
        let next = plans.get_mut(task).and_then(VecDeque::pop_front);
        next.unwrap_or_else(|| rdm::FINISH.to_owned())
    }
}

/// The decision model an example asks, and the name it goes by.
#[derive(Clone)]
pub struct Decider {
    model: Arc<dyn DecisionModel>,
    name: String,
    stand_in: bool,
}

impl Decider {
    /// The name of the model, as the cells print it: `decision_model.model` in Python.
    pub fn model(&self) -> &str {
        &self.name
    }

    /// Whether it is the stand-in, which answers by the example's own rules.
    pub fn is_stand_in(&self) -> bool {
        self.stand_in
    }

    fn real(model: impl DecisionModel + 'static, name: impl Into<String>) -> Self {
        Decider {
            model: Arc::new(model),
            name: name.into(),
            stand_in: false,
        }
    }
}

impl DecisionModel for Decider {
    fn decide(&self, request: &Request) -> Result<Response, systemone::Error> {
        self.model.decide(request)
    }

    fn name(&self) -> String {
        self.model.name()
    }
}

/// An environment variable, if it is set to something.
fn set(variable: &str) -> Option<String> {
    std::env::var(variable)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Whether a key is in the environment. It is never read here: only whether it is there.
fn has(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|value| !value.is_empty())
}

/// Whether the program was asked only what it takes: then nothing is said about the models.
fn asked_for_help() -> bool {
    std::env::args().skip(1).any(|argument| argument == "--help" || argument == "-h")
}

fn note(line: &str) {
    if !asked_for_help() {
        eprintln!("{line}");
    }
}

/// A System One server, asked again after a failure that asking again could cure.
fn client(client: Client) -> Client {
    client.retry(Retry::default()).timeout(Duration::from_secs(120))
}

/// A System One server at an address. One that asks for a key is told which variable holds
/// it, by `RDM_DECISION_MODEL_KEY`: the key is read by the client when it asks, and by nothing
/// here. One that serves several models, as a one-system gateway does, is asked for the one
/// `RDM_DECISION_MODEL_NAME` names: llama.cpp wants neither.
fn server(address: &str) -> Decider {
    let server = Client::local(address);
    let server = match set("RDM_DECISION_MODEL_KEY") {
        Some(variable) => server.key_env(variable),
        None => server,
    };
    match set("RDM_DECISION_MODEL_NAME") {
        Some(model) => Decider::real(client(server.model(model.as_str())), model),
        None => Decider::real(Unnamed(client(server)), address),
    }
}

/// A server that is asked for no model by name. One that serves several refuses such a
/// request as one it cannot read, which does not say what is missing: this says it.
struct Unnamed(Client);

impl DecisionModel for Unnamed {
    fn decide(&self, request: &Request) -> Result<Response, systemone::Error> {
        self.0.decide(request).map_err(|error| match error {
            systemone::Error::Api {
                status: 422,
                code,
                message,
            } if code == "schema_validation" => systemone::Error::Api {
                status: 422,
                code,
                message: format!(
                    "{}. No model was named: a server of several, as a one-system gateway is, wants one in RDM_DECISION_MODEL_NAME.",
                    message.trim_end_matches('.')
                ),
            },
            other => other,
        })
    }

    fn name(&self) -> String {
        self.0.name()
    }
}

/// The decision model the environment asks for, by `RDM_DECISION_MODEL`:
///
/// - `typesafe`: hosted Jev, with the key in `TYPESAFE_API_KEY` (`typesafe:jev-latest` names the model)
/// - `openrouter`: Jev through OpenRouter, with the key in `OPENROUTER_API_KEY`
/// - `local`, or a URL: a System One server on this machine, as llama.cpp serving Laya. For a
///   server that asks for a key or serves several models, as a one-system gateway does,
///   `RDM_DECISION_MODEL_KEY` names the variable the key is in and `RDM_DECISION_MODEL_NAME` the model
/// - `gateway:<path>`: the gateway a one-system registry file describes
/// - `lm`: the language model of `RDM_LANGUAGE_MODEL`, answering decision questions. It has to
///   be a real one: the stand-in for a language model cannot decide
/// - `standin`: `rules`, with no network
///
/// With nothing set it is hosted Jev if there is a key for it, else OpenRouter if there is
/// a key for that, else the stand-in, and one line on stderr says which.
pub fn decision_model<S: Into<Say>>(rules: impl Fn(&Json, &str, &Question) -> S + Send + Sync + 'static) -> Result<Decider, Failure> {
    let stand_in = || Decider {
        model: Arc::new(standin(rules)),
        name: STAND_IN.to_owned(),
        stand_in: true,
    };
    let Some(asked) = set("RDM_DECISION_MODEL") else {
        const OTHERS: &str = "RDM_DECISION_MODEL picks another: standin, typesafe, openrouter, local, a URL, gateway:<registry.json>, lm";
        return Ok(if has("TYPESAFE_API_KEY") {
            note(&format!("decision model: hosted Jev, since TYPESAFE_API_KEY is set. {OTHERS}."));
            Decider::real(client(Client::typesafe()), "jev-latest")
        } else if has("OPENROUTER_API_KEY") {
            note(&format!(
                "decision model: Jev through OpenRouter, since OPENROUTER_API_KEY is set. {OTHERS}."
            ));
            Decider::real(client(Client::openrouter()), "~typesafe/jev-latest")
        } else {
            note(&format!(
                "decision model: the stand-in, which answers by this example's rules with no network. {OTHERS}."
            ));
            stand_in()
        });
    };
    let needs = |key: &str| match has(key) {
        true => Ok(()),
        false => Err(Failure::new(format!("RDM_DECISION_MODEL={asked} needs a key in {key}."))),
    };
    let (kind, rest) = asked
        .split_once(':')
        .map_or((asked.as_str(), None), |(kind, rest)| (kind, Some(rest)));
    Ok(match (kind, rest) {
        ("standin" | "stand-in", None) => stand_in(),
        ("typesafe", model) => {
            needs("TYPESAFE_API_KEY")?;
            let model = model.unwrap_or("jev-latest");
            Decider::real(client(Client::typesafe().model(model)), model)
        }
        ("openrouter", model) => {
            needs("OPENROUTER_API_KEY")?;
            let model = model.unwrap_or("~typesafe/jev-latest");
            Decider::real(client(Client::openrouter().model(model)), model)
        }
        ("local", None) => server(LOCAL),
        ("http" | "https", Some(_)) => server(&asked),
        ("gateway", Some(path)) => {
            let gateway =
                Gateway::load(path).map_err(|error| Failure::new(format!("The registry {path} does not load: {}", error.message())))?;
            let name = gateway.name();
            Decider::real(gateway, name)
        }
        ("lm", None) => {
            let lm = language_model_named()?.ok_or_else(|| match set("RDM_LANGUAGE_MODEL") {
                // The stand-in for a language model writes by an example's rules, and they are not rules to decide by.
                Some(stand_in) => Failure::new(format!(
                    "RDM_DECISION_MODEL=lm asks a language model to decide, and RDM_LANGUAGE_MODEL={stand_in} is a stand-in, \
                     which cannot: name a real language model, or set RDM_DECISION_MODEL=standin."
                )),
                None => Failure::new(
                    "RDM_DECISION_MODEL=lm asks a language model to decide, and none is named: set RDM_LANGUAGE_MODEL, or OPENROUTER_API_KEY.",
                ),
            })?;
            let name = lm.model().to_owned();
            Decider::real(lm.temperature(0.0).decision_model(), name)
        }
        _ => {
            return Err(Failure::new(format!(
                "RDM_DECISION_MODEL={asked} is not one of: standin, typesafe, openrouter, local, a URL, gateway:<registry.json>, lm."
            )));
        }
    })
}

/// The language model the environment names, if it names a real one.
fn language_model_named() -> Result<Option<Lm>, Failure> {
    // What the site's hosted mode sends with every request to its language model.
    let as_the_site = |lm: Lm| lm.temperature(0.0).max_tokens(512);
    let lm = match set("RDM_LANGUAGE_MODEL") {
        None if has("OPENROUTER_API_KEY") => as_the_site(Lm::new(HOSTED_LM)?),
        None => return Ok(None),
        Some(asked) if asked == "standin" || asked == "stand-in" => return Ok(None),
        Some(url) if url.starts_with("http://") || url.starts_with("https://") => {
            let name = set("RDM_LANGUAGE_MODEL_NAME").unwrap_or_else(|| "local".to_owned());
            Lm::local(url, name)?.temperature(0.0)
        }
        Some(model) => Lm::new(model)?.temperature(0.0),
    };
    Ok(Some(match set("RDM_LANGUAGE_MODEL_MAX_TOKENS") {
        Some(most) => lm.max_tokens(
            most.parse()
                .map_err(|_| Failure::new(format!("RDM_LANGUAGE_MODEL_MAX_TOKENS={most} is not a number.")))?,
        ),
        None => lm,
    }))
}

/// The language model the environment asks for, by `RDM_LANGUAGE_MODEL`:
///
/// - an lm15 model string, as `openrouter:typesafe/jev-router` or `anthropic:claude-sonnet-5-5`,
///   with the provider's key read from the environment by lm15
/// - a URL: a server on this machine that speaks Chat Completions, `/v1` included, serving the
///   model `RDM_LANGUAGE_MODEL_NAME` names
/// - `standin`: `writes(system, prompt)`, with no network
///
/// With nothing set it is `openrouter:typesafe/jev-router` if `OPENROUTER_API_KEY` is set, which
/// is what the site's hosted mode asks, else the stand-in, and one line on stderr says which.
/// `RDM_LANGUAGE_MODEL_MAX_TOKENS` is the most tokens one reply may be. Unset, the model taken
/// by default is asked for at most 512, as the site asks it, and one that was named for no fewer
/// than it gives.
/// A stand-in is given the system prompt and the last message, and replies with the JSON a
/// model would: `rdm::lm::input_of(prompt, "task")` reads an input out of the message.
pub fn language_model(writes: impl Fn(&str, &str) -> String + Send + Sync + 'static) -> Result<Lm, Failure> {
    const OTHERS: &str = "RDM_LANGUAGE_MODEL picks another: standin, an lm15 model string, or the URL of a local server";
    let named = language_model_named()?;
    if set("RDM_LANGUAGE_MODEL").is_none() {
        match &named {
            Some(lm) => note(&format!(
                "language model: {}, since OPENROUTER_API_KEY is set. {OTHERS}.",
                lm.model()
            )),
            None => note(&format!(
                "language model: the stand-in, which writes by this example's rules with no network. {OTHERS}."
            )),
        }
    }
    Ok(named.unwrap_or_else(|| Lm::of(ScriptedLm::new(writes), STAND_IN_LM)))
}
