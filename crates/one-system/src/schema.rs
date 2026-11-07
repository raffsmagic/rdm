//! What a request and its answers must look like.
//!
//! The rules are those of TypeSafe's published schema, written out by hand: there are few of
//! them, and none says anything about a model.

use indexmap::IndexMap;
use systemone::{Answer, Error, Json, Kind, Question, Request, Response};

use crate::error::api;

/// The most a request or a response body may be: 8 MiB, and exactly that much is accepted.
pub const MAX_BODY_BYTES: usize = 8 << 20;

/// Text, an object or a list: what a state, and one level of a score, may be.
fn content(value: &Json) -> bool {
    value.is_string() || value.is_object() || value.is_array()
}

/// As [`content`], or nothing: what instructions and an option's description may be.
fn optional_content(value: &Json) -> bool {
    value.is_null() || content(value)
}

/// Whether one question is one the protocol allows.
pub(crate) fn valid_question(question: &Question) -> bool {
    if !optional_content(&question.instructions) {
        return false;
    }
    match (question.kind, &question.criteria) {
        (Kind::Noul, None | Some(Json::Null)) => true,
        (Kind::Noul, Some(Json::Object(meanings))) => ["true", "false"]
            .iter()
            .all(|side| meanings.get(*side).is_none_or(optional_content)),
        (Kind::Choice, Some(Json::Object(options))) => options.values().all(optional_content),
        (Kind::Score, Some(Json::Array(levels))) => !levels.is_empty() && levels.iter().all(content),
        _ => false,
    }
}

fn schema_error() -> Error {
    api(
        422,
        "schema_validation",
        "Request does not match the official SystemOneRequest schema",
    )
}

/// Check a request against the protocol: a state that is text, an object or a list, and at
/// least one question, each of a shape its kind allows.
///
/// ```
/// use systemone::{Question, Request};
///
/// let request = Request::new("I was charged twice.").question("refund", Question::noul("Is a refund asked for?"));
/// assert!(one_system::validate(&request).is_ok());
///
/// let nothing_asked = Request::new("I was charged twice.");
/// assert_eq!(one_system::validate(&nothing_asked).unwrap_err().status(), Some(422));
/// ```
pub fn validate(request: &Request) -> Result<(), Error> {
    if content(&request.state) && !request.questions.is_empty() && request.questions.values().all(valid_question) {
        Ok(())
    } else {
        Err(schema_error())
    }
}

/// Read a request from the bytes of a body, as the server does: one JSON object with a
/// `model` that is a string, a `state` and `questions`. Members the protocol does not name
/// are dropped.
pub fn parse_request(body: &[u8]) -> Result<Request, Error> {
    let value: Json = serde_json::from_slice(body).map_err(|_| schema_error())?;
    if !value.get("model").is_some_and(Json::is_string) {
        return Err(schema_error());
    }
    let request: Request = serde_json::from_value(value).map_err(|_| schema_error())?;
    validate(&request)?;
    Ok(request)
}

/// Check that a response answers the questions that were asked, and nothing else.
///
/// Every question has an answer and every answer a question; an answer is of its question's
/// kind; a noul carries its probability; a choice or a score carries a probability for each
/// option or level declared, and for no other; and the option a server says it picked is one
/// of those declared. What the probabilities are is not judged here: [`systemone::decide`]
/// does that. Neither is it asked that a choice say which option it picked, or a score its
/// level: a decision is made from the probabilities, and those are what must be there.
///
/// ```
/// use systemone::{Answer, Question, Request, Response};
///
/// let request = Request::new("ok").question("tone", Question::choice("Which tone?", [("warm", None), ("cold", None)]));
/// let mut response = Response::default();
/// response.answers.insert("tone".into(), Answer::choice([("warm", 0.8), ("cold", 0.2)]));
/// assert!(one_system::check_answers(&request, &response).is_ok());
///
/// response.answers.insert("tone".into(), Answer::choice([("warm", 0.8), ("hot", 0.2)]));
/// assert!(one_system::check_answers(&request, &response).is_err());
/// ```
pub fn check_answers(request: &Request, response: &Response) -> Result<(), Error> {
    match mismatch(&request.questions, &response.answers) {
        None => Ok(()),
        Some(fault) => Err(fault.evidence()),
    }
}

/// How an error begins that says answers are answers, to other questions than those asked.
/// A server tells such an error from one about a response that is no response by this.
pub(crate) const MISMATCHED: &str = "The answers are not for the questions asked";

/// What is wrong with a set of answers. The text names questions, never content.
///
/// The two kinds are the two errors the Go gateway has for a response it will not pass on,
/// and a caller over HTTP is told which.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Fault {
    /// An answer is not a System One answer: it says no kind, or lacks its probabilities.
    /// `invalid_upstream_response`.
    Malformed(String),
    /// The answers are answers, to other questions: other names, other kinds, other options.
    /// `mismatched_answers`.
    Mismatched(String),
}

impl Fault {
    /// As a client reports it: what came back is not evidence for the questions.
    pub(crate) fn evidence(&self) -> Error {
        match self {
            Fault::Malformed(what) => Error::Evidence(format!("The server's response is not a System One response: {what}.")),
            Fault::Mismatched(what) => Error::Evidence(format!("{MISMATCHED}: {what}.")),
        }
    }

    /// As a gateway reports it of a backend, in the fixed words a caller over HTTP is given.
    pub(crate) fn refusal(&self) -> Error {
        match self {
            Fault::Malformed(_) => invalid_upstream(),
            Fault::Mismatched(_) => mismatched(),
        }
    }
}

/// A backend's response that is not a System One response.
pub(crate) fn invalid_upstream() -> Error {
    api(502, "invalid_upstream_response", "Upstream returned an invalid SystemOneResponse")
}

/// A backend's answers that are for other questions than the ones it was asked.
pub(crate) fn mismatched() -> Error {
    api(
        502,
        "mismatched_answers",
        "Upstream answers do not match the requested question IDs, types, or choices",
    )
}

/// What is wrong with the answers, if anything is.
pub(crate) fn mismatch(questions: &IndexMap<String, Question>, answers: &IndexMap<String, Answer>) -> Option<Fault> {
    if let Some(name) = questions.keys().find(|name| !answers.contains_key(*name)) {
        return Some(Fault::Mismatched(format!("there is no answer for '{name}'")));
    }
    if answers.len() != questions.len() {
        return Some(Fault::Mismatched("there is an answer for a question that was not asked".into()));
    }
    // An answer that is none is told of before one that is for another question, as a
    // response is read before it is compared.
    if let Some((name, _)) = answers.iter().find(|(_, answer)| answer.kind.is_none()) {
        return Some(Fault::Malformed(format!("the answer for '{name}' says no type")));
    }
    for (name, question) in questions {
        let answer = &answers[name];
        let kind = question.kind.as_str();
        if answer.kind != Some(question.kind) {
            return Some(Fault::Mismatched(format!("the answer for '{name}' is not a {kind}")));
        }
        if question.kind == Kind::Noul {
            if answer.noul.is_none() {
                return Some(Fault::Malformed(format!("the answer for '{name}' has no probability")));
            }
            continue;
        }
        let labels = question.labels();
        let Some(probabilities) = &answer.probabilities else {
            return Some(Fault::Malformed(format!("the answer for '{name}' has no probabilities")));
        };
        if probabilities.len() != labels.len() || labels.iter().any(|label| !probabilities.contains_key(label)) {
            return Some(Fault::Mismatched(format!(
                "the probabilities for '{name}' are not for what the {kind} declared"
            )));
        }
        if question.kind == Kind::Choice && answer.choice.as_ref().is_some_and(|picked| !labels.contains(picked)) {
            return Some(Fault::Mismatched(format!("the option picked for '{name}' is not one it declared")));
        }
    }
    None
}

/// Make a response whole by TypeSafe's published schema, which is what a server sends.
///
/// The schema requires of a choice its `choice` and `confidence`, and of a score its `score`,
/// `confidence` and `legend`. In the process none of them is needed, since a decision is made
/// from the probabilities, and a backend that is a function gives only those. So what a
/// backend left out is worked out here, the way TypeSafe publishes it: the likeliest option,
/// the expected level, the confidence of each, and the rubric by level. What a backend did say
/// is left as it said it.
///
/// It is for answers that already passed [`mismatch`]. Two things cannot be made whole: a
/// choice among no options, which has no option to pick, and a number that is not finite,
/// which JSON cannot write.
#[cfg(any(all(feature = "server", not(target_family = "wasm")), test))]
pub(crate) fn complete(questions: &IndexMap<String, Question>, response: &mut Response) -> Result<(), Error> {
    use systemone::{choice_confidence, likeliest, score_confidence};

    for (name, answer) in &mut response.answers {
        let (Some(question), Some(kind)) = (questions.get(name), answer.kind) else {
            return Err(invalid_upstream());
        };
        if kind != Kind::Noul {
            let Some(probabilities) = &answer.probabilities else {
                return Err(invalid_upstream());
            };
            // In the order declared: which comes first decides between two as likely.
            let ordered: IndexMap<String, f64> = question
                .labels()
                .into_iter()
                .filter_map(|label| probabilities.get(&label).map(|probability| (label, *probability)))
                .collect();
            let levels: Vec<f64> = ordered.values().copied().collect();
            if kind == Kind::Choice {
                if answer.choice.is_none() {
                    answer.choice = Some(likeliest(&ordered).ok_or_else(invalid_upstream)?.to_string());
                }
                answer.confidence.get_or_insert_with(|| choice_confidence(levels.iter().copied()));
            } else {
                answer.score.get_or_insert_with(|| {
                    let total: f64 = levels.iter().sum();
                    let weighted: f64 = levels
                        .iter()
                        .enumerate()
                        .map(|(level, probability)| level as f64 * probability)
                        .sum();
                    if total > 0.0 { weighted / total } else { 0.0 }
                });
                answer.confidence.get_or_insert_with(|| score_confidence(&levels));
                if answer.legend.is_none() {
                    let described = match &question.criteria {
                        Some(Json::Array(described)) => described.as_slice(),
                        _ => &[],
                    };
                    let legend = described
                        .iter()
                        .enumerate()
                        .map(|(level, meaning)| (level.to_string(), meaning.clone()));
                    answer.legend = Some(Json::Object(legend.collect()));
                }
            }
        }
        let numbers = [answer.noul, answer.score, answer.confidence];
        let finite = numbers.into_iter().flatten().all(f64::is_finite)
            && answer
                .probabilities
                .iter()
                .flat_map(IndexMap::values)
                .all(|probability| probability.is_finite());
        if !finite {
            return Err(api(
                502,
                "invalid_response",
                "Upstream response cannot be returned as SystemOneResponse",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use systemone::{choice_confidence, score_confidence};

    #[test]
    fn the_shapes_the_schema_allows() {
        let state = |state: Json| Request::new(state).question("q", Question::noul("?"));
        assert!(validate(&state(json!("text"))).is_ok());
        assert!(validate(&state(json!({"a": 1}))).is_ok());
        assert!(validate(&state(json!([1, 2]))).is_ok());
        for bad in [json!(null), json!(3), json!(true)] {
            assert!(validate(&state(bad)).is_err());
        }

        let asked = |question: Json| serde_json::from_value::<Question>(question).is_ok_and(|question| valid_question(&question));
        assert!(asked(json!({"type": "noul"})));
        assert!(asked(json!({"type": "noul", "instructions": null, "criteria": null})));
        assert!(asked(json!({"type": "noul", "criteria": {"true": "yes", "false": null}})));
        assert!(!asked(json!({"type": "noul", "criteria": {"true": 1}})));
        assert!(!asked(json!({"type": "noul", "criteria": ["a"]})));
        assert!(!asked(json!({"type": "noul", "instructions": 7})));
        assert!(asked(json!({"type": "choice", "criteria": {}})));
        assert!(asked(json!({"type": "choice", "criteria": {"a": null, "b": {"x": 1}}})));
        assert!(!asked(json!({"type": "choice"})));
        assert!(!asked(json!({"type": "choice", "criteria": {"a": 1}})));
        assert!(asked(json!({"type": "score", "criteria": ["low", ["x"], {"y": 1}]})));
        assert!(!asked(json!({"type": "score", "criteria": []})));
        assert!(!asked(json!({"type": "score", "criteria": ["low", null]})));
        assert!(!asked(json!({"type": "Noul"})));
        assert!(!asked(json!({"type": "essay"})));
    }

    #[test]
    fn a_body_is_one_object_with_a_model() {
        let ok = br#"{"model":"m","state":"s","questions":{"q":{"type":"noul"}},"opaque":1}"#;
        assert_eq!(parse_request(ok).unwrap().model, "m");
        for bad in [
            &br#"{"state":"s","questions":{"q":{"type":"noul"}}}"#[..],
            br#"{"model":1,"state":"s","questions":{"q":{"type":"noul"}}}"#,
            br#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}}}"#,
            br#"{"model":"m","state":"s","questions":{}}"#,
            br#"{"model":"m","state":"s","questions":{"q":{"type":"noul"}}} {}"#,
            br#"{"model":"m","state":"s""#,
            br#"[]"#,
            b"",
        ] {
            let error = parse_request(bad).unwrap_err();
            assert_eq!((error.status(), error.to_string().contains("schema_validation")), (Some(422), true));
        }
    }

    #[test]
    fn answers_are_for_the_questions() {
        let questions: IndexMap<String, Question> = IndexMap::from([
            ("yes".to_string(), Question::noul("?")),
            ("pick".to_string(), Question::choice("?", [("a", None), ("b", None)])),
            ("rate".to_string(), Question::score("?", ["low", "high"])),
        ]);
        let good: IndexMap<String, Answer> = IndexMap::from([
            ("rate".to_string(), Answer::score([0.5, 0.5])),
            ("pick".to_string(), Answer::choice([("b", 0.4), ("a", 0.6)])),
            ("yes".to_string(), Answer::noul(0.5)),
        ]);
        assert_eq!(mismatch(&questions, &good), None);

        let with = |name: &str, answer: Option<Answer>| {
            let mut answers = good.clone();
            match answer {
                Some(answer) => answers.insert(name.to_string(), answer),
                None => answers.shift_remove(name),
            };
            mismatch(&questions, &answers)
        };
        assert!(with("yes", None).is_some());
        assert!(with("extra", Some(Answer::noul(0.5))).is_some());
        assert!(with("yes", Some(Answer::choice([("a", 1.0)]))).is_some());
        assert!(
            with(
                "yes",
                Some(Answer {
                    noul: None,
                    ..Answer::noul(0.5)
                })
            )
            .is_some()
        );
        assert!(
            with(
                "yes",
                Some(Answer {
                    kind: None,
                    ..Answer::noul(0.5)
                })
            )
            .is_some()
        );
        assert!(with("pick", Some(Answer::choice([("a", 1.0)]))).is_some());
        assert!(with("pick", Some(Answer::choice([("a", 0.5), ("c", 0.5)]))).is_some());
        assert!(
            with(
                "pick",
                Some(Answer {
                    choice: Some("c".into()),
                    ..Answer::choice([("a", 0.5), ("b", 0.5)])
                })
            )
            .is_some()
        );
        assert!(
            with(
                "pick",
                Some(Answer {
                    choice: None,
                    ..Answer::choice([("a", 0.5), ("b", 0.5)])
                })
            )
            .is_none()
        );
        assert!(with("rate", Some(Answer::score([0.2, 0.3, 0.5]))).is_some());
        assert!(
            with(
                "rate",
                Some(Answer {
                    probabilities: None,
                    ..Answer::score([0.5, 0.5])
                })
            )
            .is_some()
        );
    }

    #[test]
    fn an_answer_that_is_none_is_told_from_one_for_another_question() {
        let asked = Request::new("s")
            .question("yes", Question::noul("?"))
            .question("pick", Question::choice("?", [("a", None), ("b", None)]));
        let with = |name: &str, answer: Answer| {
            let mut response = Response::default();
            response.answers.insert("yes".into(), Answer::noul(0.5));
            response.answers.insert("pick".into(), Answer::choice([("a", 0.5), ("b", 0.5)]));
            response.answers.insert(name.into(), answer);
            let fault = mismatch(&asked.questions, &response.answers).expect("a fault");
            let evidence = check_answers(&asked, &response).unwrap_err();
            assert_eq!(evidence, fault.evidence());
            let for_others = evidence.to_string().starts_with(MISMATCHED);
            (matches!(fault, Fault::Mismatched(_)), for_others, fault.refusal().to_string())
        };
        let mismatched = |name: &str, answer: Answer| {
            let (is, says, told) = with(name, answer);
            is && says && told.starts_with("502 mismatched_answers")
        };
        let malformed = |name: &str, answer: Answer| {
            let (is, says, told) = with(name, answer);
            !is && !says && told.starts_with("502 invalid_upstream_response")
        };
        assert!(mismatched("extra", Answer::noul(0.5)));
        assert!(mismatched("yes", Answer::choice([("a", 1.0)])));
        assert!(mismatched("pick", Answer::choice([("a", 0.5), ("c", 0.5)])));
        assert!(mismatched(
            "pick",
            Answer {
                choice: Some("c".into()),
                ..Answer::choice([("a", 0.5), ("b", 0.5)])
            }
        ));
        assert!(malformed(
            "yes",
            Answer {
                kind: None,
                ..Answer::noul(0.5)
            }
        ));
        assert!(malformed(
            "yes",
            Answer {
                noul: None,
                ..Answer::noul(0.5)
            }
        ));
        assert!(malformed(
            "pick",
            Answer {
                probabilities: None,
                ..Answer::choice([("a", 0.5), ("b", 0.5)])
            }
        ));
    }

    #[test]
    fn a_response_is_made_whole() {
        let asked = Request::new("s")
            .question("yes", Question::noul("?"))
            .question("pick", Question::choice("?", [("a", None), ("b", None), ("c", None)]))
            .question("rate", Question::score("?", [Json::from("low"), serde_json::json!({"is": "high"})]));
        let bare = |kind: Kind, probabilities: &[(&str, f64)]| Answer {
            kind: Some(kind),
            probabilities: Some(probabilities.iter().map(|(label, p)| (label.to_string(), *p)).collect()),
            ..Answer::default()
        };
        let mut response = Response::default();
        response.answers.insert("yes".into(), Answer::noul(0.25));
        // Two as likely as each other, given in another order than they were declared in.
        response
            .answers
            .insert("pick".into(), bare(Kind::Choice, &[("c", 0.4), ("b", 0.4), ("a", 0.2)]));
        response
            .answers
            .insert("rate".into(), bare(Kind::Score, &[("1", 0.75), ("0", 0.25)]));
        let given = response.clone();
        complete(&asked.questions, &mut response).unwrap();
        assert_eq!(response.answers["yes"], given.answers["yes"]);
        let pick = &response.answers["pick"];
        assert_eq!(pick.choice.as_deref(), Some("b"));
        assert_eq!(pick.confidence, Some(choice_confidence([0.2, 0.4, 0.4])));
        assert_eq!(pick.probabilities, given.answers["pick"].probabilities);
        let rate = &response.answers["rate"];
        assert_eq!((rate.score, rate.confidence), (Some(0.75), Some(score_confidence(&[0.25, 0.75]))));
        assert_eq!(rate.legend, Some(serde_json::json!({"0": "low", "1": {"is": "high"}})));
        // It is what the constructors of an answer work out, and a second time changes nothing.
        assert_eq!(rate.score, Answer::score([0.25, 0.75]).score);
        let whole = response.clone();
        complete(&asked.questions, &mut response).unwrap();
        assert_eq!(response, whole);

        // What was said is kept.
        let mut said = Response::default();
        said.answers.insert("yes".into(), Answer::noul(0.25));
        said.answers.insert(
            "pick".into(),
            Answer {
                choice: Some("a".into()),
                confidence: Some(0.125),
                ..bare(Kind::Choice, &[("a", 0.2), ("b", 0.4), ("c", 0.4)])
            },
        );
        said.answers.insert(
            "rate".into(),
            Answer {
                score: Some(0.0),
                confidence: Some(0.5),
                legend: Some(serde_json::json!({"0": "theirs"})),
                ..bare(Kind::Score, &[("0", 0.25), ("1", 0.75)])
            },
        );
        let kept = said.clone();
        complete(&asked.questions, &mut said).unwrap();
        assert_eq!(said, kept);

        // What cannot be written cannot be sent.
        for unwritable in [f64::NAN, f64::INFINITY] {
            let mut response = given.clone();
            response.answers["yes"].noul = Some(unwritable);
            assert_eq!(complete(&asked.questions, &mut response).unwrap_err().status(), Some(502));
            let mut response = given.clone();
            response.answers["rate"].probabilities = Some(IndexMap::from([("0".to_string(), unwritable), ("1".to_string(), 0.5)]));
            let error = complete(&asked.questions, &mut response).unwrap_err();
            assert!(error.to_string().contains("invalid_response"), "{error}");
        }
    }
}
