//! Routing, in the process: backends that are functions, and what the gateway does with them.

mod common;

use std::sync::Arc;

use common::{answers, api, failing, model};
use indexmap::IndexMap;
use one_system::{Backend, Capabilities, Gateway, Limits, Picked, Rule, SELECTOR_INSTRUCTIONS, SELECTOR_QUESTION, Selection, Stage};
use serde_json::json;
use systemone::{Answer, DecisionModel, Error, Kind, Question, Request, Scripted, Usage};

const LOCAL: &str = "Short, plain English.";
const HOSTED: &str = "Everything else.";

fn refund() -> Request {
    Request::new("Please refund the duplicate charge.").question("refund", Question::noul("Does the customer request a refund?"))
}

/// A selector that names a backend, this sure, and answers ordinary questions as well.
fn selector(name: &str, choose: &'static str, confidence: f64, tokens: u64) -> Arc<Scripted> {
    Arc::new(
        Scripted::new(move |request| {
            let mut given = answers(request);
            if let Some(question) = request.questions.get(SELECTOR_QUESTION) {
                let labels = question.labels();
                let mut answer = Answer::chose(choose, labels.iter().map(String::as_str));
                answer.confidence = Some(confidence);
                given.insert(SELECTOR_QUESTION.to_string(), answer);
            }
            given
        })
        .named(name)
        .tokens(tokens),
    )
}

/// The gateway most tests use: a local and a hosted backend, the hosted one selecting.
fn two(local: Arc<Scripted>, hosted: Arc<Scripted>) -> one_system::GatewayBuilder {
    Gateway::builder("demo")
        .backend(Backend::new("local", local).description(LOCAL))
        .backend(Backend::new("hosted", hosted).description(HOSTED))
        .selector("hosted")
}

#[test]
fn a_backend_named_is_asked_directly() {
    let (local, hosted) = (model("laya-english", 7), model("jev-1.13.0", 300));
    let gateway = Gateway::builder("demo")
        // Its limits would keep this request away, were the gateway choosing.
        .backend(
            Backend::new("local", local.clone())
                .model("laya-english")
                .limits(Limits::default().max_characters(5)),
        )
        .backend(Backend::new("hosted", hosted.clone()))
        .selector("hosted")
        .build()
        .unwrap();

    let routed = gateway.route(&refund().model("local"));
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(
        response.usage,
        Usage {
            input_tokens: 7,
            output_tokens: 0
        }
    );
    assert_eq!(routed.route.eligible, ["local"]);
    assert_eq!(routed.route.backend.as_deref(), Some("local"));
    assert_eq!(routed.route.picked, Some(Picked::Direct));
    assert_eq!(routed.route.failed, None);

    // The backend was asked once, with its own model's name and the caller's state and questions.
    let asked = local.requests();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0], refund().model("laya-english"));
    // Nobody selected anything.
    assert!(hosted.requests().is_empty());
}

#[test]
fn a_model_nobody_knows_is_refused() {
    let gateway = two(model("laya", 0), model("jev", 0)).build().unwrap();
    let routed = gateway.route(&refund().model("gpt-9"));
    assert_eq!(
        api(&routed.result.unwrap_err()),
        (422, "unsupported_model".into(), "Unknown model; see GET /v1/models".into())
    );
    assert_eq!(routed.route.failed, Some(Stage::Gateway));
}

#[test]
fn a_request_that_is_not_one_is_refused() {
    let gateway = two(model("laya", 0), model("jev", 0)).build().unwrap();
    let nothing_asked = Request::new("state");
    assert_eq!(api(&gateway.decide(&nothing_asked).unwrap_err()).1, "schema_validation");
    let no_state = Request::new(json!(null)).question("q", Question::noul("?"));
    assert_eq!(api(&gateway.decide(&no_state).unwrap_err()).1, "schema_validation");
}

#[test]
fn a_backend_named_must_still_be_able() {
    let local = model("laya", 0);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("local", local.clone()).capabilities(Capabilities::new([Kind::Choice])))
        .backend(Backend::new("hosted", model("jev", 0)))
        .selector("hosted")
        .build()
        .unwrap();
    let error = gateway.decide(&refund().model("local")).unwrap_err();
    assert_eq!(
        api(&error),
        (
            422,
            "unsupported_capability".into(),
            "No requested backend supports these inputs".into()
        )
    );
    // It is not sent to the backend that could have taken it.
    assert!(local.requests().is_empty());
}

#[test]
fn one_eligible_backend_is_asked_without_a_selector() {
    let (local, hosted) = (model("laya-english", 7), selector("jev-1.13.0", "local", 1.0, 300));
    let gateway = Gateway::builder("demo")
        // A noul is not something this local model answers.
        .backend(Backend::new("local", local.clone()).capabilities(Capabilities::new([Kind::Choice, Kind::Score])))
        .backend(Backend::new("hosted", hosted.clone()))
        .selector("local")
        .fallback("local")
        .build()
        .unwrap();

    // By the gateway's name, and by no name at all.
    for request in [refund().model("demo"), refund()] {
        hosted.clear();
        let routed = gateway.route(&request);
        let response = routed.result.unwrap();
        assert_eq!(response.model, "jev-1.13.0");
        assert_eq!(response.usage.input_tokens, 300);
        assert_eq!(routed.route.eligible, ["hosted"]);
        assert_eq!(routed.route.picked, Some(Picked::Only));
        // One request reached the hosted model: the caller's, not a selection.
        assert_eq!(hosted.requests().len(), 1);
        assert_eq!(hosted.requests()[0].questions.keys().collect::<Vec<_>>(), ["refund"]);
    }
    // The selector, and fallback, was never asked: there was nothing to select.
    assert!(local.requests().is_empty());
}

#[test]
fn nobody_able_is_a_refusal() {
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("a", model("a", 0)).capabilities(Capabilities::new([Kind::Choice])))
        .backend(Backend::new("b", model("b", 0)).capabilities(Capabilities::all().max_questions(1).structured_state(false)))
        .selector("a")
        .build()
        .unwrap();
    // `a` answers no noul, and `b` takes one question about a text: not two, and not about an object.
    let two_questions = refund().question("again", Question::noul("?"));
    assert_eq!(api(&gateway.decide(&two_questions).unwrap_err()).1, "unsupported_capability");
    let about_an_object = Request::new(json!({"k": 1})).question("q", Question::noul("?"));
    assert_eq!(api(&gateway.decide(&about_an_object).unwrap_err()).1, "unsupported_capability");
    assert!(gateway.decide(&refund()).is_ok());
}

#[test]
fn several_eligible_are_decided_by_the_selector() {
    let (local, hosted) = (model("laya-english", 7), selector("jev-1.13.0", "local", 0.9, 300));
    let gateway = two(local.clone(), hosted.clone()).build().unwrap();

    let routed = gateway.route(&refund());
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(routed.route.eligible, ["hosted", "local"]);
    assert_eq!(routed.route.backend.as_deref(), Some("local"));
    assert_eq!(routed.route.picked, Some(Picked::Selected { confidence: 0.9 }));
    // What the selector used and what the backend used.
    assert_eq!(
        response.usage,
        Usage {
            input_tokens: 307,
            output_tokens: 0
        }
    );
    assert_eq!(response.answers.keys().collect::<Vec<_>>(), ["refund"]);

    // The selector was asked one choice, among the eligible backends by their descriptions, in order of id.
    let asked = hosted.requests();
    assert_eq!(asked.len(), 1);
    let question = &asked[0].questions[SELECTOR_QUESTION];
    assert_eq!(asked[0].questions.len(), 1);
    assert_eq!(question.kind, Kind::Choice);
    assert_eq!(question.instructions, json!(SELECTOR_INSTRUCTIONS));
    assert_eq!(question.criteria, Some(json!({"hosted": HOSTED, "local": LOCAL})));
    assert_eq!(question.labels(), ["hosted", "local"]);
    // It is not the fallback, so it sees the questions and the size of the state, and not the state.
    assert_eq!(
        asked[0].state,
        json!({
            "question_definitions": [{"instructions": "Does the customer request a refund?", "type": "noul"}],
            "input_summary": {"characters": 37, "non_ascii_letter_fraction": 0.0},
        })
    );
    assert!(!asked[0].state.to_string().contains("duplicate charge"));
    // The backend got the caller's request, once.
    assert_eq!(local.requests(), [refund()]);
}

#[test]
fn a_selector_that_is_the_fallback_sees_the_state_and_only_the_state() {
    let local = selector("laya-english", "hosted", 1.0, 7);
    let hosted = model("jev-1.13.0", 300);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("local", local.clone()).description(LOCAL).model("laya-english"))
        .backend(Backend::new("hosted", hosted.clone()).description(HOSTED))
        .selector("local")
        .fallback("local")
        .build()
        .unwrap();
    let response = gateway.decide(&refund()).unwrap();
    assert_eq!(response.model, "jev-1.13.0");
    let asked = &local.requests()[0];
    assert_eq!(asked.state, json!("Please refund the duplicate charge."));
    // The selector is asked under its own model's name.
    assert_eq!(asked.model, "laya-english");
}

#[test]
fn question_definitions_are_in_order_of_name() {
    let hosted = selector("jev", "local", 1.0, 0);
    let gateway = two(model("laya", 0), hosted.clone()).build().unwrap();
    let request = Request::new(json!({"m": "éa界😀 9"}))
        .question("z", Question::noul("Zulu"))
        .question("a", Question::choice("Alpha", [("x", None), ("y", None)]));
    gateway.decide(&request).unwrap();
    assert_eq!(
        hosted.requests()[0].state,
        json!({
            "question_definitions": [
                {"instructions": "Alpha", "type": "choice", "criteria": {"x": null, "y": null}},
                {"instructions": "Zulu", "type": "noul"},
            ],
            "input_summary": {"characters": 14, "non_ascii_letter_fraction": 0.5},
        })
    );
}

#[test]
fn soft_limits_keep_a_request_away_and_are_set_aside_when_nobody_is_left() {
    let (local, hosted) = (model("laya-english", 7), selector("jev-1.13.0", "hosted", 1.0, 300));
    let small = Capabilities::all().max_questions(2);
    let build = |hosted_limits: Limits| {
        Gateway::builder("demo")
            .backend(
                Backend::new("local", local.clone())
                    .capabilities(small.clone())
                    .limits(Limits::default().max_characters(20).max_non_ascii_letter_fraction(0.02)),
            )
            .backend(Backend::new("hosted", hosted.clone()).limits(hosted_limits))
            .backend(Backend::new("never", model("never", 0)).capabilities(Capabilities::new([Kind::Score])))
            .selector("hosted")
            .build()
            .unwrap()
    };
    let short = Request::new("Refund, please.").question("refund", Question::noul("?"));
    let long = refund();

    // Inside everyone's limits: two are eligible, and the selector decides.
    let gateway = build(Limits::default());
    assert_eq!(gateway.route(&short).route.eligible, ["hosted", "local"]);
    // At a limit is inside it: the state, with its quotes, is 17 characters.
    assert_eq!(short.state.to_string().chars().count(), 17);
    let exact = Gateway::builder("exact")
        .backend(Backend::new("local", local.clone()).limits(Limits::default().max_characters(17)))
        .backend(Backend::new("hosted", hosted.clone()).limits(Limits::default().max_characters(16)))
        .selector("hosted")
        .build()
        .unwrap();
    assert_eq!(exact.route(&short).route.eligible, ["local"]);

    // Too long for the local model: one is left, and no selector is asked.
    hosted.clear();
    let routed = gateway.route(&long);
    assert_eq!(routed.route.eligible, ["hosted"]);
    assert_eq!(routed.route.picked, Some(Picked::Only));
    assert_eq!(hosted.requests().len(), 1);

    // Not English enough for the local model.
    let german = Request::new("Grüße").question("refund", Question::noul("?"));
    assert_eq!(gateway.route(&german).route.eligible, ["hosted"]);

    // Too long for everyone: the limits are set aside, and those that are able are back.
    // The backend that cannot answer a noul is not among them.
    let gateway = build(Limits::default().max_characters(10));
    let routed = gateway.route(&long);
    assert_eq!(routed.route.eligible, ["hosted", "local"]);
    assert!(routed.result.is_ok());

    // Limits count questions and the widest question too.
    let gateway = build(Limits::default().max_questions(1).max_criteria(2));
    let two_questions = short.clone().question("again", Question::noul("?"));
    assert_eq!(gateway.route(&two_questions).route.eligible, ["local"]);
    let wide = Request::new("Refund.").question("which", Question::choice("?", [("a", None), ("b", None), ("c", None)]));
    assert_eq!(gateway.route(&wide).route.eligible, ["local"]);
}

#[test]
fn a_selector_that_fails_sends_the_request_to_the_fallback() {
    let outage = Error::Api {
        status: 503,
        code: "overloaded".into(),
        message: "try later".into(),
    };
    let local = model("laya-english", 7);
    let hosted = failing("jev", outage.clone());
    let gateway = two(local.clone(), hosted.clone()).fallback("local").build().unwrap();

    let routed = gateway.route(&refund());
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(routed.route.backend.as_deref(), Some("local"));
    assert_eq!(routed.route.picked, Some(Picked::Fallback { error: outage.clone() }));
    // The selector was asked once, and what it used (nothing it reported) is not counted.
    assert_eq!(hosted.requests().len(), 1);
    assert_eq!(response.usage.input_tokens, 7);

    // With no fallback there is nowhere safe to go, and the selector's error is the caller's.
    let gateway = two(local.clone(), hosted.clone()).build().unwrap();
    local.clear();
    let routed = gateway.route(&refund());
    assert_eq!(routed.result.unwrap_err(), outage);
    assert_eq!(routed.route.failed, Some(Stage::Selector));
    assert_eq!(routed.route.backend, None);
    assert!(local.requests().is_empty());
}

#[test]
fn a_selector_that_answers_something_else_is_a_selector_that_failed() {
    // It names a backend that was not on offer.
    let stray = Arc::new(Scripted::new(|_| [(SELECTOR_QUESTION, Answer::chose("elsewhere", ["elsewhere", "local"]))]).tokens(50));
    let local = model("laya-english", 7);
    let gateway = two(local.clone(), stray.clone()).fallback("local").build().unwrap();
    let routed = gateway.route(&refund());
    assert_eq!(routed.result.unwrap().usage.input_tokens, 7);
    assert!(matches!(routed.route.picked, Some(Picked::Fallback { .. })));

    // Without a fallback, that is an error of the gateway's own making.
    let gateway = two(local, stray).build().unwrap();
    let routed = gateway.route(&refund());
    assert_eq!(
        api(&routed.result.unwrap_err()),
        (
            502,
            "mismatched_answers".into(),
            "Upstream answers do not match the requested question IDs, types, or choices".into()
        )
    );
    assert_eq!(routed.route.failed, Some(Stage::Gateway));
}

#[test]
fn a_selector_surer_than_sure_is_not_fallen_back_from() {
    let local = model("laya-english", 7);
    let gateway = two(local.clone(), selector("jev", "local", 1.5, 0))
        .fallback("local")
        .build()
        .unwrap();
    let routed = gateway.route(&refund());
    assert_eq!(
        api(&routed.result.unwrap_err()),
        (
            502,
            "invalid_selector".into(),
            "Upstream returned an invalid backend selection".into()
        )
    );
    assert!(local.requests().is_empty());
}

#[test]
fn leaving_the_fallback_takes_a_selector_that_is_sure_enough() {
    let (local, hosted) = (selector("laya-english", "hosted", 0.6, 7), model("jev-1.13.0", 300));
    let build = |threshold: f64| {
        Gateway::builder("demo")
            .backend(Backend::new("local", local.clone()).description(LOCAL))
            .backend(Backend::new("hosted", hosted.clone()).description(HOSTED))
            .selector("local")
            .fallback("local")
            .escalation_confidence(threshold)
            .build()
            .unwrap()
    };
    // Less sure than required: it stays, and the selector's tokens are still counted.
    let routed = build(0.8).route(&refund());
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(
        routed.route.picked,
        Some(Picked::Withheld {
            proposed: "hosted".into(),
            confidence: 0.6
        })
    );
    assert_eq!(response.usage.input_tokens, 14);
    // As sure as required is sure enough.
    let routed = build(0.6).route(&refund());
    assert_eq!(routed.result.unwrap().model, "jev-1.13.0");
    assert_eq!(routed.route.picked, Some(Picked::Selected { confidence: 0.6 }));
}

#[test]
fn a_backend_that_fails_is_returned_as_it_is() {
    for failure in [
        Error::Api {
            status: 429,
            code: "rate_limited".into(),
            message: "slow down".into(),
        },
        Error::Transport("connection reset".into()),
        Error::Evidence("nonsense".into()),
    ] {
        let local = failing("laya", failure.clone());
        let hosted = selector("jev", "local", 1.0, 300);
        // A fallback is no second chance for a backend that failed.
        let gateway = two(local.clone(), hosted.clone()).fallback("hosted").build().unwrap();
        let routed = gateway.route(&refund());
        assert_eq!(routed.result.unwrap_err(), failure);
        assert_eq!(routed.route.failed, Some(Stage::Backend));
        assert_eq!(routed.route.backend.as_deref(), Some("local"));
        // Asked once, and nobody else was asked the caller's questions.
        assert_eq!(local.requests().len(), 1);
        assert_eq!(hosted.requests().len(), 1);
        assert!(hosted.requests()[0].questions.contains_key(SELECTOR_QUESTION));
    }
}

#[test]
fn a_backend_that_answers_other_questions_is_an_error() {
    let confused = Arc::new(Scripted::fallible(|_| {
        Ok(IndexMap::from([("refund".to_string(), Answer::chose("a", ["a", "b"]))]))
    }));
    let gateway = Gateway::builder("solo").backend(Backend::new("only", confused)).build().unwrap();
    let routed = gateway.route(&refund());
    assert_eq!(api(&routed.result.unwrap_err()).1, "mismatched_answers");
    assert_eq!(routed.route.failed, Some(Stage::Gateway));
}

#[test]
fn usage_that_cannot_be_added_is_an_error() {
    let gateway = two(model("laya", u64::MAX), selector("jev", "local", 1.0, 1)).build().unwrap();
    assert_eq!(
        api(&gateway.decide(&refund()).unwrap_err()),
        (502, "invalid_usage".into(), "Upstream token usage cannot be aggregated".into())
    );
}

/// A selector for the feature mode: it gives `math_or_logic` this probability.
fn domain_selector(math: f64, tokens: u64) -> Arc<Scripted> {
    Arc::new(
        Scripted::new(move |request| {
            if request.questions.contains_key("domain") {
                let mut given = answers(request);
                given.insert("domain".into(), Answer::choice([("math_or_logic", math), ("chitchat", 1.0 - math)]));
                return given;
            }
            answers(request)
        })
        .named("laya-english")
        .tokens(tokens),
    )
}

fn domain() -> Question {
    Question::choice(
        "What domain does `request` belong to?",
        [
            ("math_or_logic", "mathematics, logic puzzles".to_string()),
            ("chitchat", "small talk".to_string()),
        ],
    )
}

#[test]
fn a_rule_escalates_when_its_value_is_strictly_above_its_threshold() {
    let build = |math: f64| {
        let (local, hosted) = (domain_selector(math, 7), model("jev-1.13.0", 300));
        let gateway = Gateway::builder("privacy-demo")
            .backend(Backend::new("local", local.clone()).description(LOCAL))
            .backend(Backend::new("hosted", hosted.clone()).description(HOSTED))
            .selector("local")
            .fallback("local")
            // Not used with rules: a sure selector is not what escalates here.
            .escalation_confidence(1.0)
            .selection(Selection::new([("domain", domain())], "hosted").rule(Rule::choice("domain", "math_or_logic", 0.5)))
            .build()
            .unwrap();
        (gateway, local, hosted)
    };
    let request = Request::new("17 + 25 = 42")
        .question("z", Question::noul("Zulu task"))
        .question("a", Question::noul("Alpha task"));

    // Above the threshold: it goes to the hosted model.
    let (gateway, local, hosted) = build(0.51);
    let routed = gateway.route(&request);
    let response = routed.result.unwrap();
    assert_eq!(response.model, "jev-1.13.0");
    assert_eq!(
        routed.route.picked,
        Some(Picked::Escalated {
            question: "domain".into(),
            value: 0.51
        })
    );
    assert_eq!(response.usage.input_tokens, 307);
    assert_eq!(hosted.requests(), std::slice::from_ref(&request));
    // The selector was asked the routing questions about the task, and saw nothing of the state.
    let asked = &local.requests()[0];
    assert_eq!(asked.state, json!({"request": "Alpha task Zulu task"}));
    assert_eq!(asked.questions, IndexMap::from([("domain".to_string(), domain())]));

    // At the threshold: it stays.
    let (gateway, local, hosted) = build(0.5);
    let routed = gateway.route(&request);
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(routed.route.picked, Some(Picked::Stayed));
    // The local model was asked twice: to select, and to answer.
    assert_eq!(local.requests().len(), 2);
    assert_eq!(response.usage.input_tokens, 14);
    assert!(hosted.requests().is_empty());
}

#[test]
fn rules_read_nouls_and_scores_too() {
    let build = |sensitive: f64, hard: [f64; 3], rules: Vec<Rule>| {
        let selector = Arc::new(Scripted::new(move |request| {
            if request.questions.contains_key("sensitive") {
                return IndexMap::from([
                    ("sensitive".to_string(), Answer::noul(sensitive)),
                    ("hard".to_string(), Answer::score(hard)),
                ]);
            }
            answers(request)
        }));
        let questions = [
            ("sensitive", Question::noul("Is `request` about something private?")),
            ("hard", Question::score("How hard is `request`?", ["easy", "middling", "hard"])),
        ];
        let mut selection = Selection::new(questions, "hosted");
        selection.rules = rules;
        Gateway::builder("demo")
            .backend(Backend::new("local", selector))
            .backend(Backend::new("hosted", model("jev", 0)))
            .selector("local")
            .fallback("local")
            .selection(selection)
            .build()
            .unwrap()
    };
    let picked = |gateway: Gateway| gateway.route(&refund()).route.picked.unwrap();
    assert_eq!(
        picked(build(0.9, [1.0, 0.0, 0.0], vec![Rule::above("sensitive", 0.8)])),
        Picked::Escalated {
            question: "sensitive".into(),
            value: 0.9
        }
    );
    assert_eq!(
        picked(build(0.8, [1.0, 0.0, 0.0], vec![Rule::above("sensitive", 0.8)])),
        Picked::Stayed
    );
    // A score is read as its expected level: here 0.5 * 1 + 0.5 * 2.
    let both = vec![Rule::above("sensitive", 0.8), Rule::above("hard", 1.0)];
    assert_eq!(
        picked(build(0.1, [0.0, 0.5, 0.5], both.clone())),
        Picked::Escalated {
            question: "hard".into(),
            value: 1.5
        }
    );
    assert_eq!(picked(build(0.1, [0.5, 0.0, 0.5], both)), Picked::Stayed);
}

#[test]
fn a_selector_that_fails_in_the_feature_mode_stays_on_the_fallback() {
    // It cannot answer the routing question, but answers the caller's.
    let local = Arc::new(Scripted::fallible(|request| {
        if request.questions.contains_key("domain") {
            return Err(Error::Transport("the selector is away".into()));
        }
        Ok(answers(request))
    }));
    let hosted = model("jev", 0);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("local", local))
        .backend(Backend::new("hosted", hosted.clone()))
        .selector("local")
        .fallback("local")
        .selection(Selection::new([("domain", domain())], "hosted").rule(Rule::choice("domain", "math_or_logic", 0.0)))
        .build()
        .unwrap();
    let routed = gateway.route(&refund());
    assert!(routed.result.is_ok());
    assert_eq!(routed.route.backend.as_deref(), Some("local"));
    assert!(hosted.requests().is_empty());
}

#[test]
fn wherever_a_request_is_sent_the_backend_must_be_able() {
    // The fallback cannot answer a noul; the selector fails; nothing is substituted.
    let local = model("laya", 0);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("a", model("a", 0)))
        .backend(Backend::new("b", failing("b", Error::Transport("down".into()))))
        .backend(Backend::new("local", local.clone()).capabilities(Capabilities::new([Kind::Choice])))
        .selector("b")
        .fallback("local")
        .build()
        .unwrap();
    let routed = gateway.route(&refund());
    assert_eq!(routed.route.eligible, ["a", "b"]);
    assert_eq!(
        api(&routed.result.unwrap_err()),
        (
            422,
            "unsupported_capability".into(),
            "Selected backend does not support these inputs".into()
        )
    );
    assert!(local.requests().is_empty());
}

#[test]
fn a_selector_must_be_able_to_take_the_routing_request() {
    // The selector takes text only, and what a selector that is not the fallback is shown is an object.
    let picky = model("picky", 0);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("a", model("a", 0)))
        .backend(Backend::new("b", model("b", 0)))
        .backend(Backend::new("picky", picky.clone()).capabilities(Capabilities::all().structured_state(false)))
        .selector("picky")
        .build()
        .unwrap();
    let request = Request::new(json!({"k": 1})).question("q", Question::noul("?"));
    let routed = gateway.route(&request);
    assert_eq!(routed.route.eligible, ["a", "b"]);
    assert_eq!(
        api(&routed.result.unwrap_err()),
        (
            422,
            "unsupported_capability".into(),
            "Selector does not support the routing request".into()
        )
    );
    assert!(picky.requests().is_empty());
}

#[test]
fn the_cache_answers_the_same_request_again_for_nothing() {
    let (local, hosted) = (model("laya-english", 7), selector("jev-1.13.0", "local", 1.0, 300));
    let gateway = two(local.clone(), hosted.clone()).cache(8).build().unwrap();

    let first = gateway.route(&refund());
    assert!(!first.route.cached);
    let first = first.result.unwrap();
    assert_eq!(first.usage.input_tokens, 307);

    let again = gateway.route(&refund());
    assert!(again.route.cached);
    assert_eq!(again.route.backend.as_deref(), Some("local"));
    assert_eq!(again.route.picked, None);
    let again = again.result.unwrap();
    // The same answers from the same model, and nothing used, because nobody was asked.
    assert_eq!((&again.model, &again.answers), (&first.model, &first.answers));
    assert_eq!(again.usage, Usage::default());
    assert_eq!((local.requests().len(), hosted.requests().len()), (1, 1));

    // Another request is another entry: a different state, a different question, a backend named.
    let other = Request::new("Another state.").question("refund", Question::noul("Does the customer request a refund?"));
    assert!(!gateway.route(&other).route.cached);
    assert!(!gateway.route(&refund().model("local")).route.cached);
    assert!(gateway.route(&refund().model("local")).route.cached);

    // Forgotten on request.
    gateway.clear_cache();
    assert!(!gateway.route(&refund()).route.cached);

    // Off unless asked for.
    let plain = two(model("laya", 1), selector("jev", "local", 1.0, 1)).build().unwrap();
    plain.decide(&refund()).unwrap();
    assert!(!plain.route(&refund()).route.cached);
}

#[test]
fn the_cache_keeps_no_failures_and_answers_nothing_it_would_refuse() {
    let flaky_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = flaky_calls.clone();
    let flaky = Scripted::fallible(move |request| {
        if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            return Err(Error::Transport("first time unlucky".into()));
        }
        Ok(answers(request))
    });
    let gateway = Gateway::builder("solo")
        .backend(Backend::new("only", flaky))
        .cache(8)
        .build()
        .unwrap();
    assert!(gateway.decide(&refund()).is_err());
    let second = gateway.route(&refund());
    assert!(second.result.is_ok() && !second.route.cached);
    assert!(gateway.route(&refund()).route.cached);
    assert_eq!(flaky_calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    // A request for a model nobody knows is refused, cache or no cache.
    assert_eq!(api(&gateway.decide(&refund().model("gpt-9")).unwrap_err()).1, "unsupported_model");
}

#[test]
fn a_gateway_routes_among_gateways() {
    // An inner gateway that chooses between two local models, as one backend of an outer one.
    let (small, large, jev) = (
        model("laya-small", 1),
        model("laya-large", 10),
        selector("jev-1.13.0", "locals", 1.0, 100),
    );
    let inner = Gateway::builder("locals")
        .backend(Backend::new("small", small.clone()).limits(Limits::default().max_characters(10)))
        .backend(Backend::new("large", large.clone()).limits(Limits::default().max_characters(1000)))
        .selector("large")
        .build()
        .unwrap();
    let outer = Gateway::builder("everything")
        .backend(Backend::new("locals", inner.clone()).description("Models on this machine."))
        .backend(Backend::new("hosted", jev.clone()).description("Hosted Jev."))
        .selector("hosted")
        .build()
        .unwrap();
    assert_eq!(outer.name(), "everything");

    let response = outer.decide(&refund()).unwrap();
    // The outer selector chose the inner gateway, whose limits left it one model.
    assert_eq!(response.model, "laya-large");
    assert_eq!(response.usage.input_tokens, 110);
    assert!(small.requests().is_empty());
    assert_eq!(large.requests(), [refund()]);

    // A boxed trait object is a backend like any other.
    let boxed: Box<dyn DecisionModel> = Box::new(inner);
    let wrapped = Gateway::builder("wrapped").backend(Backend::new("inner", boxed)).build().unwrap();
    assert_eq!(wrapped.decide(&refund()).unwrap().model, "laya-large");
}

#[test]
fn a_backend_that_gives_no_name_is_named_by_its_id() {
    let anonymous = Scripted::new(answers).named("");
    let gateway = Gateway::builder("solo").backend(Backend::new("only", anonymous)).build().unwrap();
    assert_eq!(gateway.decide(&refund()).unwrap().model, "only");
}

#[test]
fn what_a_gateway_lists() {
    let gateway = Gateway::builder("demo")
        .backend(
            Backend::new("local", model("laya", 0))
                .description(LOCAL)
                .capabilities(Capabilities::all().max_questions(16).min_criteria(2).structured_state(true)),
        )
        .backend(Backend::new("hosted", model("jev", 0)).description(HOSTED))
        .selector("hosted")
        .fallback("local")
        .build()
        .unwrap();
    assert_eq!(gateway.name(), "demo");
    assert_eq!((gateway.selector(), gateway.fallback()), ("hosted", Some("local")));
    assert_eq!(
        gateway.backends().map(|backend| backend.id.as_str()).collect::<Vec<_>>(),
        ["hosted", "local"]
    );
    assert_eq!(gateway.backend("local").unwrap().description, LOCAL);
    assert!(format!("{gateway:?}").contains("demo"));

    let models = gateway.models();
    let names: Vec<&str> = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["demo", "hosted", "local"]);
    assert_eq!(models["models"][2]["description"], LOCAL);
    assert!(
        models["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|model| model["release_date"].is_string())
    );

    assert_eq!(
        gateway.capabilities(),
        json!({"version": 1, "models": [
            {"name": "hosted", "capabilities": null},
            {"name": "local", "capabilities": {"question_types": ["choice", "score", "noul"], "max_questions": 16, "min_criteria": 2, "structured_state": true}},
        ]})
    );
}

#[test]
fn gateways_that_cannot_be_built() {
    let backend = |id: &str| Backend::new(id, model("m", 0));
    let broken = |builder: one_system::GatewayBuilder| builder.build().unwrap_err().to_string();
    assert!(broken(Gateway::builder("has space").backend(backend("a"))).contains("configuration name"));
    assert!(broken(Gateway::builder("demo")).contains("at least one backend"));
    assert!(broken(Gateway::builder("demo").backend(backend("demo"))).contains("differ from the configuration name"));
    assert!(broken(Gateway::builder("demo").backend(backend("a/b"))).contains("backend IDs must contain"));
    assert!(broken(Gateway::builder("demo").backend(backend("a")).backend(backend("a"))).contains("unique"));
    // With more than one backend, one must be named to select.
    assert!(broken(Gateway::builder("demo").backend(backend("a")).backend(backend("b"))).contains("selector must name"));
    assert!(broken(Gateway::builder("demo").backend(backend("a")).selector("b")).contains("selector must name"));
    assert!(broken(Gateway::builder("demo").backend(backend("a")).fallback("b")).contains("fallback must name"));
    assert!(broken(Gateway::builder("demo").backend(backend("a")).escalation_confidence(0.5)).contains("requires a fallback"));
    assert!(
        broken(
            Gateway::builder("demo")
                .backend(backend("a"))
                .fallback("a")
                .escalation_confidence(1.5)
        )
        .contains("between 0 and 1")
    );
    assert!(broken(Gateway::builder("demo").backend(backend("a").capabilities(Capabilities::new([])))).contains("question_types"));
    assert!(broken(Gateway::builder("demo").backend(backend("a").limits(Limits::default().max_criteria(1)))).contains("at least 2"));

    let two = || Gateway::builder("demo").backend(backend("a")).backend(backend("b")).selector("a");
    let rule = Rule::choice("domain", "math_or_logic", 0.5);
    let selection = |to: &str| Selection::new([("domain", domain())], to).rule(rule.clone());
    assert!(broken(two().selection(selection("b"))).contains("selection requires a fallback"));
    assert!(broken(two().fallback("a").selection(selection("c"))).contains("escalate_to must name"));
    assert!(broken(two().fallback("a").selection(selection("a"))).contains("must differ from the fallback"));
    assert!(broken(two().fallback("a").selection(Selection::new([("domain", domain())], "b"))).contains("at least one rule"));
    let asks = |rule: Rule| {
        two()
            .fallback("a")
            .selection(Selection::new([("domain", domain())], "b").rule(rule))
    };
    assert!(broken(asks(Rule::above("elsewhere", 0.5))).contains("must name a question"));
    assert!(broken(asks(Rule::choice("domain", "poetry", 0.5))).contains("configured criterion"));
    assert!(broken(asks(Rule::above("domain", 0.5))).contains("configured criterion"));
    assert!(broken(asks(Rule::choice("domain", "chitchat", 1.5))).contains("threshold"));
    assert!(asks(Rule::choice("domain", "chitchat", 1.0)).build().is_ok());
    // One backend needs no selector named.
    assert_eq!(
        Gateway::builder("solo").backend(backend("only")).build().unwrap().selector(),
        "only"
    );
}

#[test]
fn a_vowel_sign_is_no_letter_and_does_not_send_a_state_away() {
    // Sixty letters and one Hindi syllable: a consonant, and the sign that gives it a vowel.
    // The Go gateway counts 61 letters, one of them outside ASCII, which is 0.02 and inside
    // the limit the registries ship with.
    let state = format!("{} \u{915}\u{93e}", "a".repeat(60));
    assert_eq!(one_system::Summary::of(&json!(state)).non_ascii_letter_fraction, 0.02);
    let local = model("laya-english", 7);
    let hosted = selector("jev", "local", 1.0, 300);
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("local", local.clone()).limits(Limits::default().max_non_ascii_letter_fraction(0.02)))
        .backend(Backend::new("hosted", hosted.clone()))
        .selector("hosted")
        .fallback("local")
        .build()
        .unwrap();
    let routed = gateway.route(&Request::new(state).question("refund", Question::noul("?")));
    assert_eq!(routed.route.eligible, ["hosted", "local"]);
    assert_eq!(routed.route.backend.as_deref(), Some("local"));
    assert_eq!(local.requests().len(), 1);
}
