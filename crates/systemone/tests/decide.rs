//! The arithmetic of a decision, with the values the DSPy tests pin.

use indexmap::IndexMap;
use systemone::{Answer, Decision, Error, Params, Question, Request, decide};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-12
}

fn noul(probability: f64, threshold: f64) -> (bool, f64) {
    let question = Question::noul("Is service blocked?");
    match decide("urgent", &question, &Answer::noul(probability), &Params::threshold(threshold)).unwrap() {
        Decision::Noul(noul) => (noul.value, noul.confidence),
        other => panic!("not a noul: {other:?}"),
    }
}

#[test]
fn a_noul_is_a_yes_at_or_over_its_threshold() {
    for (probability, threshold, value, confidence) in [
        (0.0, 0.75, false, 1.0),
        (0.6, 0.75, false, 0.2),
        (0.75, 0.75, true, 0.0),
        (0.9, 0.75, true, 0.2),
        (1.0, 0.75, true, 1.0 / 3.0),
        (0.8, 0.7, true, 1.0 / 7.0),
        (0.8, 0.9, false, 1.0 / 9.0),
        (0.0, 0.0, true, 0.0),
        (0.99, 1.0, false, 0.01),
        (1.0, 1.0, true, 0.0),
        (0.98, 0.5, true, 0.96),
    ] {
        let (decided, sure) = noul(probability, threshold);
        assert_eq!(decided, value, "p={probability} t={threshold}");
        assert!(
            (sure - confidence).abs() < 1e-9,
            "p={probability} t={threshold}: {sure} != {confidence}"
        );
    }
}

#[test]
fn a_score_is_the_expected_level_and_its_cuts_give_the_level() {
    let question = Question::score("Rate impact.", ["Minor", "Disruptive", "Blocking"]);
    let answer = Answer {
        confidence: Some(0.9),
        ..Answer::score([0.1, 0.3, 0.6])
    };
    let level = |cuts: [f64; 2]| match decide("severity", &question, &answer, &Params::cuts(cuts)).unwrap() {
        Decision::Score(score) => (score.value, score.level, score.confidence),
        other => panic!("not a score: {other:?}"),
    };
    assert!(close(level([0.5, 1.5]).0, 1.5));
    assert_eq!(level([0.5, 1.5]).1, 2);
    assert_eq!(level([0.5, 1.6]).1, 1);
    assert_eq!(level([0.5, 1.4]).1, 2);
    assert_eq!(level([0.5, 1.5]).2, 0.9);

    // A distribution that does not sum to one gives the same value.
    let short = Answer::score([0.099, 0.297, 0.594]);
    let Decision::Score(score) = decide("severity", &question, &short, &Params::default()).unwrap() else {
        panic!()
    };
    assert!((score.value - 1.5).abs() < 1e-9);
    assert_eq!(score.probabilities.keys().collect::<Vec<_>>(), ["0", "1", "2"]);

    for cuts in [vec![0.5], vec![1.5, 0.5], vec![0.0, 1.0], vec![0.5, 2.0]] {
        let refused = decide("severity", &question, &answer, &Params::cuts(cuts));
        assert!(matches!(refused, Err(Error::Invalid(_))));
    }
}

#[test]
fn a_choice_is_the_likeliest_option_after_its_weights() {
    let question = Question::choice(
        "Classify the issue.",
        [
            ("billing", "Payment issue".to_string()),
            ("technical", "Product malfunction".to_string()),
        ],
    );
    // The server's own pick is not read, and neither is the order it answered in.
    let answer = Answer {
        choice: Some("billing".into()),
        ..Answer::choice([("technical", 0.8), ("billing", 0.2)])
    };
    let Decision::Choice(choice) = decide("category", &question, &answer, &Params::default()).unwrap() else {
        panic!()
    };
    assert_eq!(choice.value, "technical");
    assert_eq!(choice.probabilities.keys().collect::<Vec<_>>(), ["billing", "technical"]);

    let weighted = decide("category", &question, &answer, &Params::weights([("technical", 0.1)])).unwrap();
    assert_eq!(weighted.value(), "billing");
    assert_eq!(weighted.as_choice().unwrap().probabilities, choice.probabilities);

    // A weighted tie goes to the option that was likeliest before weights; a raw tie to the first declared.
    let tied = decide("category", &question, &answer, &Params::weights([("billing", 4.0)])).unwrap();
    assert_eq!(tied.value(), "technical");
    let even = Answer::choice([("billing", 0.5), ("technical", 0.5)]);
    assert_eq!(decide("category", &question, &even, &Params::default()).unwrap().value(), "billing");

    let nothing = decide(
        "category",
        &question,
        &answer,
        &Params::weights([("billing", 0.0), ("technical", 0.0)]),
    );
    assert!(matches!(nothing, Err(Error::Invalid(_))));
}

#[test]
fn evidence_that_does_not_fit_the_question_is_refused() {
    let noul = Question::noul("?");
    for probability in [1.1, -0.1, f64::NAN] {
        assert!(matches!(
            decide("x", &noul, &Answer::noul(probability), &Params::default()),
            Err(Error::Evidence(_))
        ));
    }
    assert!(matches!(
        decide("x", &noul, &Answer::default(), &Params::default()),
        Err(Error::Evidence(_))
    ));

    let score = Question::score("?", ["low", "middle", "high"]);
    let levels = |pairs: &[(&str, f64)]| Answer {
        probabilities: Some(pairs.iter().map(|(label, p)| (label.to_string(), *p)).collect::<IndexMap<_, _>>()),
        confidence: Some(0.5),
        ..Answer::default()
    };
    for bad in [
        levels(&[("0", 0.5), ("1", 0.5)]),
        levels(&[("0", 0.0), ("1", 0.0), ("2", 0.0)]),
        levels(&[("0", -0.2), ("1", 0.6), ("2", 0.6)]),
        levels(&[("00", 0.2), ("1", 0.2), ("2", 0.6)]),
        levels(&[("0", 0.2), ("1", 0.2), ("3", 0.6)]),
    ] {
        assert!(
            matches!(decide("x", &score, &bad, &Params::default()), Err(Error::Evidence(_))),
            "{bad:?}"
        );
    }

    let choice = Question::choice("?", [("a", None::<String>), ("b", None)]);
    let unknown = Answer::choice([("a", 0.5), ("c", 0.5)]);
    assert!(matches!(
        decide("x", &choice, &unknown, &Params::default()),
        Err(Error::Evidence(_))
    ));
    let sure = Answer {
        confidence: Some(1.2),
        ..Answer::choice([("a", 0.5), ("b", 0.5)])
    };
    assert!(matches!(decide("x", &choice, &sure, &Params::default()), Err(Error::Evidence(_))));
}

#[test]
fn a_request_is_written_as_the_api_reads_it() {
    let request = Request::new(serde_json::json!({
        "instructions": "Assess the document.",
        "input_fields": "1. `text` (str):",
        "inputs": {"text": "example"},
    }))
    .model("jev-latest")
    .question("urgent", Question::noul("Is service blocked?"))
    .question("severity", Question::score("Rate impact.", ["Minor", "Disruptive", "Blocking"]))
    .question(
        "category",
        Question::choice("Classify the issue.", [("2", String::new()), ("other", String::new())]),
    )
    .question(
        "blocked",
        Question::noul_with("Is it blocked?", Some("Service unavailable"), Some("Workaround available")),
    );
    let written = serde_json::to_string(&request).unwrap();
    assert_eq!(
        written,
        concat!(
            r#"{"model":"jev-latest","state":{"instructions":"Assess the document.","input_fields":"1. `text` (str):","inputs":{"text":"example"}},"#,
            r#""questions":{"urgent":{"instructions":"Is service blocked?","type":"noul"},"#,
            r#""severity":{"instructions":"Rate impact.","type":"score","criteria":["Minor","Disruptive","Blocking"]},"#,
            r#""category":{"instructions":"Classify the issue.","type":"choice","criteria":{"2":null,"other":null}},"#,
            r#""blocked":{"instructions":"Is it blocked?","type":"noul","criteria":{"true":"Service unavailable","false":"Workaround available"}}}}"#
        )
    );
    assert_eq!(serde_json::from_str::<Request>(&written).unwrap(), request);
}

#[test]
fn a_response_is_read_as_a_server_writes_it() {
    let hosted = r#"{"model":"jev-1.13.0","answers":{
        "urgent":{"type":"noul","noul":0.98},
        "severity":{"type":"score","score":1.7,"confidence":0.9,"legend":{"0":"Minor","1":"Disruptive","2":"Blocking"},"probabilities":{"0":0.1,"1":0.1,"2":0.8}},
        "category":{"type":"choice","choice":"technical","confidence":0.9,"probabilities":{"technical":0.8,"billing":0.2}}},
        "usage":{"input_tokens":120,"output_tokens":12}}"#;
    let response: systemone::Response = serde_json::from_str(hosted).unwrap();
    assert_eq!(response.usage.total(), 132);
    assert_eq!(response.answers["urgent"].noul, Some(0.98));
    let score = Question::score("Rate impact.", ["Minor", "Disruptive", "Blocking"]);
    let Decision::Score(severity) = decide("severity", &score, &response.answers["severity"], &Params::default()).unwrap() else {
        panic!()
    };
    assert!((severity.value - 1.7).abs() < 1e-9);
    assert_eq!(severity.level, 2);
    // What a decision is written as, for a record of a run.
    let written = serde_json::to_value(Decision::Score(severity)).unwrap();
    assert_eq!(written["level"], 2);
    assert_eq!(written["probabilities"]["2"], 0.8);
    let noul = decide("urgent", &Question::noul("?"), &response.answers["urgent"], &Params::default()).unwrap();
    let written = serde_json::to_string(&noul).unwrap();
    assert!(
        written.starts_with(r#"{"confidence":0.96"#) && written.ends_with(r#""value":true,"probability":0.98}"#),
        "{written}"
    );
}

#[test]
fn published_confidence_formulas() {
    // (p_max - 1/n) / (1 - 1/n)
    assert!(close(systemone::choice_confidence([0.8, 0.2]), 0.6));
    assert!(close(systemone::choice_confidence([0.25, 0.25, 0.25, 0.25]), 0.0));
    assert!(close(systemone::choice_confidence([1.0]), 1.0));
    // Certain of one level: 1. Every level as likely: 0.
    assert!(close(systemone::score_confidence(&[0.0, 1.0, 0.0]), 1.0));
    assert!(close(systemone::score_confidence(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.0));
}
