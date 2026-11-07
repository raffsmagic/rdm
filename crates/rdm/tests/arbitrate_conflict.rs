//! A conflict is among the results used, judged with all the request asked.

//! Current-contract probes: `conflict` describes disagreement in RAW results, not the
//! eventual accepted set. A future scoped-conflict API should adapt this scripted evidence
//! while retaining the positive resolved-conflict and negative unresolved-conflict controls.
use rdm::arbitrate::{Arbiter, Arbitrate, Branch, Ledger, Ruling, State};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use serde_json::json;
use std::sync::{Arc, Mutex};
use systemone::{Answer, Scripted};

struct Probe {
    ledger: Ledger,
    got: Arc<Mutex<Option<Ruling>>>,
}
impl Arbiter for Probe {
    fn arbitrate(&self, _: &Ledger, call: &Call) -> rdm::Result<Ruling> {
        *self.got.lock().unwrap() = Some(Arbitrate::new().arbitrate(&self.ledger, call)?);
        Ok(Ruling {
            used: vec!["probe_tick".into()],
            done: true,
            ..Default::default()
        })
    }
}
fn rule(second: bool, use_wrong: bool) -> Ruling {
    let task = if second {
        "Report the supported authoritative balance and deadline. Discard the contradicted guess."
    } else {
        "Report the supported authoritative balance. Discard the contradicted guess."
    };
    let mut signature = Signature::new(task)
        .input("question")
        .noul("supported", "Read authoritative balance?");
    if second {
        signature = signature.noul("second", "Read authoritative deadline?");
    }
    signature = signature.noul("wrong", "An unsupported guess?");
    let mut results = vec![("supported", "The authoritative record says the balance is 12.")];
    if second {
        results.push(("second", "The authoritative record says the deadline is Friday."));
    }
    results.push(("wrong", "An unsupported guess says the balance is 99."));
    let branches: Vec<Branch> = results
        .iter()
        .map(|(name, text)| Branch {
            output: (*name).into(),
            tool: (*name).into(),
            about: (*name).into(),
            selected: true,
            state: State::Returned,
            result: Some(json!(text)),
            partial: None,
            version: 1,
        })
        .collect();
    let ledger = Ledger {
        signature,
        shown: [("question".into(), json!(task))].into_iter().collect(),
        inputs: inputs! {"question" => task},
        answers: results.iter().map(|(name, _)| ((*name).into(), json!(true))).collect(),
        branches,
        round: 0,
        before: vec![],
        first: false,
        request: (0, 0),
    };
    let got = Arc::new(Mutex::new(None));
    let saved = got.clone();
    let model = Scripted::new(move |r| {
        r.questions
            .iter()
            .map(|(name, q)| {
                let answer = if name == "next" {
                    Answer::chose("finish", q.labels().iter().map(String::as_str))
                } else {
                    let yes = matches!(name.as_str(), "probe_tick" | "use_supported" | "use_second" | "conflict")
                        || (matches!(name.as_str(), "use_wrong" | "contradict") && use_wrong);
                    Answer::noul(if yes { 0.99 } else { 0.01 })
                };
                (name.clone(), answer)
            })
            .collect::<Vec<_>>()
    });
    Rdm::new(Signature::new("Run sealed policy probe").noul("probe_tick", "Produce a tick?"))
        .model(model)
        .tool(Tool::new("probe_tick", "Pure tick", |_| Ok(true)))
        .arbiter(Probe { ledger, got })
        .call(inputs! {})
        .unwrap();
    saved.lock().unwrap().clone().unwrap()
}
#[test]
fn a_supported_result_can_supersede_a_rejected_contradicted_guess() {
    let r = rule(false, false);
    assert_eq!(r.used, ["supported"]);
    assert!(r.done && !r.unresolved, "Rejected raw evidence vetoed supported completion: {r:?}");
}
#[test]
fn two_supported_obligations_are_not_vetoed_by_a_third_rejected_guess() {
    let r = rule(true, false);
    assert_eq!(r.used, ["supported", "second"]);
    assert!(r.done && !r.unresolved, "A rejected guess vetoed a consistent accepted set: {r:?}");
}
#[test]
fn incompatible_accepted_results_do_not_claim_completion() {
    let r = rule(false, true);
    assert_eq!(r.used, ["supported", "wrong"]);
    assert!(!r.done, "Incompatible selected claims were marked complete: {r:?}");
}

#[test]
fn the_selected_result_check_retains_parent_arguments_needed_to_interpret_claims() {
    let request = |account: &'static str| {
        let task = "Report the balance of the account chosen by the account answer.";
        let signature = Signature::new(task)
            .input("question")
            .choice(
                "account",
                "Which account was chosen?",
                [("personal", "Personal account"), ("business", "Business account")],
            )
            .noul("supported", "Read a balance claim?")
            .noul("second", "Read the account ledger?");
        let branches = [
            ("supported", "The requested account has a balance of 12."),
            (
                "second",
                "The personal account has balance 12; the business account has balance 99.",
            ),
        ]
        .into_iter()
        .map(|(name, text)| Branch {
            output: name.into(),
            tool: name.into(),
            about: name.into(),
            selected: true,
            state: State::Returned,
            result: Some(json!(text)),
            partial: None,
            version: 1,
        })
        .collect();
        let ledger = Ledger {
            signature,
            shown: [("question".into(), json!("What is its balance?"))].into_iter().collect(),
            inputs: inputs! {"question" => "What is its balance?"},
            answers: [
                ("account".into(), json!(account)),
                ("supported".into(), json!(true)),
                ("second".into(), json!(true)),
            ]
            .into_iter()
            .collect(),
            branches,
            round: 0,
            before: vec![],
            first: false,
            request: (0, 0),
        };
        let model = Arc::new(Scripted::new(move |r| {
            r.questions
                .iter()
                .map(|(name, q)| {
                    let a = if name == "next" {
                        Answer::chose("finish", q.labels().iter().map(String::as_str))
                    } else {
                        Answer::noul(if name == "contradict" {
                            if account == "business" { 0.99 } else { 0.01 }
                        } else {
                            0.99
                        })
                    };
                    (name.clone(), a)
                })
                .collect::<Vec<_>>()
        }));
        Rdm::new(Signature::new("Run sealed probe").noul("probe_tick", "Tick?"))
            .model(model.clone())
            .tool(Tool::new("probe_tick", "Pure tick", |_| Ok(true)))
            .arbiter(Probe {
                ledger,
                got: Arc::new(Mutex::new(None)),
            })
            .call(inputs! {})
            .unwrap();
        model
            .requests()
            .into_iter()
            .find(|r| r.questions.contains_key("contradict"))
            .expect("The selected-result coherence question was not asked")
    };
    // The first claim agrees with the ledger only for the personal account. The selected
    // account is therefore relevant evidence, despite not being a raw input field.
    assert_ne!(
        request("personal"),
        request("business"),
        "The coherence child lost the chosen account and cannot interpret 'requested account'"
    );
}

fn selected_check_for(signature: Signature) -> systemone::Request {
    let ledger = Ledger {
        signature: signature
            .input("question")
            .noul("supported", "Use first evidence?")
            .noul("second", "Use second evidence?"),
        shown: [("question".into(), json!("Compare the selected evidence."))].into_iter().collect(),
        inputs: inputs! {"question" => "Compare the selected evidence."},
        answers: [
            ("meaning".into(), json!("chosen")),
            ("supported".into(), json!(true)),
            ("second".into(), json!(true)),
        ]
        .into_iter()
        .collect(),
        branches: ["supported", "second"]
            .into_iter()
            .map(|name| Branch {
                output: name.into(),
                tool: name.into(),
                about: "Evidence".into(),
                selected: true,
                state: State::Returned,
                result: Some(json!("12")),
                partial: None,
                version: 1,
            })
            .collect(),
        round: 0,
        before: vec![],
        first: false,
        request: (0, 0),
    };
    let model = Arc::new(Scripted::new(|r| {
        r.questions
            .iter()
            .map(|(name, q)| {
                let answer = if name == "next" {
                    Answer::chose("finish", q.labels().iter().map(String::as_str))
                } else {
                    Answer::noul(if name == "contradict" { 0.01 } else { 0.99 })
                };
                (name.clone(), answer)
            })
            .collect::<Vec<_>>()
    }));
    Rdm::new(Signature::new("Run sealed probe").noul("probe_tick", "Tick?"))
        .model(model.clone())
        .tool(Tool::new("probe_tick", "Pure tick", |_| Ok(true)))
        .arbiter(Probe {
            ledger,
            got: Arc::new(Mutex::new(None)),
        })
        .call(inputs! {})
        .unwrap();
    model
        .requests()
        .into_iter()
        .find(|r| r.questions.contains_key("contradict"))
        .expect("No selected-result coherence request was recorded")
}

#[test]
fn the_selected_result_check_retains_the_original_output_contract() {
    let request = |contract| selected_check_for(Signature::new("Compare the selected values.").noul("meaning", contract));
    assert_ne!(
        request("The value must be a minimum."),
        request("The value must be a maximum."),
        "Opposite parent output contracts produced the same coherence child request"
    );
}

#[test]
fn the_selected_result_check_retains_the_original_answer_criteria() {
    let request = |criteria| {
        selected_check_for(Signature::new("Compare the selected values.").with_output(
            "meaning",
            rdm::Output::noul_with("Does the value meet the criterion?", Some(criteria), None),
        ))
    };
    assert_ne!(
        request("It is the minimum value."),
        request("It is the maximum value."),
        "Opposite parent answer criteria produced the same coherence child request"
    );
}

#[test]
fn the_selected_result_check_retains_the_original_choice_meanings() {
    let request = |meaning| {
        selected_check_for(Signature::new("Interpret the chosen operation.").choice(
            "meaning",
            "What did chosen select?",
            [("chosen", meaning), ("other", "Do something else")],
        ))
    };
    assert_ne!(
        request("Compare the minimums."),
        request("Compare the maximums."),
        "Opposite meanings of the same selected label produced the same coherence child request"
    );
}
