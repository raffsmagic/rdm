//! A ruling keeps the whole request it settles, and its prediction says what it came to.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::{Arbitrate, Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use systemone::{Answer, Scripted};

const DEADLINE: Duration = Duration::from_secs(3);

fn yes() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions
            .iter()
            .map(|(name, question)| {
                let answer = if name == "next" {
                    let labels = question.labels();
                    Answer::chose("finish", labels.iter().map(String::as_str))
                } else {
                    Answer::noul(if name == "conflict" { 0.1 } else { 0.9 })
                };
                (name.clone(), answer)
            })
            .collect::<Vec<_>>()
    }))
}

struct Quick;
impl Arbiter for Quick {
    fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling {
            used: vec!["quick".into()],
            done: true,
            ..Default::default()
        })
    }
}

#[test]
fn first_arrival_without_preflight_asks_use_and_done() {
    let result = Rdm::new(Signature::new("Answer").input("q").noul("quick", "Answer?"))
        .model(yes())
        .arbiter(Arbitrate::new())
        .tool(Tool::new("quick", "Answer", |_| Ok("supported answer")))
        .call(inputs! { "q" => "question" })
        .unwrap();
    assert!(
        result.arbitration.as_ref().unwrap().answered,
        "The yes-to-all model was never asked whether the returned answer finishes the request: {:?}",
        result.arbitration
    );
    assert_eq!(result.trajectory.len(), 2);
}

#[test]
fn queued_branch_does_not_start_before_the_result_is_arbitrated() {
    struct Observe {
        started: Mutex<mpsc::Receiver<()>>,
        observed: Arc<AtomicBool>,
    }
    impl Arbiter for Observe {
        fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            assert_eq!(ledger.branch("quick").unwrap().state, State::Returned);
            self.observed
                .store(self.started.lock().unwrap().recv_timeout(DEADLINE).is_ok(), Ordering::SeqCst);
            Ok(Ruling {
                used: vec!["quick".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (started, heard) = mpsc::channel();
    let observed = Arc::new(AtomicBool::new(false));
    let result = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Answer?")
            .noul("slow", "Start another operation?"),
    )
    .threads(1)
    .model(yes())
    .arbiter(Observe {
        started: Mutex::new(heard),
        observed: observed.clone(),
    })
    .tool(Tool::new("quick", "Answer", |_| Ok("done")))
    .tool(Tool::new("slow", "Unneeded operation", move |_| {
        started.send(()).unwrap();
        Ok("started")
    }))
    .call(inputs! { "q" => "question" })
    .unwrap();
    assert_eq!(result.arbitration.unwrap().used, ["quick"]);
    assert!(
        !observed.load(Ordering::SeqCst),
        "The queued operation started before arbitration could withdraw it"
    );
}

#[test]
fn cancelling_a_branch_while_its_child_model_runs_prevents_new_tools() {
    let (entered, model_entered) = mpsc::channel();
    let (release, model_release) = mpsc::channel();
    let model_entered = Mutex::new(model_entered);
    let model_release = Mutex::new(model_release);
    let dispatched = Arc::new(AtomicBool::new(false));
    let effect = dispatched.clone();
    let (ended, child_ended) = mpsc::channel();
    let child = Rdm::new(Signature::new("Choose effect").input("q").noul("effect", "Perform it?"))
        .model(Scripted::new(move |_| {
            entered.send(()).unwrap();
            model_release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
            [("effect", Answer::noul(0.9))]
        }))
        .tool(Tool::new("effect", "Record dispatch", move |_| {
            effect.store(true, Ordering::SeqCst);
            Ok("performed")
        }));
    let root = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Answer?")
            .noul("slow", "Investigate?"),
    )
    .model(yes())
    .arbiter(Quick)
    .tool(Tool::new("quick", "Answer", move |_| {
        model_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("slow", "Investigate", move |call| {
        let result = call.on(&child).sub(inputs! { "q" => "child question" });
        ended.send(result.is_err_and(|e| e.is_stopped())).unwrap();
        Ok("ended")
    }));
    let result = root.call(inputs! { "q" => "question" }).unwrap();
    assert_eq!(result.arbitration.unwrap().stopped, ["slow"]);
    // The parent has already stopped this branch. Only now may the in-flight model return.
    release.send(()).unwrap();
    let stopped = child_ended.recv_timeout(DEADLINE).unwrap();
    assert!(
        !dispatched.load(Ordering::SeqCst),
        "A new tool dispatched after its ancestor was stopped"
    );
    assert!(stopped, "The stopped child incorrectly returned success");
}

fn arbitration_request_for(task: &str, contract: &str) -> systemone::Request {
    let model = yes();
    let result = Rdm::new(
        Signature::new(task)
            .input_with("values", "list[int]", "Values to compare")
            .noul("scan", contract),
    )
    .model(model.clone())
    .arbiter(Arbitrate::new())
    .tool(Tool::new("scan", "Read values", |_| Ok(1)))
    .call(inputs! { "values" => vec![1,9] })
    .unwrap();
    assert!(result.arbitration.unwrap().answered);
    let request = model.requests().last().cloned().unwrap();
    assert!(request.questions.contains_key("done") || request.questions.contains_key("next"));
    request
}

#[test]
fn arbitration_retains_the_original_task_and_constraints() {
    assert_ne!(
        arbitration_request_for("Find the MINIMUM value.", "Read the values?"),
        arbitration_request_for("Find the MAXIMUM value.", "Read the values?"),
        "Opposite tasks produce exactly the same arbitration request; the original task is missing"
    );
}

#[test]
fn arbitration_retains_the_original_output_contract() {
    assert_ne!(
        arbitration_request_for("Choose an appropriate result.", "Return only the minimum value."),
        arbitration_request_for("Choose an appropriate result.", "Return only the maximum value."),
        "Opposite output contracts produce the same arbitration request; the selected work meaning is missing"
    );
}

#[test]
fn arbitration_retains_the_original_answer_criteria() {
    let request = |meaning: &str| {
        let model = yes();
        let result = Rdm::new(
            Signature::new("Resolve the requested comparison.")
                .input("values")
                .with_output("scan", rdm::Output::noul_with("Use the computed value?", Some(meaning), None)),
        )
        .model(model.clone())
        .arbiter(Arbitrate::new())
        .tool(Tool::new("scan", "Read values", |_| Ok(1)))
        .call(inputs! { "values" => vec![1,9] })
        .unwrap();
        assert!(result.arbitration.unwrap().answered);
        model.requests().last().cloned().unwrap()
    };
    assert_ne!(
        request("It is the MINIMUM value."),
        request("It is the MAXIMUM value."),
        "The meanings of parent answers were omitted from the arbiter context"
    );
}

#[test]
fn an_existing_input_named_request_can_be_arbitrated() {
    let result = Rdm::new(Signature::new("Answer the request.").input("request").noul("quick", "Answer?"))
        .model(yes())
        .arbiter(Arbitrate::new())
        .tool(Tool::new("quick", "Answer", |_| Ok("answer")))
        .call(inputs! { "request" => "A normal existing request input" });
    assert!(result.is_ok(), "Enabling arbitration broke a legal parent signature: {result:?}");
    assert!(result.unwrap().arbitration.unwrap().answered);
}

#[test]
fn siblings_see_completed_results_while_arbitration_is_in_flight() {
    struct Inspect {
        release: Mutex<mpsc::Sender<()>>,
        inspected: Mutex<mpsc::Receiver<bool>>,
        visible: Arc<AtomicBool>,
        tested: AtomicBool,
    }
    impl Arbiter for Inspect {
        fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            if !self.tested.swap(true, Ordering::SeqCst) {
                self.release.lock().unwrap().send(()).unwrap();
                let visible = self.inspected.lock().unwrap().recv_timeout(DEADLINE * 2).unwrap();
                self.visible.store(visible, Ordering::SeqCst);
            }
            Ok(Ruling {
                used: vec!["quick".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (entered, consumer_entered) = mpsc::channel();
    let consumer_entered = Mutex::new(consumer_entered);
    let (release, producer_release) = mpsc::channel();
    let producer_release = Mutex::new(producer_release);
    let (returned, producer_returned) = mpsc::channel();
    let producer_returned = Mutex::new(producer_returned);
    let (inspected, observed) = mpsc::channel();
    let visible = Arc::new(AtomicBool::new(false));
    let result = Rdm::new(
        Signature::new("Find evidence")
            .input("q")
            .noul("quick", "Start?")
            .noul("producer", "Start?")
            .noul("consumer", "Start?"),
    )
    .model(yes())
    .arbiter(Inspect {
        release: Mutex::new(release),
        inspected: Mutex::new(observed),
        visible: visible.clone(),
        tested: AtomicBool::new(false),
    })
    .record(move |event: &rdm::record::Event| {
        if matches!(event, rdm::record::Event::Returned { tool, .. } if tool == "producer") {
            returned.send(()).unwrap();
        }
        Ok(())
    })
    .tool(Tool::new("quick", "Initial evidence", move |_| {
        consumer_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("initial")
    }))
    .tool(Tool::new("producer", "More evidence", move |_| {
        producer_release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("new evidence")
    }))
    .tool(Tool::new("consumer", "Uses sibling evidence", move |call: &Call| {
        entered.send(()).unwrap();
        producer_returned.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        let deadline = std::time::Instant::now() + DEADLINE;
        let visible = loop {
            if call
                .siblings()
                .iter()
                .any(|b| b.output == "producer" && b.state == State::Returned && b.result.is_some())
            {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            std::thread::yield_now();
        };
        inspected.send(visible).unwrap();
        Ok("inspected")
    }))
    .call(inputs! { "q" => "question" })
    .unwrap();
    assert_eq!(result.arbitration.unwrap().used, ["quick"]);
    assert!(
        visible.load(Ordering::SeqCst),
        "Producer had returned, but its sibling could not see it until arbitration finished"
    );
}

#[test]
fn serialized_prediction_preserves_which_results_were_accepted() {
    let result = Rdm::new(Signature::new("Answer").input("q").noul("quick", "Answer?"))
        .model(yes())
        .arbiter(Quick)
        .tool(Tool::new("quick", "Answer", |_| Ok("supported answer")))
        .call(inputs! { "q" => "question" })
        .unwrap();
    assert_eq!(result.arbitration.as_ref().unwrap().used, ["quick"]);
    let serialized = serde_json::to_value(&result).unwrap();
    assert_eq!(
        serialized["arbitration"]["used"],
        serde_json::json!(["quick"]),
        "JSON consumers lose the arbitration decision while raw observations survive"
    );
}

#[test]
fn a_ruling_is_reconsidered_if_new_evidence_arrives_while_it_is_being_decided() {
    struct Reconsider {
        count: Arc<std::sync::atomic::AtomicUsize>,
        release: Mutex<mpsc::Sender<()>>,
    }
    impl Arbiter for Reconsider {
        fn arbitrate(&self, _: &Ledger, call: &Call) -> rdm::Result<Ruling> {
            if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
                self.release.lock().unwrap().send(()).unwrap();
                let deadline = std::time::Instant::now() + DEADLINE;
                while !call.siblings().iter().any(|b| b.output == "producer" && b.state == State::Returned) {
                    assert!(std::time::Instant::now() < deadline, "New evidence never reached the live ledger");
                    std::thread::yield_now();
                }
                // This answer is based on the OLD snapshot, before the producer returned.
                return Ok(Ruling {
                    used: vec!["quick".into()],
                    done: true,
                    ..Default::default()
                });
            }
            Ok(Ruling {
                used: vec!["producer".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (entered, producer_entered) = mpsc::channel();
    let producer_entered = Mutex::new(producer_entered);
    let (release, producer_release) = mpsc::channel();
    let producer_release = Mutex::new(producer_release);
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let result = Rdm::new(
        Signature::new("Resolve conflicting evidence")
            .input("q")
            .noul("quick", "Investigate?")
            .noul("producer", "Investigate?"),
    )
    .model(yes())
    .arbiter(Reconsider {
        count: count.clone(),
        release: Mutex::new(release),
    })
    .tool(Tool::new("quick", "Initial hypothesis", move |_| {
        producer_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("wrong first hypothesis")
    }))
    .tool(Tool::new("producer", "Correcting evidence", move |_| {
        entered.send(()).unwrap();
        producer_release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("corrected hypothesis")
    }))
    .call(inputs! { "q" => "question" })
    .unwrap();
    assert!(
        count.load(Ordering::SeqCst) >= 2,
        "A stale done ruling bypassed newly arrived evidence"
    );
    assert_eq!(result.arbitration.unwrap().used, ["producer"]);
    assert!(result.observations.contains_key("producer"));
}

#[test]
fn arbitration_retains_the_selected_arguments_of_the_parent_request() {
    let request = |chosen: &'static str| {
        let model = Arc::new(Scripted::new(move |r| {
            r.questions
                .iter()
                .map(|(name, question)| {
                    let labels = question.labels();
                    let answer = match name.as_str() {
                        "account" => Answer::chose(chosen, labels.iter().map(String::as_str)),
                        "next" => Answer::chose("finish", labels.iter().map(String::as_str)),
                        _ => Answer::noul(0.9),
                    };
                    (name.clone(), answer)
                })
                .collect::<Vec<_>>()
        }));
        Rdm::new(
            Signature::new("Read the requested account balance.")
                .input("q")
                .choice(
                    "account",
                    "Which account is being read?",
                    [("personal", "Personal account"), ("business", "Business account")],
                )
                .noul("read", "Read the balance?"),
        )
        .model(model.clone())
        .arbiter(Arbitrate::new())
        .tool(Tool::new("read", "Read selected account", |call: &Call| {
            let _: String = call.arg("account")?;
            // A generic receipt alone does not identify which account the read used.
            Ok("Balance unavailable")
        }))
        .call(inputs! { "q" => "What is the balance?" })
        .unwrap();
        model.requests().last().cloned().unwrap()
    };
    assert_ne!(
        request("personal"),
        request("business"),
        "Different actual account arguments produced identical arbitration context"
    );
}

#[test]
fn a_legal_answer_named_arbitration_does_not_hide_settlement_metadata() {
    fn has_settlement(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(map) => map.get("used") == Some(&serde_json::json!(["quick"])) || map.values().any(has_settlement),
            serde_json::Value::Array(values) => values.iter().any(has_settlement),
            _ => false,
        }
    }
    let result = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("arbitration", "Existing domain question?")
            .noul("quick", "Answer?"),
    )
    .model(yes())
    .arbiter(Quick)
    .tool(Tool::new("quick", "Answer", |_| Ok("supported answer")))
    .call(inputs! { "q" => "question" })
    .unwrap();
    assert_eq!(result.arbitration.as_ref().unwrap().used, ["quick"]);
    let exported = serde_json::to_value(result).unwrap();
    assert!(
        has_settlement(&exported),
        "Export silently discarded the used-result metadata: {exported}"
    );
}
