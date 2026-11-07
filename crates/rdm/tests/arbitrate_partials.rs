//! What a branch says it has so far is recorded, versioned and settled as a result is.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
use rdm::arbitrate::{Arbiter, Ledger, Ruling};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use systemone::{Answer, Scripted};

const DEADLINE: Duration = Duration::from_secs(5);
const WORKER_DEADLINE: Duration = Duration::from_secs(20);
fn model() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions.keys().map(|n| (n.clone(), Answer::noul(0.9))).collect::<Vec<_>>()
    }))
}
struct Partial;
impl Arbiter for Partial {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        assert_eq!(ledger.branch("slow").unwrap().partial, Some(json!("early answer")));
        Ok(Ruling {
            used: vec!["slow".into()],
            done: true,
            ..Default::default()
        })
    }
}

#[test]
fn a_sufficient_partial_returns_before_the_tool_and_keeps_that_snapshot() {
    let (release, worker_release) = mpsc::channel();
    let worker_release = Mutex::new(worker_release);
    let (ended, worker_ended) = mpsc::channel();
    let rdm = Rdm::new(Signature::new("Answer the question").input("q").noul("slow", "Work?"))
        .model(model())
        .arbiter(Partial)
        .tool(Tool::new("slow", "Answer then continue", move |call: &Call| {
            call.partial("early answer");
            worker_release.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
            ended.send(()).unwrap();
            Ok("different final output")
        }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(rdm.call(inputs! { "q" => "question" }));
    });
    let result = received.recv_timeout(DEADLINE);
    // Release even on failure, so the review never leaves a blocked worker behind.
    let _ = release.send(());
    let result = result
        .expect("The caller waited for final output instead of the sufficient partial")
        .unwrap();
    assert_eq!(result.observations["slow"], "early answer");
    assert_eq!(result.arbitration.as_ref().unwrap().partial, ["slow"]);
    assert_eq!(result.arbitration.as_ref().unwrap().versions, [("slow".to_owned(), 1)]);
    worker_ended.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        result.observations["slow"], "early answer",
        "A later final return changed the accepted partial"
    );
}

#[test]
fn a_revised_partial_invalidates_an_in_flight_ruling_about_the_same_branch() {
    struct Revise {
        calls: Arc<AtomicUsize>,
        revise: Mutex<mpsc::Sender<()>>,
    }
    impl Arbiter for Revise {
        fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                assert_eq!(ledger.branch("slow").unwrap().partial, Some(json!("wrong draft")));
                self.revise.lock().unwrap().send(()).unwrap();
                let until = Instant::now() + DEADLINE;
                while !call
                    .siblings()
                    .iter()
                    .any(|b| b.output == "slow" && b.partial == Some(json!("corrected answer")))
                {
                    assert!(Instant::now() < until, "The correction never reached the live ledger");
                    std::thread::yield_now();
                }
                // This response was computed from the old wrong-draft snapshot.
            } else {
                assert_eq!(ledger.branch("slow").unwrap().partial, Some(json!("corrected answer")));
            }
            Ok(Ruling {
                used: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (revise, worker_revise) = mpsc::channel();
    let worker_revise = Mutex::new(worker_revise);
    let (release, worker_release) = mpsc::channel();
    let worker_release = Mutex::new(worker_release);
    let calls = Arc::new(AtomicUsize::new(0));
    let rdm = Rdm::new(Signature::new("Use supported evidence").input("q").noul("slow", "Work?"))
        .model(model())
        .arbiter(Revise {
            calls: calls.clone(),
            revise: Mutex::new(revise),
        })
        .tool(Tool::new("slow", "Revises its evidence", move |call: &Call| {
            call.partial("wrong draft");
            worker_revise.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
            call.partial("corrected answer");
            worker_release.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
            Ok("final answer")
        }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(rdm.call(inputs! { "q" => "question" }));
    });
    let result = received.recv_timeout(DEADLINE);
    let _ = release.send(());
    let result = result.expect("The caller waited for final output").unwrap();
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "A stale ruling committed the old partial despite a newer correction"
    );
    assert_eq!(result.observations["slow"], "corrected answer");
    assert_eq!(result.arbitration.as_ref().unwrap().versions, [("slow".to_owned(), 2)]);
    assert_eq!(result.arbitration.unwrap().partial, ["slow"]);
}

#[test]
fn a_shadowed_verifier_can_continue_model_work_and_propagates_its_identity() {
    struct Shadow;
    impl Arbiter for Shadow {
        fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            Ok(Ruling {
                used: vec!["quick".into()],
                shadow: vec!["verifier".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let child = Rdm::new(Signature::new("Verify without effects").input("q").noul("read", "Read evidence?"))
        .model(model())
        .tool(Tool::new("read", "Pure test evidence", |call: &Call| {
            Ok(json!({"shadowed":call.shadowed(), "stopped":call.stopped()}))
        }));
    let (entered, verifier_entered) = mpsc::channel();
    let verifier_entered = Mutex::new(verifier_entered);
    let (release, verifier_release) = mpsc::channel();
    let verifier_release = Mutex::new(verifier_release);
    let (reported, report) = mpsc::channel();
    let rdm = Rdm::new(
        Signature::new("Answer and verify")
            .input("q")
            .noul("quick", "Answer?")
            .noul("verifier", "Verify?"),
    )
    .model(model())
    .arbiter(Shadow)
    .tool(Tool::new("quick", "Answer", move |_| {
        verifier_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("supported answer")
    }))
    .tool(Tool::new("verifier", "Pure model verification", move |call: &Call| {
        entered.send(()).unwrap();
        verifier_release.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
        let result = call.on(&child).sub(inputs! { "q" => "verify" });
        reported.send(result.map(|p| p.observations["read"].clone())).unwrap();
        Ok("verified")
    }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(rdm.call(inputs! { "q" => "question" }));
    });
    let result = received.recv_timeout(DEADLINE);
    let _ = release.send(());
    let result = result.expect("The caller waited for background verification").unwrap();
    assert_eq!(result.arbitration.unwrap().shadowed, ["verifier"]);
    assert_eq!(
        report.recv_timeout(DEADLINE).unwrap().unwrap(),
        json!({"shadowed":true,"stopped":false})
    );
}

#[test]
fn a_final_return_during_disposition_does_not_replace_the_accepted_partial() {
    struct Settle;
    impl Arbiter for Settle {
        fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            assert_eq!(ledger.branch("slow").unwrap().partial, Some(json!("accepted snapshot")));
            Ok(Ruling {
                used: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (entered, first_entered) = mpsc::channel();
    let first_entered = Mutex::new(first_entered);
    let (finish, finish_worker) = mpsc::channel();
    let finish_worker = Mutex::new(finish_worker);
    let observer: Arc<Mutex<Option<Call>>> = Arc::new(Mutex::new(None));
    let observed = observer.clone();
    let rdm = Rdm::new(
        Signature::new("Preserve accepted evidence")
            .input("q")
            .noul("withdraw", "Unneeded work?")
            .noul("slow", "Answer?"),
    )
    .model(model())
    .arbiter(Settle)
    .record(move |event: &rdm::record::Event| {
        match event {
            rdm::record::Event::Stopped { output, .. } if output == "withdraw" => {
                finish.send(()).unwrap();
                let until = Instant::now() + DEADLINE;
                loop {
                    let visible = observer.lock().unwrap().as_ref().unwrap().siblings();
                    if visible
                        .iter()
                        .any(|b| b.output == "slow" && b.state == rdm::arbitrate::State::Returned)
                    {
                        break;
                    }
                    assert!(Instant::now() < until, "Final result never reached the live ledger");
                    std::thread::yield_now();
                }
            }
            _ => {}
        }
        Ok(())
    })
    .tool(Tool::new("withdraw", "Work to withdraw first", move |call: &Call| {
        *observed.lock().unwrap() = Some(call.clone());
        entered.send(()).unwrap();
        let until = Instant::now() + WORKER_DEADLINE;
        while !call.stopped() {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        Err::<String, _>(rdm::Error::Stopped("stopped".into()))
    }))
    .tool(Tool::new("slow", "Answer with a partial", move |call: &Call| {
        first_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        call.partial("accepted snapshot");
        finish_worker.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
        Ok("unreviewed final correction")
    }));
    let result = rdm.call(inputs! { "q" => "question" }).unwrap();
    assert_eq!(result.arbitration.as_ref().unwrap().partial, ["slow"]);
    assert_eq!(
        result.arbitration.as_ref().unwrap().versions,
        [("slow".to_owned(), 1)],
        "A final return during disposition replaced the identity of the reviewed partial"
    );
    assert_eq!(
        result.observations["slow"], "accepted snapshot",
        "The final output replaced evidence the ruling actually reviewed"
    );
}

#[test]
fn a_same_value_partial_revision_is_recorded_and_invalidates_an_in_flight_ruling() {
    struct Revise {
        calls: Arc<AtomicUsize>,
        revise: Mutex<mpsc::Sender<()>>,
    }
    impl Arbiter for Revise {
        fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                assert_eq!(ledger.branch("slow").unwrap().version, 1);
                self.revise.lock().unwrap().send(()).unwrap();
                let until = Instant::now() + DEADLINE;
                while !call.siblings().iter().any(|b| b.output == "slow" && b.version == 2) {
                    assert!(Instant::now() < until, "The same-value revision was not published");
                    std::thread::yield_now();
                }
            } else {
                assert_eq!(ledger.branch("slow").unwrap().version, 2);
            }
            Ok(Ruling {
                used: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (revise, worker_revise) = mpsc::channel();
    let worker_revise = Mutex::new(worker_revise);
    let (release, worker_release) = mpsc::channel();
    let worker_release = Mutex::new(worker_release);
    let calls = Arc::new(AtomicUsize::new(0));
    let versions = Arc::new(Mutex::new(Vec::new()));
    let recorded = versions.clone();
    let rdm = Rdm::new(Signature::new("Use current evidence").input("q").noul("slow", "Work?"))
        .model(model())
        .arbiter(Revise {
            calls: calls.clone(),
            revise: Mutex::new(revise),
        })
        .record(move |event: &rdm::record::Event| {
            if let rdm::record::Event::Partial {
                output, version, partial, ..
            } = event
            {
                assert_eq!(output, "slow");
                assert_eq!(*partial, json!("same answer"));
                recorded.lock().unwrap().push(*version);
            }
            Ok(())
        })
        .tool(Tool::new("slow", "Republish evidence", move |call: &Call| {
            call.partial("same answer");
            worker_revise.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
            call.partial("same answer");
            worker_release.lock().unwrap().recv_timeout(WORKER_DEADLINE).unwrap();
            Ok("final answer")
        }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(rdm.call(inputs! {"q" => "question"}));
    });
    let result = received.recv_timeout(DEADLINE);
    let _ = release.send(());
    let result = result.unwrap().unwrap();
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "A ruling committed before seeing the new revision"
    );
    assert_eq!(*versions.lock().unwrap(), [1, 2]);
    assert_eq!(result.observations["slow"], "same answer");
    assert_eq!(
        result.arbitration.as_ref().unwrap().versions,
        [("slow".to_owned(), 2)],
        "Word equality erased the reviewed revision's identity"
    );
    assert_eq!(serde_json::to_value(&result.arbitration).unwrap()["versions"], json!([["slow", 2]]));
}
