//! What a branch, or one nested below it, does outside is waited
//! for and kept: it is neither stopped, nor let finish unwaited, nor lost when the branch is
//! stopped after.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use systemone::{Answer, Scripted};
const DEADLINE: Duration = Duration::from_secs(5);
fn model() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions.keys().map(|n| (n.clone(), Answer::noul(0.9))).collect::<Vec<_>>()
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
struct All;
impl Arbiter for All {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling::all(ledger))
    }
}
#[test]
fn an_effect_in_a_descendant_keeps_its_ancestor_waiting_for_the_receipt() {
    let (entered, action_entered) = mpsc::channel();
    let action_entered = Mutex::new(action_entered);
    let (release, action_release) = mpsc::channel();
    let action_release = Mutex::new(action_release);
    let (finished, action_finished) = mpsc::channel();
    let child = Rdm::new(Signature::new("Run one external operation").input("q").noul("effect", "Do it?"))
        .model(model())
        .arbiter(All)
        .tool(Tool::new("effect", "Stub external operation", move |call: &Call| {
            call.acting(|| {
                entered.send(()).unwrap();
                action_release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
                finished.send(()).unwrap();
                "accepted operation receipt"
            })
        }));
    let parent = Rdm::new(
        Signature::new("Answer and perform requested work")
            .input("q")
            .noul("quick", "Answer?")
            .noul("slow", "Do requested operation?"),
    )
    .model(model())
    .arbiter(Quick)
    .tool(Tool::new("quick", "Quick answer", move |_| {
        action_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("answer in hand")
    }))
    .tool(Tool::new("slow", "Nested external operation", move |call: &Call| {
        let result = call.on(&child).sub(inputs! {"q" => "operation"})?;
        Ok(result.observations["effect"].clone())
    }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(parent.call(inputs! {"q" => "question"}));
    });
    let early = received.recv_timeout(Duration::from_millis(300));
    // Always release the stub; this test performs no real external action.
    release.send(()).unwrap();
    let returned_before_receipt = early.is_ok();
    let result = early.or_else(|_| received.recv_timeout(DEADLINE)).unwrap().unwrap();
    action_finished.recv_timeout(DEADLINE).unwrap();
    assert!(
        !returned_before_receipt,
        "The parent returned while a descendant's accepted operation was still in flight: {:?}",
        result.arbitration
    );
    assert_eq!(
        result.observations["slow"], "accepted operation receipt",
        "The enclosing request lost the receipt"
    );
}

#[test]
fn a_finished_effect_receipt_survives_a_stop_before_its_branch_returns() {
    struct ReceiptPending {
        entered: Mutex<mpsc::Sender<()>>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl Arbiter for ReceiptPending {
        fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            assert_eq!(
                ledger.branch("effect").unwrap().result.as_ref(),
                Some(&serde_json::json!("accepted operation receipt"))
            );
            self.entered.lock().unwrap().send(()).unwrap();
            self.release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
            Ok(Ruling::all(ledger))
        }
    }
    let (entered, receipt_available) = mpsc::channel();
    let receipt_available = Mutex::new(receipt_available);
    let (release, finish_child) = mpsc::channel();
    let (ended, outer_ended) = mpsc::channel();
    let child = Rdm::new(
        Signature::new("Do requested operation and settle receipt")
            .input("q")
            .noul("effect", "Do it?"),
    )
    .model(model())
    .arbiter(ReceiptPending {
        entered: Mutex::new(entered),
        release: Mutex::new(finish_child),
    })
    .tool(Tool::new("effect", "Stub accepted operation", |call: &Call| {
        call.acting(|| "accepted operation receipt")
    }));
    let parent = Rdm::new(
        Signature::new("Answer and perform requested work")
            .input("q")
            .noul("quick", "Answer?")
            .noul("slow", "Perform requested operation?"),
    )
    .model(model())
    .arbiter(Quick)
    .tool(Tool::new("quick", "Answer only after operation returned its receipt", move |_| {
        receipt_available.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("answer in hand")
    }))
    .tool(Tool::new("slow", "Operation followed by receipt processing", move |call: &Call| {
        let result = call.on(&child).sub(inputs! {"q" => "operation"});
        let _ = ended.send(());
        result.map(|p| p.observations["effect"].clone())
    }));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(parent.call(inputs! {"q" => "question"}));
    });
    let early = received.recv_timeout(Duration::from_millis(300));
    release.send(()).unwrap();
    let result = early.or_else(|_| received.recv_timeout(DEADLINE)).unwrap().unwrap();
    outer_ended.recv_timeout(DEADLINE).unwrap();
    let exported = serde_json::to_string(&result).unwrap();
    assert!(
        exported.contains("accepted operation receipt"),
        "An accepted effect's known receipt vanished while its branch processed it: {exported}"
    );
}

#[test]
fn an_in_flight_external_operation_cannot_be_detached_as_a_shadow() {
    struct Shadow;
    impl Arbiter for Shadow {
        fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            Ok(Ruling {
                used: vec!["quick".into()],
                shadow: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (entered, action_entered) = mpsc::channel();
    let action_entered = Mutex::new(action_entered);
    let (release, action_release) = mpsc::channel();
    let action_release = Mutex::new(action_release);
    let (finished, action_finished) = mpsc::channel();
    let parent = Rdm::new(
        Signature::new("Answer with a counterfactual verifier")
            .input("q")
            .noul("quick", "Answer?")
            .noul("slow", "Verify?"),
    )
    .model(model())
    .arbiter(Shadow)
    .tool(Tool::new("quick", "Answer after another route began an operation", move |_| {
        action_entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("answer in hand")
    }))
    .tool(Tool::new(
        "slow",
        "An already-accepted operation cannot become counterfactual",
        move |call: &Call| {
            let result = call.acting(|| {
                entered.send(()).unwrap();
                action_release.lock().unwrap().recv_timeout(DEADLINE).unwrap();
                "accepted operation receipt"
            });
            finished.send(()).unwrap();
            result
        },
    ));
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sent.send(parent.call(inputs! {"q" => "question"}));
    });
    let early = received.recv_timeout(Duration::from_millis(300));
    release.send(()).unwrap();
    let returned_before_receipt = early.is_ok();
    let result = early.or_else(|_| received.recv_timeout(DEADLINE)).unwrap().unwrap();
    action_finished.recv_timeout(DEADLINE).unwrap();
    assert!(
        !returned_before_receipt,
        "Shadowing detached an accepted in-flight operation before it had an outcome: {:?}",
        result.arbitration
    );
    assert!(serde_json::to_string(&result).unwrap().contains("accepted operation receipt"));
}

#[test]
fn a_legal_domain_answer_named_receipts_does_not_hide_operation_receipts() {
    let result = Rdm::new(
        Signature::new("Keep domain answers and execution receipts")
            .input("q")
            .noul("receipts", "Existing domain question?")
            .noul("_receipts", "Another existing domain question?")
            .noul("quick", "Run requested operation?"),
    )
    .model(model())
    .arbiter(Quick)
    .tool(Tool::new("quick", "Stub operation", |call: &Call| {
        call.acting(|| "accepted operation receipt")
    }))
    .call(inputs! {"q" => "question"})
    .unwrap();
    let exported = serde_json::to_value(&result).unwrap();
    assert_eq!(exported["receipts"]["value"], true);
    assert_eq!(exported["_receipts"]["value"], true);
    assert_eq!(exported["__receipts"][0]["receipt"], "accepted operation receipt");
}
