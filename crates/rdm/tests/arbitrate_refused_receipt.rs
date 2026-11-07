//! A record that refused an event ends the run: nothing more is
//! begun, outside or in.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
use rdm::arbitrate::{Ledger, Ruling};
use rdm::record::Event;
use rdm::{Arbiter, Call, Rdm, Signature, Tool, inputs};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use systemone::{Answer, Scripted};
struct Accept;
impl Arbiter for Accept {
    fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling {
            used: vec!["work".into()],
            done: true,
            ..Default::default()
        })
    }
}
fn verify(arbitrated: bool) {
    let effects = Arc::new(AtomicUsize::new(0));
    let counted = effects.clone();
    let attempted_receipts = Arc::new(Mutex::new(Vec::<Value>::new()));
    let kept = attempted_receipts.clone();
    let observations = Arc::new(Mutex::new(None));
    let observed = observations.clone();
    let root = Rdm::new(Signature::new("Do requested work").input("q").noul("work", "Work?"))
        .model(Scripted::new(|_| [("work", Answer::noul(0.9))]))
        .tool(Tool::new("work", "Two bounded marked actions", move |c: &Call| {
            let first = c.acting(|| {
                counted.fetch_add(1, Ordering::SeqCst);
                json!({"id":"first-known-receipt","done":true})
            });
            let failed_barrier = c.go_on().is_err();
            let second = c.acting(|| {
                counted.fetch_add(1, Ordering::SeqCst);
                json!({"id":"second-receipt","done":true})
            });
            *observed.lock().unwrap() = Some((failed_barrier, second.is_err()));
            Ok(json!({"first":first.ok(),"second":second.ok()}))
        }))
        .record(move |e: &Event| {
            if let Event::Acted { receipt, .. } = e {
                kept.lock().unwrap().push(receipt.clone());
                Err(rdm::Error::Tool("receipt recorder refused".into()))
            } else {
                Ok(())
            }
        });
    let root = if arbitrated { root.arbiter(Accept) } else { root };
    let result = root.call(inputs! {"q"=>"do bounded work"});
    assert!(result.is_err(), "recorder refusal must fail the root call");
    // The first action already happened: its known receipt must remain available to the recorder.
    assert_eq!(
        *attempted_receipts.lock().unwrap(),
        vec![json!({"id":"first-known-receipt","done":true})]
    );
    let state = observations.lock().unwrap().unwrap();
    assert_eq!(
        effects.load(Ordering::SeqCst),
        1,
        "arbitrated={arbitrated}: second effect ran after receipt refusal; go_on_failed={}, second_acting_failed={}",
        state.0,
        state.1
    );
    assert_eq!(
        state,
        (true, true),
        "known recorder failure was not enforced at both go_on and acting"
    );
}
#[test]
fn a_refused_acted_record_blocks_another_marked_action_in_an_arbitrated_branch() {
    verify(true)
}
#[test]
fn a_refused_acted_record_blocks_another_marked_action_without_an_arbiter() {
    verify(false)
}
