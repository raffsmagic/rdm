//! A request does not end while a branch nested below it is in the
//! middle of doing something outside.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::systemone::{Answer, Scripted};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::Duration;

struct All;
impl Arbiter for All {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling::all(l))
    }
}
struct Quick;
impl Arbiter for Quick {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        if l.branches.iter().any(|b| b.output == "quick" && b.state == State::Returned) {
            Ok(Ruling {
                used: vec!["quick".into()],
                done: true,
                ..Default::default()
            })
        } else {
            Ok(Ruling::all(l))
        }
    }
}

#[test]
fn parent_waits_for_a_descendant_that_has_started_an_effect() {
    let (entered, enter_rx) = channel();
    let (release, release_rx) = channel();
    let release_rx = Mutex::new(release_rx);
    let entered = Mutex::new(entered);
    let enter_rx = Mutex::new(enter_rx);
    let model = Arc::new(Scripted::new(|r| {
        r.questions
            .keys()
            .map(|name| (name.clone(), Answer::noul(0.99)))
            .collect::<Vec<_>>()
    }));
    let child = Rdm::new(
        Signature::new("Perform the selected effect.")
            .input("q")
            .noul("work", "Do the work?"),
    )
    .tool(Tool::new("work", "Performs an external effect.", move |c: &Call| {
        c.acting(|| {
            entered.lock().unwrap().send(()).unwrap();
            release_rx.lock().unwrap().recv_timeout(Duration::from_secs(5)).unwrap();
        })?;
        Ok("effect received")
    }))
    .arbiter(All)
    .model(model.clone());
    let root = Rdm::new(
        Signature::new("Resolve the request.")
            .input("q")
            .noul("quick", "Read quickly?")
            .noul("nested", "Run the child?"),
    )
    .tool(Tool::new("quick", "Answers once descendant is acting.", move |_| {
        enter_rx
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| "child did not enter")?;
        Ok("fast answer")
    }))
    .tool(Tool::new("nested", "Calls a child with its own arbitration.", move |c: &Call| {
        let p = c.on(&child).sub(inputs! {"q"=>"perform"})?;
        Ok(p.observations)
    }))
    .arbiter(Quick)
    .model(model);
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        tx.send(root.call(inputs! {"q"=>"request"})).unwrap();
    });
    let early = rx.recv_timeout(Duration::from_millis(300));
    let returned_before_receipt = early.is_ok();
    release.send(()).unwrap();
    let result = early.unwrap_or_else(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
    assert!(
        !returned_before_receipt,
        "parent returned before descendant effect finished; arbitration={:?}, observations={:?}",
        result.arbitration, result.observations
    );
    assert_eq!(result.observations["nested"]["work"], "effect received");
}
