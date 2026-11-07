//! A stopped branch begins no tool by name, and a request that
//! fails stops every branch it left running.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Ledger, Ruling, State};
use rdm::{Arbiter, Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::Duration;
use systemone::{Answer, Scripted};

struct AcceptQuick;
impl Arbiter for AcceptQuick {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        let used = l
            .in_state(State::Returned)
            .filter(|b| b.output == "quick")
            .map(|b| b.output.clone())
            .collect();
        Ok(Ruling {
            used,
            done: true,
            ..Default::default()
        })
    }
}
fn model() -> Scripted {
    Scripted::new(|r| {
        r.questions
            .keys()
            .map(|k| (k.clone(), Answer::noul(if k == "effect" { 0.1 } else { 0.9 })))
            .collect::<Vec<_>>()
    })
}
fn ask() -> Signature {
    Signature::new("Answer")
        .input("q")
        .noul("quick", "Quick?")
        .noul("slow", "Slow?")
        .noul("effect", "Effect?")
}

#[test]
fn a_stopped_branch_cannot_start_another_mapped_tool() {
    let effects = Arc::new(AtomicUsize::new(0));
    let counter = effects.clone();
    let (sent, came) = channel();
    let sent = Mutex::new(sent);
    // `quick` comes back only once `slow` has begun, so `slow` is stopped while it runs.
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let rdm = Rdm::new(ask())
        .model(model())
        .arbiter(AcceptQuick)
        .tool(Tool::new("quick", "Quick", move |_| {
            begins.lock().unwrap().recv().unwrap();
            Ok("answer")
        }))
        .tool(Tool::new("slow", "Slow", move |c: &Call| {
            begun.lock().unwrap().send(()).unwrap();
            while !c.stopped() {
                std::thread::yield_now();
            }
            let effect = c.run("effect");
            sent.lock().unwrap().send(effect.map(|_| ())).unwrap();
            Ok("done")
        }))
        .tool(Tool::new("effect", "Effect", move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok("changed")
        }));
    let result = rdm.call(inputs! {"q"=>"q"}).unwrap();
    assert_eq!(result.arbitration.unwrap().stopped, ["slow"]);
    let effect_result = came.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        effect_result.is_err(),
        "Call::run dispatched after stopped() was true; effects={}",
        effects.load(Ordering::SeqCst)
    );
}

#[test]
fn a_fatal_branch_error_stops_its_running_siblings() {
    let (ready, heard) = channel();
    let heard = Mutex::new(heard);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let ready = Mutex::new(ready);
    let rdm = Rdm::new(ask())
        .model(model())
        .arbiter(AcceptQuick)
        .tool(Tool::new("quick", "Fails", move |_| -> rdm::Result<&str> {
            heard.lock().unwrap().recv().unwrap();
            Err(rdm::Error::Usage("fatal from selected tool".into()))
        }))
        .tool(Tool::new("slow", "Continues", move |c: &Call| {
            ready.lock().unwrap().send(()).unwrap();
            wait.lock().unwrap().recv().unwrap();
            observed.lock().unwrap().send(c.stopped()).unwrap();
            Ok("late")
        }));
    assert!(rdm.call(inputs! {"q"=>"q"}).unwrap_err().is_usage());
    release.send(()).unwrap();
    assert!(
        came.recv_timeout(Duration::from_secs(5)).unwrap(),
        "sibling still has a live token after the entire call failed"
    );
}
