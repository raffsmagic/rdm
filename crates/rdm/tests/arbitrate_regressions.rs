//! The first ruling after a result asks about it, a queued branch
//! waits for the ruling, and a branch stopped while its request is asked starts no tool.
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
    // Yes to every yes or no; `next`, the ruling's one choice (since v2), is `finish`.
    Arc::new(Scripted::new(|r| {
        r.questions
            .iter()
            .map(|(n, q)| match n.as_str() {
                "next" => (n.clone(), Answer::chose("finish", q.labels().iter().map(String::as_str))),
                _ => (n.clone(), Answer::noul(0.9)),
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
