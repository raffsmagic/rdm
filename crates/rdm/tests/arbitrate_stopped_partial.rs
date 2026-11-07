//! A branch a ruling used for what it had so far and stopped, while the request went on for
//! another, is what the request comes to as much as the other: a later ruling is shown what it
//! had said, at the version it was stopped at, and may use it. Deterministic: the other branch
//! comes back only once the first ruling has been acted on.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::record::Event;
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use systemone::{Answer, Scripted};

const DEADLINE: Duration = Duration::from_secs(5);

/// What one ruling saw of each branch: its name, and the version of it.
type Saw = Vec<(String, usize)>;

fn yes() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions.keys().map(|n| (n.clone(), Answer::noul(0.9))).collect::<Vec<_>>()
    }))
}

/// Uses `early` for what it has so far and waits for `late` while `late` runs; uses both, and
/// finishes, once `late` is back. The first ruling acted on lets `late` come back.
struct Settle {
    release: Mutex<Option<mpsc::Sender<()>>>,
    later: Mutex<Vec<(State, Option<usize>)>>,
}

impl Arbiter for Settle {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        let late = ledger.branch("late").map(|branch| branch.state);
        let early = ledger.branch("early").unwrap();
        if late == Some(State::Running) {
            return Ok(Ruling {
                used: vec!["early".into()],
                wait: vec!["late".into()],
                ..Default::default()
            });
        }
        self.later.lock().unwrap().push((early.state, early.shown().map(|_| early.version)));
        Ok(Ruling {
            used: vec!["early".into(), "late".into()],
            done: true,
            ..Default::default()
        })
    }

    fn acted(&self, _: &Ledger, ruling: &Ruling, acted: bool) {
        if acted
            && ruling.wait.iter().any(|w| w == "late")
            && let Some(release) = self.release.lock().unwrap().take()
        {
            release.send(()).unwrap();
        }
    }
}

#[test]
fn a_partial_used_and_stopped_while_the_request_goes_on_is_still_what_it_comes_to() {
    let (release, released) = mpsc::channel();
    let released = Mutex::new(released);
    let stopped = Arc::new(AtomicBool::new(false));
    let seen_stopped = stopped.clone();
    let (observed_stop, stop_observed) = mpsc::channel();
    let saw: Arc<Mutex<Vec<Saw>>> = Arc::new(Mutex::new(Vec::new()));
    let rulings = saw.clone();
    let arbiter = Arc::new(Settle {
        release: Mutex::new(Some(release)),
        later: Mutex::new(Vec::new()),
    });
    let result = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("early", "Say what is known so far?")
            .noul("late", "Work it out?"),
    )
    .model(yes())
    .arbiter(ArbiterOf(arbiter.clone()))
    .record(move |event: &Event| {
        if let Event::Arbitrated { saw, .. } = event {
            rulings.lock().unwrap().push(saw.clone());
        }
        Ok(())
    })
    .tool(Tool::new("early", "Says what it has, then goes on", move |call: &Call| {
        call.partial("what early had so far");
        let deadline = Instant::now() + DEADLINE;
        while !call.stopped() {
            assert!(Instant::now() < deadline, "the first ruling stops it");
            std::thread::yield_now();
        }
        seen_stopped.store(true, Ordering::SeqCst);
        observed_stop.send(()).unwrap();
        Ok("a tail no ruling saw")
    }))
    .tool(Tool::new("late", "Comes back once let", move |_| {
        released.lock().unwrap().recv_timeout(DEADLINE).unwrap();
        Ok("what late found")
    }))
    .call(inputs! { "q" => "question" })
    .unwrap();
    // Stopping a branch does not join its worker: await the observer before reading its flag.
    stop_observed.recv_timeout(DEADLINE).unwrap();
    assert!(stopped.load(Ordering::SeqCst), "the branch was stopped while the request went on");
    // The ruling after it was stopped was shown what it had said, at the version it was stopped at.
    assert_eq!(*arbiter.later.lock().unwrap(), [(State::Stopped, Some(1))]);
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["early", "late"]);
    assert_eq!(arbitration.partial, ["early"]);
    assert_eq!(arbitration.versions, [("early".to_string(), 1)]);
    assert_eq!(arbitration.stopped, ["early"]);
    // Each ruling records the version it saw of each branch: the one after the stop names the
    // stopped branch at the version it was stopped at, as it would a running one, never 0.
    let saw = saw.lock().unwrap();
    let last = saw.last().unwrap();
    assert!(last.contains(&("early".to_string(), 1)), "{saw:?}");
    assert!(last.contains(&("late".to_string(), 0)), "{saw:?}");
}

/// An arbiter held behind an `Arc`, so the test can read what it saw.
struct ArbiterOf(Arc<Settle>);

impl Arbiter for ArbiterOf {
    fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
        self.0.arbitrate(ledger, call)
    }

    fn acted(&self, ledger: &Ledger, ruling: &Ruling, acted: bool) {
        self.0.acted(ledger, ruling, acted)
    }
}

// A stopped worker can still be finishing after its parent has returned.
#[test]
fn parent_can_end_before_stopped_worker_sets_observer_flag() {
    let (release, released) = mpsc::channel();
    let released = Mutex::new(released);
    let (observed, observation) = mpsc::channel();
    let (store, store_allowed) = mpsc::channel();
    let store_allowed = Mutex::new(store_allowed);
    let (stored, finished) = mpsc::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let flag = stopped.clone();
    let arbiter = Arc::new(Settle {
        release: Mutex::new(Some(release)),
        later: Mutex::new(Vec::new()),
    });
    let got = Rdm::new(Signature::new("Answer").input("q").noul("early", "Early?").noul("late", "Late?"))
        .model(yes())
        .arbiter(ArbiterOf(arbiter.clone()))
        .tool(Tool::new("early", "Stops with a paused observer", move |call: &Call| {
            call.partial("what early had so far");
            let deadline = Instant::now() + DEADLINE;
            while !call.stopped() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            observed.send(()).unwrap();
            store_allowed.lock().unwrap().recv_timeout(DEADLINE).unwrap();
            flag.store(true, Ordering::SeqCst);
            stored.send(()).unwrap();
            Ok("a tail no ruling saw")
        }))
        .tool(Tool::new("late", "Independent answer", move |_| {
            released.lock().unwrap().recv_timeout(DEADLINE).unwrap();
            Ok("what late found")
        }))
        .call(inputs! {"q"=>"question"})
        .unwrap();
    // The parent has returned while the worker cannot yet store its flag.
    observation.recv_timeout(DEADLINE).unwrap();
    let before_store = stopped.load(Ordering::SeqCst);
    store.send(()).unwrap();
    finished.recv_timeout(DEADLINE).unwrap();
    let arbitration = got.arbitration.unwrap();
    assert!(!before_store, "the original immediate observer assertion legitimately reads false");
    assert!(stopped.load(Ordering::SeqCst));
    assert_eq!(arbitration.stopped, ["early"]);
    assert_eq!(arbitration.partial, ["early"]);
    assert_eq!(arbitration.versions, [("early".to_owned(), 1)]);
    assert_eq!(got.observations["early"], "what early had so far");
    assert_eq!(*arbiter.later.lock().unwrap(), [(State::Stopped, Some(1))]);
}
