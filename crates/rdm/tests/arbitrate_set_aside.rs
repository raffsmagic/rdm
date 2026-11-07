//! A ruling set aside is counted among the rulings made, and not shown as tried.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use systemone::{Answer, Scripted};

type History = Arc<Mutex<Vec<(usize, Vec<Ruling>)>>>;
struct Reconsider {
    attempts: Arc<AtomicUsize>,
    release: Mutex<mpsc::Sender<()>>,
    history: History,
}
impl Arbiter for Reconsider {
    fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
        self.history.lock().unwrap().push((ledger.round, ledger.before.clone()));
        if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            self.release.lock().unwrap().send(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while !call.siblings().iter().any(|b| b.output == "producer" && b.state == State::Returned) {
                assert!(std::time::Instant::now() < deadline, "Correcting evidence never arrived");
                std::thread::yield_now();
            }
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
#[allow(clippy::type_complexity)]
fn run() -> (Vec<(usize, Vec<Ruling>)>, Vec<usize>, usize) {
    let (entered, producer_entered) = mpsc::channel();
    let producer_entered = Mutex::new(producer_entered);
    let (release, producer_release) = mpsc::channel();
    let producer_release = Mutex::new(producer_release);
    let history: History = Arc::new(Mutex::new(vec![]));
    let events = Arc::new(Mutex::new(vec![]));
    let kept = events.clone();
    let result = Rdm::new(
        Signature::new("Correct initial evidence")
            .input("q")
            .noul("quick", "Read initial evidence?")
            .noul("producer", "Read correcting evidence?"),
    )
    .model(Scripted::new(|_| [("quick", Answer::noul(0.99)), ("producer", Answer::noul(0.99))]))
    .arbiter(Reconsider {
        attempts: Arc::new(AtomicUsize::new(0)),
        release: Mutex::new(release),
        history: history.clone(),
    })
    .record(move |event: &rdm::record::Event| {
        if let rdm::record::Event::Arbitrated { round, .. } = event {
            kept.lock().unwrap().push(*round);
        }
        Ok(())
    })
    .tool(Tool::new("quick", "Initial hypothesis", move |_| {
        producer_entered
            .lock()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        Ok("wrong initial value")
    }))
    .tool(Tool::new("producer", "Correcting evidence", move |_| {
        entered.send(()).unwrap();
        producer_release
            .lock()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        Ok("corrected value")
    }))
    .call(inputs! {"q" => "Report the supported value"})
    .unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["producer"]);
    let history = history.lock().unwrap().clone();
    let events = events.lock().unwrap().clone();
    (history, events, arbitration.rounds)
}
#[test]
fn a_stale_ruling_is_not_shown_as_applied_history() {
    let (history, events, count) = run();
    assert_eq!(history.len(), 2);
    assert!(
        history[1].1.is_empty(),
        "The rejected first ruling leaked into applied history: {history:?}"
    );
    assert_eq!(count, 2);
    assert_eq!(events.len(), 2, "A discarded attempt vanished from audit events");
}
#[test]
fn stale_attempts_keep_distinct_round_numbers_in_the_audit_record() {
    let (history, events, count) = run();
    assert_eq!(count, 2);
    assert_eq!(events, [0, 1], "Distinct ruling attempts have duplicate audit round identities");
    assert_eq!(
        history.iter().map(|h| h.0).collect::<Vec<_>>(),
        [0, 1],
        "Ledger round conflates attempt count with applied-history length"
    );
}
