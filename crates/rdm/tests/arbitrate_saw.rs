//! What a ruling was shown of each branch is recorded with it: the order of events alone may not
//! tell, since a `partial` event is recorded after the ledger has what it says.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
use rdm::arbitrate::{Arbiter, Ledger, Ruling};
use rdm::record::Event;
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use systemone::{Answer, Scripted};

const DEADLINE: Duration = Duration::from_secs(5);

struct UseFast;

impl Arbiter for UseFast {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        assert_eq!(ledger.branch("slow").unwrap().version, 1);
        Ok(Ruling {
            used: vec!["fast".into()],
            done: true,
            ..Default::default()
        })
    }
}

#[test]
fn a_ruling_names_the_version_it_saw_even_before_that_version_is_recorded() {
    let ruled = Arc::new(AtomicBool::new(false));
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let (seen, kept) = (ruled.clone(), events.clone());
    let (recorded_partial, partial_recorded) = mpsc::channel();
    let slow_said: Arc<Mutex<Option<Call>>> = Arc::new(Mutex::new(None));
    let watched = slow_said.clone();
    let result = Rdm::new(
        Signature::new("Answer from what came")
            .input("q")
            .noul("fast", "Look it up?")
            .noul("slow", "Work it out?"),
    )
    .model(Scripted::new(|_| [("fast", Answer::noul(0.99)), ("slow", Answer::noul(0.99))]))
    .arbiter(UseFast)
    .threads(2)
    .record(move |event: &Event| {
        // The ledger has slow's first version; its event waits until a ruling is recorded.
        if matches!(event, Event::Partial { output, .. } if output == "slow") {
            let until = Instant::now() + DEADLINE;
            while !seen.load(Ordering::SeqCst) {
                assert!(Instant::now() < until, "No ruling was recorded while slow's partial event waited");
                std::thread::yield_now();
            }
        }
        kept.lock().unwrap().push(event.clone());
        // Release the partial only after the ruling is actually in the test's record.
        if matches!(event, Event::Arbitrated { .. }) {
            seen.store(true, Ordering::SeqCst);
        }
        if matches!(event, Event::Partial { output, .. } if output == "slow") {
            recorded_partial.send(()).unwrap();
        }
        Ok(())
    })
    .tool(Tool::new("fast", "Comes back once slow has said something", move |call: &Call| {
        let until = Instant::now() + DEADLINE;
        while !call.siblings().iter().any(|b| b.output == "slow" && b.version == 1) {
            assert!(Instant::now() < until, "Slow never said what it had");
            std::thread::yield_now();
        }
        Ok("looked up")
    }))
    .tool(Tool::new("slow", "Says what it has, then finishes", move |call: &Call| {
        *watched.lock().unwrap() = Some(call.clone());
        call.partial("half worked");
        let until = Instant::now() + DEADLINE;
        while !call.stopped() && Instant::now() < until {
            std::thread::yield_now();
        }
        Ok("worked out")
    }))
    .call(inputs! {"q" => "question"})
    .unwrap();
    // A stopped worker may still be finishing its recorder callback after the parent returns.
    partial_recorded.recv_timeout(DEADLINE).unwrap();
    assert_eq!(result.arbitration.unwrap().used, ["fast"]);
    let events = events.lock().unwrap();
    let at = |find: &dyn Fn(&Event) -> bool| events.iter().position(find).unwrap();
    let ruling = at(&|e| matches!(e, Event::Arbitrated { .. }));
    let partial = at(&|e| matches!(e, Event::Partial { output, .. } if output == "slow"));
    assert!(ruling < partial, "The case did not record the ruling first: {events:?}");
    let Event::Arbitrated { saw, .. } = &events[ruling] else {
        unreachable!()
    };
    assert_eq!(saw, &[("fast".to_owned(), 0), ("slow".to_owned(), 1)]);
}
