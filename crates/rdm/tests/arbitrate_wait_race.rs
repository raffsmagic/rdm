//! A waited branch's final, arriving while earlier dispositions are held, is ruled again before the request ends.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use systemone::{Answer, Scripted};

#[test]
fn a_waited_final_arriving_during_another_disposition_is_ruled_on() {
    struct WaitThenUse(Arc<AtomicUsize>);
    impl Arbiter for WaitThenUse {
        fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                assert_eq!(ledger.branch("slow").unwrap().state, State::Running);
                return Ok(Ruling {
                    wait: vec!["slow".into()],
                    ..Default::default()
                });
            }
            assert_eq!(ledger.branch("slow").unwrap().state, State::Returned);
            Ok(Ruling {
                used: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let (entered, first_entered) = mpsc::channel();
    let first_entered = Mutex::new(first_entered);
    let (finish, worker_finish) = mpsc::channel();
    let worker_finish = Mutex::new(worker_finish);
    let observer: Arc<Mutex<Option<Call>>> = Arc::new(Mutex::new(None));
    let observed = observer.clone();
    let result = Rdm::new(
        Signature::new("Wait for requested evidence")
            .input("q")
            .noul("withdraw", "Unneeded work?")
            .noul("slow", "Read evidence?"),
    )
    .model(Scripted::new(|_| [("withdraw", Answer::noul(0.99)), ("slow", Answer::noul(0.99))]))
    .arbiter(WaitThenUse(calls.clone()))
    .threads(2)
    .record(move |event: &rdm::record::Event| {
        if matches!(event, rdm::record::Event::Stopped {output,..} if output == "withdraw") {
            // This hook is after the initial freshness check and before slow's disposition.
            finish.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(3);
            while !observer
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .siblings()
                .iter()
                .any(|b| b.output == "slow" && b.state == State::Returned)
            {
                assert!(Instant::now() < deadline, "Waited final never reached ledger");
                std::thread::yield_now();
            }
        }
        Ok(())
    })
    .tool(Tool::new(
        "withdraw",
        "Withdraw this before inspecting the waited branch",
        move |call: &Call| {
            *observed.lock().unwrap() = Some(call.clone());
            entered.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while !call.stopped() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            Err::<String, _>(rdm::Error::Stopped("withdrawn".into()))
        },
    ))
    .tool(Tool::new(
        "slow",
        "Publish progress then return requested evidence",
        move |call: &Call| {
            first_entered.lock().unwrap().recv_timeout(Duration::from_secs(3)).unwrap();
            call.partial("working");
            worker_finish.lock().unwrap().recv_timeout(Duration::from_secs(10)).unwrap();
            Ok("requested final evidence")
        },
    ))
    .call(inputs! {"q" => "Read final evidence"})
    .unwrap();
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "The WAIT ruling ended without reviewing the final it waited for"
    );
    assert_eq!(result.arbitration.unwrap().used, ["slow"]);
    assert_eq!(result.observations["slow"], "requested final evidence");
}
