//! An arbiter is told whether each ruling was acted on, keyed by its request.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
#![allow(clippy::type_complexity)]
use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::record::Event;
use rdm::{Ask, Call, Rdm, Signature, Tool, inputs};
use std::sync::{Arc, Mutex, mpsc};
use systemone::{Answer, Scripted};
type Reports = Arc<Mutex<Vec<((u64, usize), usize, bool)>>>;
fn model() -> Scripted {
    Scripted::new(|r| r.questions.keys().map(|n| (n.clone(), Answer::noul(0.99))).collect::<Vec<_>>())
}
#[allow(clippy::type_complexity)]
struct Stale {
    release: mpsc::Sender<()>,
    reported: Reports,
}
impl Arbiter for Stale {
    fn arbitrate(&self, l: &Ledger, c: &Call) -> rdm::Result<Ruling> {
        if l.round == 0 {
            self.release.send(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while !c.siblings().iter().any(|b| b.output == "producer" && b.state == State::Returned) {
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
        Ok(Ruling {
            used: vec![if l.round == 0 { "quick" } else { "producer" }.into()],
            done: true,
            ..Default::default()
        })
    }
    fn acted(&self, l: &Ledger, _: &Ruling, applied: bool) {
        self.reported.lock().unwrap().push((l.request, l.round, applied));
    }
}
#[test]
fn stale_and_applied_callbacks_have_exact_request_and_attempt_identity() {
    let (entered, waiting) = mpsc::channel();
    let waiting = Mutex::new(waiting);
    let (release, released) = mpsc::channel();
    let released = Mutex::new(released);
    let reports: Reports = Arc::new(Mutex::new(vec![]));
    let got = Rdm::new(
        Signature::new("Correct a stale answer")
            .noul("quick", "Read first?")
            .noul("producer", "Read corrected?"),
    )
    .model(model())
    .arbiter(Stale {
        release,
        reported: reports.clone(),
    })
    .tool(Tool::new("quick", "Initial", move |_| {
        waiting.lock().unwrap().recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        Ok("first")
    }))
    .tool(Tool::new("producer", "Correction", move |_| {
        entered.send(()).unwrap();
        released.lock().unwrap().recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        Ok("corrected")
    }))
    .call(inputs! {})
    .unwrap();
    assert_eq!(got.arbitration.unwrap().used, ["producer"]);
    let r = reports.lock().unwrap();
    assert_eq!(r.len(), 2, "{r:?}");
    assert_eq!(r[0].0, r[1].0);
    assert_eq!((r[0].1, r[0].2), (0, false));
    assert_eq!((r[1].1, r[1].2), (1, true));
}
struct Observe {
    seen: Arc<Mutex<Vec<((u64, usize), String)>>>,
    reports: Reports,
}
impl Arbiter for Observe {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        self.seen.lock().unwrap().push((l.request, l.signature.instructions().to_string()));
        Ok(Ruling {
            used: l
                .branches
                .iter()
                .filter(|b| b.state == State::Returned)
                .map(|b| b.output.clone())
                .collect(),
            wait: l
                .branches
                .iter()
                .filter(|b| b.state == State::Running)
                .map(|b| b.output.clone())
                .collect(),
            done: !l.branches.iter().any(|b| b.state == State::Running),
            ..Default::default()
        })
    }
    fn acted(&self, l: &Ledger, _: &Ruling, applied: bool) {
        self.reports.lock().unwrap().push((l.request, l.round, applied));
    }
}
#[test]
fn shared_arbiter_request_keys_separate_nested_and_later_calls() {
    let seen = Arc::new(Mutex::new(vec![]));
    let reports: Reports = Arc::new(Mutex::new(vec![]));
    let rdm = Rdm::new(Signature::new("Parent request").noul("nested", "Run child?"))
        .max_depth(2)
        .model(model())
        .arbiter(Observe {
            seen: seen.clone(),
            reports: reports.clone(),
        })
        .tool(Tool::new("nested", "Run nested child", |call| {
            let got = call.sub_with(Ask::new(inputs! {}).signature(Signature::new("Child request").noul("leaf", "Read child?")))?;
            Ok(serde_json::json!({"child":got.observations["leaf"]}))
        }))
        .tool(Tool::new("leaf", "Synthetic read", |_| Ok("child result")));
    for _ in 0..2 {
        rdm.call(inputs! {}).unwrap();
    }
    let seen = seen.lock().unwrap();
    let keys: std::collections::HashSet<_> = seen.iter().map(|(key, _)| *key).collect();
    assert_eq!(seen.len(), 4, "{seen:?}");
    assert_eq!(keys.len(), 4, "nested/later requests collided: {seen:?}");
    assert_eq!(seen.iter().filter(|(_, label)| label == "Child request").count(), 2);
    let reports = reports.lock().unwrap();
    assert_eq!(reports.len(), 4);
    assert!(reports.iter().all(|(key, _, applied)| *applied && keys.contains(key)));
}
#[test]
fn refusal_to_record_a_ruling_does_not_report_it_as_applied() {
    let seen = Arc::new(Mutex::new(vec![]));
    let reports: Reports = Arc::new(Mutex::new(vec![]));
    let got = Rdm::new(Signature::new("Refused ruling").noul("only", "Read it?"))
        .model(model())
        .arbiter(Observe {
            seen: seen.clone(),
            reports: reports.clone(),
        })
        .tool(Tool::new("only", "Synthetic read", |_| Ok("result")))
        .record(|event: &Event| {
            if matches!(event, Event::Arbitrated { .. }) {
                Err(rdm::Error::Usage("refuse ruling record".into()))
            } else {
                Ok(())
            }
        })
        .call(inputs! {});
    assert!(got.is_err());
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(reports.lock().unwrap().is_empty());
}
