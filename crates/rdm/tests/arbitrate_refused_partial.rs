//! A record that refused an event ends the run: nothing more is
//! begun, outside or in.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Ledger, Ruling};
use rdm::record::{Event, replay};
use rdm::{Arbiter, Arbitrate, Call, Rdm, Signature, Tool, inputs};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::{Duration, Instant};
const LIMIT: Duration = Duration::from_secs(5);
const MARKER: &str = "accepted-partial-evidence-q3";
struct Accept;
impl Arbiter for Accept {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        assert!(l.branch("slow").unwrap().partial.is_some());
        Ok(Ruling {
            used: vec!["slow".into()],
            done: true,
            ..Default::default()
        })
    }
}
fn model() -> Arc<dyn systemone::DecisionModel> {
    Arc::new(
        systemone::Scripted::new(|r| {
            r.questions
                .iter()
                .map(|(name, q)| {
                    let a = if name == "next" {
                        systemone::Answer::choice(q.labels().iter().map(|l| (l.clone(), if l == "finish" { 1.0 } else { 0.0 })))
                    } else {
                        systemone::Answer::noul(0.9)
                    };
                    (name.clone(), a)
                })
                .collect::<Vec<_>>()
        })
        .named("partial-replay"),
    )
}
fn run(dm: Arc<dyn systemone::DecisionModel>, arbiter: Arc<dyn Arbiter>, partials: Vec<Value>) -> (rdm::Prediction, Vec<Event>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let kept = events.clone();
    let (ended, heard) = channel();
    let ended = Mutex::new(ended);
    let root = Rdm::new(Signature::new("Answer using evidence").input("q").noul("slow", "Find the answer?"))
        .model(dm)
        .arbiter_shared(arbiter)
        .tool(Tool::new(
            "slow",
            "Publishes then is stopped",
            move |c: &Call| -> rdm::Result<&str> {
                for p in &partials {
                    c.partial(p.clone());
                }
                let deadline = Instant::now() + LIMIT;
                while !c.stopped() {
                    assert!(Instant::now() < deadline, "producer never stopped");
                    std::thread::yield_now();
                }
                ended.lock().unwrap().send(()).unwrap();
                c.go_on()?;
                Ok("never returned")
            },
        ))
        .record(move |e: &Event| {
            kept.lock().unwrap().push(e.clone());
            Ok(())
        });
    let result = root.call(inputs! {"q"=>"question"}).unwrap();
    heard.recv_timeout(LIMIT).unwrap();
    let events = events.lock().unwrap().clone();
    assert!(
        !events.iter().any(|e| matches!(e,Event::Returned{tool,..}if tool=="slow")),
        "fixture must exercise a stopped producer with NO final return"
    );
    (result, events)
}
fn partials(events: &[Event]) -> Vec<Value> {
    events
        .iter()
        .filter(|e| e.name() == "partial")
        .map(|e| serde_json::to_value(e).unwrap()["partial"].clone())
        .collect()
}
#[test]
fn an_accepted_partial_has_recorded_snapshot_identity_and_revision_without_a_final_return() {
    let payload = json!({"answer":42,"evidence":MARKER});
    let (result, events) = run(model(), Arc::new(Accept), vec![payload.clone()]);
    assert_eq!(result.observations["slow"], payload);
    let recorded: Vec<Value> = events
        .iter()
        .filter(|e| e.name() == "partial")
        .map(|e| serde_json::to_value(e).unwrap())
        .collect();
    assert_eq!(
        recorded.len(),
        1,
        "accepted partial has no producer event; recorded kinds={:?}",
        events.iter().map(Event::name).collect::<Vec<_>>()
    );
    let p = &recorded[0];
    assert_eq!(p["partial"], payload);
    assert_eq!(p["output"], "slow");
    assert_eq!(p["tool"], "slow");
    assert_eq!(p["index"], 0);
    assert!(p["version"].as_u64().is_some_and(|v| v > 0));
    let pos = |name: &str| events.iter().position(|e| e.name() == name).unwrap();
    assert!(pos("started") < pos("partial") && pos("partial") < pos("arbitrated") && pos("arbitrated") < pos("ended"));
}
#[test]
fn recorded_partial_events_and_model_answers_replay_the_accepted_observation() {
    let payload = json!({"answer":42,"evidence":MARKER});
    let (live, events) = run(model(), Arc::new(Arbitrate::new()), vec![payload.clone()]);
    let replayed_partials = partials(&events);
    assert_eq!(
        replayed_partials,
        vec![payload],
        "record-only replay cannot recover the producer's accepted evidence"
    );
    let (again, _) = run(
        Arc::new(replay("partial-replay", &events)),
        Arc::new(Arbitrate::new()),
        replayed_partials,
    );
    assert_eq!(again.observations, live.observations);
    assert_eq!(again.arbitration, live.arbitration);
}

#[test]
fn partial_provenance_survives_json_roundtrip_and_image_redaction() {
    let payload = json!({"answer":rdm::image("image/png",b"partial-image")});
    let (_, events) = run(model(), Arc::new(Accept), vec![payload.clone()]);
    let event = events.iter().find(|e| e.name() == "partial").expect("missing partial event");
    let serialized = serde_json::to_string(event).unwrap();
    let restored: Event = serde_json::from_str(&serialized).unwrap();
    assert_eq!(&restored, event);
    let redacted = serde_json::to_value(event.without_images()).unwrap();
    assert_eq!(redacted["output"], "slow");
    assert_eq!(redacted["version"], 1);
    assert_eq!(redacted["partial"]["answer"]["type"], "image_ref");
    assert!(redacted["partial"]["answer"]["sha256"].is_string());
}

#[test]
fn a_refused_partial_record_prevents_the_producer_starting_another_tool() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct WaitForProducer(Mutex<std::sync::mpsc::Receiver<()>>);
    impl Arbiter for WaitForProducer {
        fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            self.0.lock().unwrap().recv_timeout(LIMIT).unwrap();
            Ok(Ruling {
                used: vec!["slow".into()],
                done: true,
                ..Default::default()
            })
        }
    }
    let (sent, came) = channel();
    let sent = Mutex::new(sent);
    let effects = Arc::new(AtomicUsize::new(0));
    let counter = effects.clone();
    let root = Rdm::new(Signature::new("Answer").input("q").noul("slow", "Work?"))
        .model(model())
        .arbiter(WaitForProducer(Mutex::new(came)))
        .tool(Tool::new("slow", "Publishes then dispatches", move |c: &Call| {
            c.partial("candidate");
            let result = c.run("effect");
            sent.lock().unwrap().send(()).unwrap();
            result
        }))
        .tool(Tool::new("effect", "External effect", move |c: &Call| {
            c.acting(|| {
                counter.fetch_add(1, Ordering::SeqCst);
                "effect receipt"
            })
        }))
        .record(|e: &Event| {
            if e.name() == "partial" {
                Err(rdm::Error::Tool("partial record refused".into()))
            } else {
                Ok(())
            }
        });
    let result = root.call(inputs! {"q"=>"question"});
    assert!(result.is_err(), "recorder refusal must fail the call");
    assert_eq!(
        effects.load(Ordering::SeqCst),
        0,
        "producer began a new marked external effect AFTER its partial record was refused"
    );
}
