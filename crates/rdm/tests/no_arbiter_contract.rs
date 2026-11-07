//! A node with no arbiter keeps the old contract: siblings and partials, no ruling, and the first selected error.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]
use rdm::record::Event;
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::Duration;
use systemone::{Answer, Scripted};
const LIMIT: Duration = Duration::from_secs(5);
fn dm() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions.keys().map(|k| (k.clone(), Answer::noul(0.9))).collect::<Vec<_>>()
    }))
}
#[test]
fn default_node_records_and_shares_partials_without_model_arbitration() {
    let (published, ready) = channel();
    let ready = Mutex::new(ready);
    let published = Mutex::new(published);
    let (observed, seen) = channel();
    let seen = Mutex::new(seen);
    let observed = Mutex::new(observed);
    let events = Arc::new(Mutex::new(Vec::new()));
    let kept = events.clone();
    let model = dm();
    let root = Rdm::new(
        Signature::new("Read two things")
            .input("q")
            .noul("left", "Left?")
            .noul("right", "Right?"),
    )
    .model(model.clone())
    .tool(Tool::new("left", "Publishes partial", move |c: &Call| {
        c.partial("left evidence");
        published.lock().unwrap().send(()).unwrap();
        seen.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("left final")
    }))
    .tool(Tool::new("right", "Reads sibling partial", move |c: &Call| {
        ready.lock().unwrap().recv_timeout(LIMIT).unwrap();
        let found = c
            .siblings()
            .into_iter()
            .any(|b| b.output == "left" && b.partial == Some(serde_json::json!("left evidence")));
        observed.lock().unwrap().send(()).unwrap();
        assert!(found, "default-node sibling evidence is missing");
        Ok("right final")
    }))
    .record(move |e: &Event| {
        kept.lock().unwrap().push(e.clone());
        Ok(())
    });
    let result = root.call(inputs! {"q"=>"read"}).unwrap();
    assert!(result.arbitration.is_none());
    assert_eq!(model.requests().len(), 1);
    assert_eq!(result.observations["left"], "left final");
    assert_eq!(result.observations["right"], "right final");
    let events = events.lock().unwrap();
    assert!(events.iter().any(|e| e.name() == "partial"));
    assert!(!events.iter().any(|e| e.name() == "arbitrated"));
}
#[test]
fn default_node_preserves_selected_order_of_multiple_fatal_errors() {
    // Width two: left blocks; right fails; only then can queued release run. Thus right's
    // error must be consumed before left is allowed to fail, without timing assumptions.
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let release = Mutex::new(release);
    let root = Rdm::new(
        Signature::new("Run tools")
            .input("q")
            .noul("left", "Left?")
            .noul("right", "Right?")
            .noul("release", "Release?"),
    )
    .threads(2)
    .model(dm())
    .tool(Tool::new("left", "First selected failure", move |_| -> rdm::Result<&str> {
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Err(rdm::Error::Usage("first selected error".into()))
    }))
    .tool(Tool::new("right", "First arriving failure", |_| -> rdm::Result<&str> {
        Err(rdm::Error::Usage("first arriving error".into()))
    }))
    .tool(Tool::new("release", "Unblocks first selected", move |_| {
        release.lock().unwrap().send(()).unwrap();
        Ok("released")
    }));
    let error = root.call(inputs! {"q"=>"q"}).unwrap_err();
    assert_eq!(
        error.to_string(),
        "first selected error",
        "default-node failure/Ended error changed with arrival order"
    );
}
