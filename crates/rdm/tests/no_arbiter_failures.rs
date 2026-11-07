//! A node with no arbiter fails as tools run at once and joined did.
#![cfg(not(target_family = "wasm"))]
use rdm::arbitrate::State;
use rdm::{Call, Error, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use systemone::{Answer, Scripted};

fn yes(names: &'static [&'static str]) -> Arc<Scripted> {
    Arc::new(Scripted::new(move |_| {
        names
            .iter()
            .map(|name| ((*name).to_string(), Answer::noul(0.99)))
            .collect::<Vec<_>>()
    }))
}

#[test]
fn default_nodes_publish_siblings_partials_and_receipts_without_arbitration() {
    let (release, continued) = mpsc::channel();
    let continued = Mutex::new(continued);
    let events = Arc::new(Mutex::new(vec![]));
    let kept = events.clone();
    let model = yes(&["producer", "consumer"]);
    let result = Rdm::new(
        Signature::new("Read both branches")
            .noul("producer", "Produce evidence?")
            .noul("consumer", "Read evidence?"),
    )
    .threads(2)
    .model(model.clone())
    .record(move |event: &rdm::record::Event| {
        kept.lock().unwrap().push(event.clone());
        Ok(())
    })
    .tool(Tool::new("producer", "Produce evidence", move |call: &Call| {
        call.partial("partial evidence");
        continued.lock().unwrap().recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        call.acting(|| serde_json::json!({"ran":true,"line":"stub operation"}))?;
        Ok("final evidence")
    }))
    .tool(Tool::new("consumer", "Read sibling evidence", move |call: &Call| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while !call
            .siblings()
            .iter()
            .any(|b| b.output == "producer" && b.partial == Some(serde_json::json!("partial evidence")))
        {
            assert!(std::time::Instant::now() < deadline, "No live sibling partial");
            std::thread::yield_now();
        }
        release.send(()).unwrap();
        while !call
            .siblings()
            .iter()
            .any(|b| b.output == "producer" && b.state == State::Returned && b.result == Some(serde_json::json!("final evidence")))
        {
            assert!(std::time::Instant::now() < deadline, "No live sibling final evidence");
            std::thread::yield_now();
        }
        Ok("read partial and final")
    }))
    .call(inputs! {})
    .unwrap();
    assert!(result.arbitration.is_none());
    assert_eq!(result.observations["producer"], "final evidence");
    assert_eq!(result.observations["consumer"], "read partial and final");
    assert_eq!(result.receipts.len(), 1);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(result.trajectory.len(), 1);
    let events = events.lock().unwrap();
    assert!(events.iter().any(|e| matches!(e, rdm::record::Event::Partial { .. })));
    assert!(!events.iter().any(|e| matches!(e, rdm::record::Event::Arbitrated { .. })));
}

#[test]
fn legacy_error_precedence_follows_selected_order_after_every_branch_finishes() {
    // Width two: first waits, second fails, then third releases first. A queued third cannot
    // start in Arrivals until the second result has been received by the settling loop.
    let (release, held) = mpsc::channel();
    let held = Mutex::new(held);
    let third_done = Arc::new(AtomicBool::new(false));
    let done = third_done.clone();
    let error = Rdm::new(
        Signature::new("Run the three branches")
            .noul("first", "First?")
            .noul("second", "Second?")
            .noul("third", "Third?"),
    )
    .threads(2)
    .model(yes(&["first", "second", "third"]))
    .tool(Tool::new("first", "First selected error, last to return", move |_| {
        held.lock().unwrap().recv_timeout(std::time::Duration::from_secs(3)).unwrap();
        Err::<String, _>(Error::Usage("FIRST_SELECTED".into()))
    }))
    .tool(Tool::new("second", "Second selected error, first to return", |_| {
        Err::<String, _>(Error::Usage("SECOND_SELECTED".into()))
    }))
    .tool(Tool::new("third", "Release first", move |_| {
        done.store(true, Ordering::SeqCst);
        release.send(()).unwrap();
        Ok(true)
    }))
    .call(inputs! {})
    .unwrap_err();
    assert!(
        third_done.load(Ordering::SeqCst),
        "A fatal error returned before other branches completed"
    );
    assert!(
        matches!(error, Error::Usage(ref text) if text == "FIRST_SELECTED"),
        "Default error compatibility changed from selected order to arrival order: {error:?}"
    );
}

#[test]
fn legacy_panic_precedence_is_not_swallowed_by_an_earlier_error() {
    let third_done = Arc::new(AtomicBool::new(false));
    let done = third_done.clone();
    let module = Rdm::new(
        Signature::new("Run the three branches")
            .noul("first", "First?")
            .noul("second", "Second?")
            .noul("third", "Third?"),
    )
    .threads(1)
    .model(yes(&["first", "second", "third"]))
    .record(|event: &rdm::record::Event| {
        if matches!(event, rdm::record::Event::Started {tool,..} if tool == "second") {
            panic!("RECORD_CALLBACK_PANIC");
        }
        Ok(())
    })
    .tool(Tool::new("first", "Fatal usage error", |_| {
        Err::<String, _>(Error::Usage("EARLY_ERROR".into()))
    }))
    .tool(Tool::new("second", "The callback panics before it starts", |_| Ok(true)))
    .tool(Tool::new("third", "Still finish", move |_| {
        done.store(true, Ordering::SeqCst);
        Ok(true)
    }));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| module.call(inputs! {})));
    assert!(
        third_done.load(Ordering::SeqCst),
        "Other selected work did not finish before failure propagated"
    );
    assert!(result.is_err(), "A panic was swallowed by the earlier error: {result:?}");
}
