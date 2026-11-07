//! Something done outside whose receipt cannot be written as JSON is kept as done, and why its
//! receipt is not there: never as null.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use rdm::record::Event;
use rdm::{Call, Rdm, Signature, Tool, inputs};
use serde_json::{Value, json};
use systemone::{Answer, Scripted};

#[test]
fn a_receipt_that_is_not_json_is_kept_as_unrecorded() {
    let acted = Arc::new(Mutex::new(Vec::<Value>::new()));
    let kept = acted.clone();
    let result = Rdm::new(Signature::new("Do the work").input("q").noul("work", "Work?"))
        .model(Scripted::new(|_| [("work", Answer::noul(0.9))]))
        .tool(Tool::new("work", "Done outside", |call: &Call| {
            // A map keyed by pairs: JSON keys are strings, so this cannot be written as JSON.
            let done = call.acting(|| BTreeMap::from([((1, 2), "done")]))?;
            Ok(json!({ "done": done.len() }))
        }))
        .record(move |event: &Event| {
            if let Event::Acted { receipt, .. } = event {
                kept.lock().expect("one test thread").push(receipt.clone());
            }
            Ok(())
        })
        .call(inputs! { "q" => "do it" })
        .expect("the run ends");
    let acted = acted.lock().expect("one test thread").clone();
    assert_eq!(acted.len(), 1, "{acted:?}");
    assert!(acted[0]["unrecorded"].as_str().is_some_and(|why| !why.is_empty()), "{acted:?}");
    assert_eq!(result.receipts.len(), 1);
    assert_eq!(result.receipts[0].receipt, acted[0]);
}
