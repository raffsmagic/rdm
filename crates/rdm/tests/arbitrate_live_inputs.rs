//! A ruling reads live inputs as they are now, not as the request first showed them.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::systemone::{Answer, Kind, Scripted};
use rdm::{Arbitrate, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[test]
fn a_new_arbitration_request_reads_live_inputs_without_rewriting_the_parent_snapshot() {
    let world = Shared::new(json!({"version":0}));
    let changed = world.clone();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let shown = seen.clone();
    let root_seen = Arc::new(Mutex::new(Vec::new()));
    let root_shown = root_seen.clone();
    let module = Rdm::new(
        Signature::new("Answer against the current world.")
            .input("world")
            .noul("work", "Obtain evidence?"),
    )
    .tool(Tool::new("work", "Obtain evidence and observe a world change.", move |_| {
        changed.set(json!({"version":1}));
        Ok("The world changed while obtaining this evidence.")
    }))
    .arbiter(Arbitrate::new())
    .model(Scripted::new(move |request| {
        if request.questions.contains_key("next") {
            shown.lock().unwrap().push(request.state["inputs"]["world"]["version"].clone());
        } else {
            root_shown.lock().unwrap().push(request.state["inputs"]["world"]["version"].clone());
        }
        request
            .questions
            .iter()
            .map(|(name, question)| {
                let answer = match question.kind {
                    Kind::Choice => {
                        let labels: Vec<String> = question.criteria.as_ref().unwrap().as_object().unwrap().keys().cloned().collect();
                        Answer::chose(&labels[0], labels.iter().map(String::as_str))
                    }
                    _ => Answer::noul(0.99),
                };
                (name.clone(), answer)
            })
            .collect::<Vec<_>>()
    }));
    module.call(inputs! {"world"=>&world}).unwrap();
    assert_eq!(
        *root_seen.lock().unwrap(),
        vec![json!(0)],
        "The original request must remain an immutable observation."
    );
    assert_eq!(world.get()["version"], 1);
    assert_eq!(
        *seen.lock().unwrap(),
        vec![json!(1)],
        "Arbitration is a NEW request about live state; it must read world version1, without altering the version0 parent observation."
    );
}
