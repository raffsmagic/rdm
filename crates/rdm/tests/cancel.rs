//! A run stopped from outside it ([`rdm::Cancel`]): it begins nothing more, what its tools would
//! begin outside is not begun, and a module a tool calls stops with it.

use std::sync::{Arc, Mutex};

use rdm::{Ask, Call, Cancel, Rdm, Signature, Tool, inputs};
use systemone::{Answer, Request, Scripted};

/// A loop whose one tool does something outside and asks another module, until `cancel` stops it on
/// its second turn. What it did outside, and what the module it asked was asked, in order.
fn looping(cancel: Cancel) -> (Rdm, Arc<Mutex<Vec<String>>>) {
    let done = Arc::new(Mutex::new(Vec::new()));
    let inner = Rdm::new(Signature::new("Inner.").input("x").noul("ok", "Ok?")).model(Scripted::new(|_| [("ok", Answer::noul(0.9))]));
    let kept = done.clone();
    let step = Tool::new("step", "Do one thing outside, and ask the inner module.", move |call: &Call| {
        let turn = kept.lock().unwrap().len();
        if turn == 2 {
            // Stopped from outside while the tool is between two things.
            cancel.stop();
        }
        let did = call.acting(|| format!("did {turn}"));
        let asked = call.on(&inner).sub(inputs! { "x" => "y" });
        let mut kept = kept.lock().unwrap();
        match (did, asked) {
            (Ok(did), Ok(_)) => kept.push(did),
            (did, asked) => kept.push(format!("stopped: acting {}, inner {}", did.is_ok(), asked.is_ok())),
        }
        // Each turn's observation its own, so that the loop does not come round.
        Ok(format!("turn {turn} done"))
    });
    let module = Rdm::new(
        Signature::new("Go on.")
            .input("task")
            .input("observation")
            .tool("next", "What next?"),
    )
    .tool(step)
    .model(Scripted::new(|request: &Request| {
        let labels = request.questions["next"].labels();
        [("next", Answer::chose("step", labels.iter().map(String::as_str)))]
    }))
    .max_iters(10);
    (module, done)
}

#[test]
fn a_run_stopped_from_outside_begins_nothing_more_outside_or_below() {
    let cancel = Cancel::new();
    let (module, done) = looping(cancel.clone());
    let came = module.call_with(Ask::new(inputs! { "task" => "go", "observation" => "" }).stoppable(&cancel));
    assert!(cancel.is_stopped());
    assert!(came.is_err_and(|error| error.is_stopped()), "the loop ends stopped");
    let done = done.lock().unwrap().clone();
    assert_eq!(done, ["did 0", "did 1", "stopped: acting false, inner false"]);
}

#[test]
fn a_run_not_stopped_goes_on_as_it_would() {
    let cancel = Cancel::new();
    let module = Rdm::new(Signature::new("Judge.").input("record").noul("holds", "Does it hold?"))
        .model(Scripted::new(|_| [("holds", Answer::noul(0.9))]));
    let came = module
        .call_with(Ask::new(inputs! { "record" => "It holds." }).stoppable(&cancel))
        .unwrap();
    assert!(came.yes("holds") && !cancel.is_stopped());
}
