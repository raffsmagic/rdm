//! A map with labels. On a long map the decision model reads every description each time it points.
//! `Near` gives each name labels: its answers to the questions of a signature. Each request is
//! asked the same questions, and the pointer is offered only the names nearest it.
//!
//! ```text
//! cargo run -p rdm --example near
//! cargo run -p rdm --example near -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::near::Near;
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new(
        "near",
        "A map with labels",
        "RDM.Near offers each request only the names nearest it.",
    )
    .text(
        "task",
        "Ask the house",
        &["Lock the front door.", "Put the TV on.", "It is too bright in here."],
    )
    .house(json!({"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}))
    .read();
    let task = form.text_of("task");
    let house = form.house_of();

    // Set one device to one state. The house pushes back: an open door cannot be locked.
    // `common::switch` is that, and it redraws the house.

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let steps = [
        ("tv", "on"),
        ("tv", "off"),
        ("lights", "on"),
        ("lights", "off"),
        ("door", "closed"),
        ("lock", "locked"),
    ]
    .map(|(device, to)| {
        let house = house.clone();
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_call| {
            Ok(common::switch(&house, device, to))
        })
    });

    let about = Signature::new("Say what a text is about.")
        .input("task")
        .noul("screen", "Is `inputs.task` about the TV?")
        .noul("light", "Is `inputs.task` about the lights?")
        .noul("entry", "Is `inputs.task` about the door or its lock?");

    let near = Near::new(about).top(2); // of the six steps, the two nearest a request are offered, with `finish`
    let guided = Rdm::new(operate).tools_from(steps).fit(near.clone()).model(decision_model);

    common::show_house(&house, &task);
    let result = guided.call(inputs! { "task" => &task, "house" => &house })?;
    common::trace(&result);
    py::print!("pointed at:", result.pointed("next"));
    let now = house.get();
    py::print!(now["door"], now["lock"]);
    py::print!(
        near.answers().len(),
        "texts were given labels, each asked once: the request and the six names"
    );
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one.
///
/// A label is a yes when the text has a word for it: the rule the Python tests its reference
/// page with (`screen` for "tv", `light` for "light", `entry` for "door" or "lock"), and
/// "bright" for the form's third request. A name on the map is asked as `name: description`,
/// so `lock locked: Set the lock to locked.` is about the door or its lock, and nothing else.
///
/// To point, it takes the one step the request names, by the first device and the first state
/// among a few words, if that step is among the names it was offered and the house does not
/// show it yet. It does not know an open door cannot be locked: it tries, and closes the door
/// when the house says that is in the way.
fn standin(inputs: &Json, name: &str, question: &Question) -> Say {
    let task = inputs["task"].as_str().unwrap_or_default().to_lowercase();
    let has = |words: &[&str]| words.iter().any(|word| task.contains(word));
    match name {
        "screen" => return has(&["tv"]).into(),
        "light" => return has(&["light", "bright"]).into(),
        "entry" => return has(&["door", "lock"]).into(),
        _ => {}
    }
    let named = |words: &[(&'static str, &'static str)]| words.iter().find(|(word, _)| task.contains(word)).map(|(_, means)| *means);
    let device = named(&[
        ("lock", "lock"),
        ("tv", "tv"),
        ("light", "lights"),
        ("bright", "lights"),
        ("door", "door"),
    ]);
    let to = named(&[
        ("lock", "locked"),
        ("shut", "closed"),
        ("close", "closed"),
        (" off", "off"),
        ("bright", "off"),
        (" on", "on"),
    ]);
    let offered = question.labels();
    let offered = |step: &str| offered.iter().any(|name| name == step);
    let refused = inputs["observation"]
        .as_str()
        .is_some_and(|observation| observation.contains("refused, the door is open"));
    if refused && offered("door closed") {
        return "door closed".into(); // the house said what is in the way
    }
    match (device, to) {
        (Some(device), Some(to)) if offered(&format!("{device} {to}")) && inputs["house"][device] != to => format!("{device} {to}").into(),
        _ => "finish".into(),
    }
}
