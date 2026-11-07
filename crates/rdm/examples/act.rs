//! One request, and what its answers run. A request is a signature whose outputs are decisions.
//! An output named after a tool calls it on a yes, and the tool's arguments are other answers of
//! the same request: the model chooses them, it does not write them.
//!
//! ```text
//! cargo run -p rdm --example act
//! cargo run -p rdm --example act -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new("act", "One request", "One decision points at a tool and chooses its arguments.")
        .text(
            "task",
            "Ask the house",
            &[
                "Turn the TV on.",
                "Switch the lights off.",
                "Please shut the door.",
                "What a lovely evening.",
            ],
        )
        .house(json!({"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}))
        .read();
    let task = form.text_of("task");
    let house = form.house_of();

    // Set one device to one state. The house pushes back: an open door cannot be locked.
    // `common::switch` is that, and it redraws the house.
    let switch = {
        let house = house.clone();
        Tool::new(
            "switch",
            "Set one device to one state. The house pushes back: an open door cannot be locked.",
            move |call| {
                let (device, to): (String, String) = (call.arg("device")?, call.arg("to")?);
                Ok(common::switch(&house, &device, &to))
            },
        )
    };

    let act = Signature::new("Operate a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul(
            "switch",
            "Does `inputs.task` ask for one device in `inputs.house` to be set to another state?",
        )
        .choice("device", "Which device does `inputs.task` mean?", ["tv", "lights", "door", "lock"])
        .choice(
            "to",
            "Which state does `inputs.task` ask for?",
            ["on", "off", "open", "closed", "locked", "unlocked"],
        );

    let act = Rdm::new(act).tool(switch).model(decision_model);

    common::show_house(&house, &task);
    let result = act.call(inputs! { "task" => &task, "house" => &house })?; // one request, three questions
    common::trace(&result);
    py::print!(result.observations);
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one: the
/// first device the task names and the first state, by a few words. It is the rule the Python
/// tests its reference page with (`switch` yes, `tv`, `off` if the task says off, else `on`),
/// widened to the cell's presets.
fn standin(inputs: &Json, name: &str, _question: &Question) -> Say {
    let task = inputs["task"].as_str().unwrap_or_default().to_lowercase();
    let named = |words: &[(&'static str, &'static str)]| words.iter().find(|(word, _)| task.contains(word)).map(|(_, means)| *means);
    let device = named(&[("lock", "lock"), ("tv", "tv"), ("light", "lights"), ("door", "door")]);
    let to = named(&[
        ("unlock", "unlocked"),
        ("lock", "locked"),
        ("shut", "closed"),
        ("close", "closed"),
        ("open", "open"),
        (" off", "off"),
        (" on", "on"),
    ]);
    match name {
        "switch" => (device.is_some() && to.is_some()).into(),
        "device" => device.unwrap_or("tv").into(),
        _ => to.unwrap_or("on").into(),
    }
}
