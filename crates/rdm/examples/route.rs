//! With a language model. A language model is a tool like any other: it writes, and the decision
//! model decides about what it wrote. Here it splits a request into parts, and each part is a
//! request of its own, routed one level down by the same module: that is the recursion.
//!
//! ```text
//! cargo run -p rdm --example route
//! cargo run -p rdm --example route -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::lm::{Generate, input_of};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    let language_model = common::language_model(writes)?; // the language model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new(
        "route",
        "With a language model",
        "The decision model routes; a language model writes only where writing is needed.",
    )
    .text(
        "task",
        "Ask the house",
        &[
            "Turn the TV on, switch the lights off, and tell me whether it will rain.",
            "Close the door.",
            "Why is the sky blue?",
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

    let act = Rdm::new(act).tool(switch).model(decision_model.clone());

    let route = Signature::new("Route a request to a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul("split", "Does `inputs.task` ask for more than one distinct thing?")
        .noul("act", "Is `inputs.task` one thing to do to the house?")
        .noul("chat", "Is `inputs.task` only a question to answer in conversation?");

    let write = Generate::parse("task -> parts: list[str]")?.lm(language_model.clone());
    let reply = Generate::parse("task -> reply")?.lm(language_model);

    // Each part the language model writes is a request of its own, routed one level down.
    let split = Tool::new(
        "split",
        "Each part the language model writes is a request of its own, routed one level down.",
        move |call| {
            let (task, house): (String, _) = (call.arg("task")?, call.input("house")?);
            let parts: Vec<String> = write.call_in(call, inputs! { "task" => task })?.read("parts")?;
            let mut routed = Vec::new();
            for part in parts {
                routed.push(call.sub(inputs! { "task" => part, "house" => &house })?);
            }
            Ok(routed)
        },
    );

    // A module is a tool with `Tool::module`: `act` is another Rdm, and joins the run as a sub-call.
    // A `Generate` is one with `tool`: what the language model wrote is the observation.
    let route = Rdm::new(route)
        .tool(split)
        .tool(Tool::module("act", act.signature().instructions(), &act))
        .tool(reply.tool("chat", reply.instructions()))
        .model(decision_model);

    common::show_house(&house, &task);
    let result = route.call(inputs! { "task" => &task, "house" => &house })?;
    common::trace(&result);
    py::print!(house.get());
    py::print!(result.termination_reason);
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one.
///
/// To route: a request with ", and " in it asks for more than one thing; one that names a device
/// and a state is one thing to do; and one that asks ("?", "tell me") and is neither is a
/// question for conversation. To act, it is the rule of the `act` example: the first device
/// the task names and the first state, by a few words.
///
/// The Python tests its reference page with rules for one sentence (`split` on ", and ", `act`
/// when the task starts with "Turn", `chat` on "rain"), by which the whole request is both
/// split and acted on. These are the same idea, widened to the cell's presets, and they route
/// each request one way.
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
    let several = task.contains(", and ");
    let one = device.is_some() && to.is_some() && !several;
    let asks = task.ends_with('?') || task.contains("tell me");
    match name {
        "split" => several.into(),
        "act" => one.into(),
        "chat" => (asks && !several && !one).into(),
        "switch" => (device.is_some() && to.is_some()).into(),
        "device" => device.unwrap_or("tv").into(),
        _ => to.unwrap_or("on").into(),
    }
}

/// The stand-in language model: the JSON a model would write for each of the two signatures.
///
/// The parts of a request are where it has ", and " or a comma, each made a sentence. A reply
/// is one of two it knows, about the weather and about the sky, or that it knows only the house.
fn writes(system: &str, prompt: &str) -> String {
    let task = input_of(prompt, "task").unwrap_or_default();
    if system.contains("`parts`") {
        let pieces = task.split(", and ").flat_map(|half| half.split(", ")).map(str::trim);
        let sentence = |piece: &str| {
            let (first, rest) = piece.split_at(piece.chars().next().map_or(0, char::len_utf8));
            let stop = if piece.ends_with(['.', '?', '!']) { "" } else { "." };
            format!("{}{rest}{stop}", first.to_uppercase())
        };
        let parts: Vec<String> = pieces.filter(|piece| !piece.is_empty()).map(sentence).collect();
        return json!({ "parts": parts }).to_string();
    }
    let replies = [
        ("rain", "I cannot see the weather from here."),
        ("sky", "Sunlight is scattered by the air, and blue light the most."),
    ];
    let knows = replies.iter().find(|(word, _)| task.to_lowercase().contains(word));
    json!({ "reply": knows.map_or("I can only tell you about the house.", |(_, reply)| reply) }).to_string()
}
