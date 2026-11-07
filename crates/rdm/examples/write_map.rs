//! Making a starter map. A map is part of the program, and what a language model writes onto it
//! is kept: it can be saved and given to another program. This writes the house's routines once, on
//! models that can write them, and saves them as starter_map.json, which "a starter map" (the
//! example `starter`) loads. Ask the hosted models for this one, with `RDM_DECISION_MODEL` and
//! `RDM_LANGUAGE_MODEL`: a small language model does not write these. The stand-ins write what the
//! hosted models wrote, which is the map the site ships.
//!
//! The Python cell saves the map beside itself, where the site's "a starter map" then finds it.
//! This saves it where `--to` says, and with no `--to` as starter_map.json in the temporary
//! directory, so that a run leaves nothing in the directory it was started in. `starter` loads the
//! map it is given with `--map`, or a starter_map.json in the directory it is run in.
//!
//! The writer is where this differs from the Python cell, as in the example `repl`. There it is a
//! `dspy.RLM`, a language model in a loop of its own, which writes code that calls `define`. There
//! is no RLM here. The writer is one request to the language model, a `Generate`, whose outputs are
//! the function as data: the `name`, the `description` and the `code` that `define` takes.
//! `repl.writer_from` makes that a tool that writes: it defines what was given, and what `define`
//! said is the tool's observation. So the instructions no longer say "and define it with `define`",
//! and there is no output `wrote`. A function that is refused is not written again inside the tool:
//! the refusal is the next request's observation, the writer is shown it in an input `observation`
//! of its own, and the decision model can point at `write` again.
//!
//! ```text
//! cargo run -p rdm --example write_map
//! cargo run -p rdm --example write_map -- --to starter_map.json
//! cargo run -p rdm --example write_map -- "Movie time: turn the TV on and the lights off."
//! cargo run -p rdm --example write_map -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::lm::Generate;
use rdm::repl::{Functions, Repl};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    let language_model = common::language_model(writes)?; // the language model the environment asks for

    // What the form sets: the routines to have written, and where the house starts.
    let form = Form::new(
        "write_map",
        "Making a starter map",
        "Hosted models write the house's routines once, and the map is saved.",
    )
    .lines(
        "routines",
        "Routines to have written, one per line",
        &[
            "Good night: turn the TV and the lights off, close the door and lock it, and bring the heat down to 18 degrees.",
            "Movie time: turn the TV on and the lights off.",
            "Good morning: turn the lights on, unlock the door, and bring the heat up to 21 degrees.",
        ],
    )
    .house(json!({"tv": "on", "lights": "on", "door": "open", "lock": "unlocked", "heat": 21}))
    .text(
        "to",
        "Where to save the map, where it is not starter_map.json in the temporary directory",
        &[],
    )
    .read();
    let routines = form.lines_of("routines");
    let house = form.house_of();
    let to = form.text_of("to");

    // The sandbox, and what written code may call in it: the house's four functions, each with
    // its description. `common::switch`, `cooler` and `warmer` are those, and each redraws the house.
    let (switched, cooled, warmed, read) = (house.clone(), house.clone(), house.clone(), house.clone());
    let repl = Repl::new()
        .function(
            "switch(device: str, to: str)",
            "Set the tv or the lights to on or off, the door to open or closed, or the lock to locked or unlocked.",
            move |arguments| {
                let (device, to): (String, String) = (arguments.get("device")?, arguments.get("to")?);
                Ok(common::switch(&switched, &device, &to))
            },
        )
        .function("cooler", "Make it one degree cooler.", move |_| Ok(common::cooler(&cooled)))
        .function("warmer", "Make it one degree warmer.", move |_| Ok(common::warmer(&warmed)))
        .function("heat", "How warm it is, in degrees.", move |_| Ok(common::heat(&read)));

    let write = Generate::new(
        r#"Write one Python function that does all of what the task asks.

Where the function runs, `switch(device, to)` sets the "tv" or the "lights" to "on" or "off", the "door" to
"open" or "closed" and the "lock" to "locked" or "unlocked" (a door is closed before it is locked);
`cooler()` and `warmer()` change the heat by one degree, and `heat()` returns the degrees it is now.
Name the function for the routine, take no parameters, and return a sentence saying what was done."#,
    )
    .input("task")
    .input_typed("house", "dict")
    .input_described("observation", "What the last step did. If a function was refused, or failed, why is here.")
    .output_described("name", "The function's name.")
    .output_described("description", "One sentence for a menu: what choosing the function does.")
    .output_described(
        "code",
        "The function's source: one `def` of that name and nothing beside it. A parameter is given the request's answer or input of the same name.",
    );

    let writer = write.lm(language_model.clone()); // one request: what it gives, the REPL defines
    let write = repl.writer_from(&writer.tool(
        "write",
        "Have a new tool written in code and put on this list: nothing listed does what is still needed.",
    ));

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let rdm = Rdm::new(operate).tool(write).source(repl.clone()).model(decision_model.clone());

    for task in &routines {
        common::show_house(&house, task);
        let result = rdm.call(inputs! { "task" => task, "house" => &house })?;
        common::trace(&result);
        py::print!(task.split(':').next().unwrap_or_default(), "->", result.pointed("next"));
    }
    // repl.shutdown(): the REPL goes with the program

    // The map, as data: each function that works, by name, with its description and its source.
    let working: Functions = repl.written().into_iter().filter(|(_, entry)| entry.failed.is_none()).collect();
    let starter = json!({
        "made_by": {"decision_model": decision_model.model(), "language_model": language_model.model()},
        "written": working,
    });
    // json.dump(starter, open("starter_map.json", "w"), indent=2)
    let saved = match to.as_str() {
        // A WebAssembly module has no temporary directory, and asking for one is a panic:
        // there the map is saved where the program was run.
        "" if cfg!(target_family = "wasm") => "starter_map.json".into(),
        "" => std::env::temp_dir().join("starter_map.json"),
        to => to.into(),
    };
    std::fs::write(&saved, serde_json::to_string_pretty(&starter)?)
        .map_err(|error| format!("The map cannot be saved as {}: {error}", saved.display()))?;
    py::print!();
    for (name, entry) in &repl.written() {
        py::print!(format!("{name}: {}", entry.failed.as_ref().unwrap_or(&entry.description)));
    }
    let named = if to.is_empty() { "starter_map.json" } else { &to };
    py::print!(format!("saved as {named}: \"a starter map\" loads it for the rest of this session"));
    if to.is_empty() {
        // Where that is, which the cell has no need to say. On stderr: what is printed is what the cell prints.
        let saved = saved.display();
        eprintln!("It is {saved}: `cargo run -p rdm --example starter -- --map {saved}` loads it.");
    }
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one. With
/// nothing done yet it points at `write`, for a routine the stand-in language model writes, or
/// at the routine's function if it is on the map already; when `define` has said a function
/// was written, at that function; and after that at `finish`.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    let shown = |name: &str| inputs[name].as_str().unwrap_or_default();
    let (task, observation) = (shown("task"), shown("observation"));
    let on_the_map = |name: &str| question.labels().iter().any(|label| label == name);
    // "`good_night` was written and is now on the list. It has not been run, ..."
    let written = observation.strip_prefix('`').and_then(|rest| rest.split_once("` was written"));
    match (written, shipped(task)) {
        (Some((name, _)), _) if on_the_map(name) => name.into(),
        (None, Some((name, _))) if observation.is_empty() && on_the_map(&name) => name.into(),
        (None, Some(_)) if observation.is_empty() => "write".into(),
        _ => rdm::FINISH.into(),
    }
}

/// The stand-in language model: the function the hosted models wrote for the routine, as the
/// JSON a model would reply with. For any other task it gives no function, and `define`
/// refuses what it gave.
fn writes(_system: &str, prompt: &str) -> String {
    let task = rdm::lm::input_of(prompt, "task").unwrap_or_default();
    let (name, written) = shipped(task).unwrap_or_default();
    json!({"name": name, "description": written["description"], "code": written["code"]}).to_string()
}

/// What the hosted models wrote for a routine: the function of the map the site ships
/// (`examples/assets/starter_map.json`) that is named for it, as "Good night: ..." is `good_night`.
fn shipped(task: &str) -> Option<(String, Json)> {
    let name = task.split(':').next()?.trim().to_lowercase().replace(' ', "_");
    let map: Json = serde_json::from_str(common::STARTER_MAP).ok()?;
    let written = map["written"].get(&name)?.clone();
    Some((name, written))
}
