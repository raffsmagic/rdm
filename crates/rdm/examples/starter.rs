//! A starter map. The house's routines were written once, by a language model that a decision model
//! pointed at ("making a starter map", the example `write_map`), and saved. Loaded here, they are on
//! the map from the first request, so the model that runs this has only to point: no language model
//! is asked anything.
//!
//! The map is `starter_map.json`: the one in the directory this is run in, if `write_map` was told
//! to save one there, and else the one the site ships, which is in `examples/assets/`. `--map`
//! names another.
//!
//! ```text
//! cargo run -p rdm --example starter
//! cargo run -p rdm --example starter -- "Time for a film." "Morning!"
//! cargo run -p rdm --example starter -- --map /tmp/starter_map.json
//! cargo run -p rdm --example starter -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::repl::{Functions, Repl};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the requests, one after another, and where the house starts.
    let form = Form::new(
        "starter",
        "A starter map",
        "The saved routines are on the map from the first request: the model has only to point.",
    )
    .lines(
        "tasks",
        "Ask the house, one request per line",
        &["Let's watch a film.", "We are off to bed.", "Morning!"],
    )
    .house(json!({"tv": "on", "lights": "on", "door": "open", "lock": "unlocked", "heat": 21}))
    .text("map", "The map to load, by its path, where it is not starter_map.json", &[])
    .read();
    let tasks = form.lines_of("tasks");
    let house = form.house_of();

    // open("starter_map.json")
    let starter = match form.text_of("map") {
        path if path.is_empty() => common::starter_map(),
        path => std::fs::read_to_string(&path).map_err(|error| format!("The map {path} cannot be read: {error}"))?,
    };
    let starter: Json = serde_json::from_str(&starter)?;
    let written: Functions = serde_json::from_value(starter["written"].clone())?;
    py::print!(
        "a map made by",
        starter["made_by"]["decision_model"],
        "and",
        starter["made_by"]["language_model"]
    );
    for (name, entry) in &written {
        py::print!(format!("  {name}: {}", entry.description));
    }
    py::print!();

    // The functions run in a REPL of this program's own, against this program's house. What
    // they may call there is the house's four functions, each with its description: `common::switch`,
    // `cooler` and `warmer` are those, and each redraws the house.
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
        .function("heat", "How warm it is, in degrees.", move |_| Ok(common::heat(&read)))
        .with_written(written);

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let rdm = Rdm::new(operate).source(repl.clone()).model(decision_model);

    for task in &tasks {
        common::show_house(&house, task);
        let result = rdm.call(inputs! { "task" => task, "house" => &house })?;
        common::trace(&result);
        py::print!(task, "->", result.pointed("next"));
    }
    py::print!(house.get());
    Ok(()) // repl.shutdown(): the REPL goes with the program
}

/// The stand-in decision model, which answers when the environment asks for no real one. It is
/// the rule of the site's own mock (`test/notebook.mjs`): the routine the task is about, once, and
/// then `finish`. A task about none of them is finished at once.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    let task = inputs["task"].as_str().unwrap_or_default().to_lowercase();
    let about = |words: &[&str]| words.iter().any(|word| task.contains(word));
    let routine = match () {
        _ if about(&["film", "movie"]) => "movie_time",
        _ if about(&["bed", "night"]) => "good_night",
        _ if about(&["morning"]) => "good_morning",
        _ => rdm::FINISH,
    };
    let ran = inputs["observation"].as_str().is_some_and(|observation| !observation.is_empty());
    let on_the_map = question.labels().iter().any(|name| name == routine);
    if ran || !on_the_map { rdm::FINISH.into() } else { routine.into() }
}
