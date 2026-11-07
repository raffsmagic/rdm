//! The map. An output that is `tool` points at one of the module's tools, and the loop is the
//! module's: the tool pointed at runs, and the same question is asked again about the house as it
//! now is. The map can change: what was done is kept as one more name.
//!
//! ```text
//! cargo run -p rdm --example map
//! cargo run -p rdm --example map -- --help
//! RDM_POST=1 cargo run -p rdm --example map
//! ```
//!
//! The last is the map as the post first writes it, in "Communicating with Maps": the loop
//! written out by hand around one request, for a house with a TV and a light.

mod common;

use std::sync::Arc;

use common::{Decider, Form, Say, py};
use indexmap::IndexMap;
use rdm::systemone::Question;
use rdm::{Ask, Inputs, Json, Output, Rdm, Signature, Tool, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: three requests, and where the house starts before each.
    let form = Form::new(
        "map",
        "The map",
        "The model points at one step at a time. What worked is kept as a new name on the map.",
    )
    .text("first", "First, ask for", &["Lock the front door.", "Turn everything off."])
    .text(
        "routine",
        "Then a routine to keep",
        &["Good night: everything off, and lock up.", "Movie night: TV on, lights off."],
    )
    .text("again", "Then ask for it again", &["Night!", "Time for bed."])
    .house(json!({"tv": "on", "lights": "on", "door": "open", "lock": "unlocked"}))
    .read();
    if as_the_post() {
        return post(decision_model);
    }
    let (first, routine, again) = (form.text_of("first"), form.text_of("routine"), form.text_of("again"));
    let house = form.house_of();
    let start = house.get();

    // Set one device to one state. The house pushes back: an open door cannot be locked.
    // `common::switch` is that, and it redraws the house.

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    // The map: the module's tools, by name, each with a description for the model. `finish` is on it too.
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
    let rdm = Rdm::new(operate).tools_from(steps).model(decision_model.clone());

    // Nothing here says to close the door before locking it: the house refuses, and the next request sees that.
    common::show_house(&house, &first);
    let result = rdm.call(inputs! { "task" => &first, "house" => &house })?;
    common::trace(&result);
    py::print!(first, "->", result.pointed("next"));

    house.set(start.clone());
    common::show_house(&house, &routine);
    let result = rdm.call(inputs! { "task" => &routine, "house" => &house })?;
    common::trace(&result);
    let worked = |(name, observed): (String, Json)| {
        let refused = observed.as_str().is_some_and(|observed| observed.contains("refused"));
        (name != "finish" && !refused).then_some(name)
    };
    let done: Vec<String> = result.steps("next").into_iter().filter_map(worked).collect();
    py::print!(routine, "->", done);

    // Take the steps that worked, in the order they were taken.
    let good_night = {
        let done = done.clone();
        Tool::new("good night", format!("{routine} ({}).", done.join(", ")), move |call| {
            done.iter().map(|name| call.run(name)).collect::<rdm::Result<Vec<Json>>>()
        })
    };

    // The map changes: a tool put in `rdm.tools()` is offered from the next run.
    rdm.tools().insert(good_night);

    house.set(start.clone());
    common::show_house(&house, &again);
    let result = rdm.call(inputs! { "task" => &again, "house" => &house })?;
    common::trace(&result);
    py::print!(again, "->", result.pointed("next"));
    py::print!(house.get());
    Ok(())
}

/// Whether to run the map as the post writes it: `RDM_POST` is set to something other than `0`.
fn as_the_post() -> bool {
    std::env::var("RDM_POST").is_ok_and(|value| !value.is_empty() && value != "0")
}

/// What is behind a name on the post's map: a thing for the code to do.
type Thing = Arc<dyn Fn() + Send + Sync>;

/// The post's map: `{name: (description, thing)}`, a description for the model and a thing for the code.
type Lines = IndexMap<&'static str, (&'static str, Option<Thing>)>;

/// The map as the post first writes it, in "Communicating with Maps". A decision model can only
/// point, so something has to say what it is pointing at. A map is that: a set of names, each
/// with a description for the model and a thing for the code. No tool and no pointer here: one
/// request, asked again by a loop written out.
fn post(decision_model: Decider) -> common::Outcome {
    let point = Rdm::new(Signature::parse("request, house: dict -> next")?).model(decision_model);

    // `lines` is the map: {name: (description, thing)}. Returns the things that were done.
    let run = |lines: &Lines, question: &str, shown: Inputs| -> rdm::Result<Vec<Thing>> {
        let mut done = Vec::new();
        loop {
            let options = lines.iter().map(|(name, (description, _))| (*name, *description));
            let signature = point.signature().clone().with_output("next", Output::choice(question, options));
            let result = point.call_with(Ask::new(shown.clone()).signature(signature))?;
            common::trace(&result);
            let name = result.chosen("next").unwrap_or_default(); // the model points at a name
            let Some(thing) = lines.get(name).and_then(|(_, thing)| thing.clone()) else {
                return Ok(done); // nothing behind the name: stop
            };
            py::print!("pointed at:", name);
            thing(); // the code looks up what the name stands for, and does it
            done.push(thing);
        }
    };

    // That loop is the whole mechanism: show the names, the model points, the code does the thing
    // behind the name, and the names are shown again. Here is a map for a house with a TV and a light.
    let house = common::house(json!({"tv": "off", "lights": "on"}));
    let set = |device: &'static str, to: &'static str| -> Option<Thing> {
        let house = house.clone();
        Some(Arc::new(move || {
            house.lock().insert(device.to_owned(), to.into());
            common::show_house(&house, &format!("{device}: now {to}"));
        }))
    };
    let mut lines: Lines = IndexMap::from([
        ("tv on", ("Turn the TV on.", set("tv", "on"))),
        ("tv off", ("Turn the TV off.", set("tv", "off"))),
        ("lights on", ("Turn the lights on.", set("lights", "on"))),
        ("lights off", ("Turn the lights off.", set("lights", "off"))),
        (
            "stop",
            ("Everything `inputs.request` asks for is already so in `inputs.house`.", None),
        ),
    ]);
    let question = "What should happen next for `inputs.request`, with the house as `inputs.house` shows it?";

    common::show_house(&house, "Movie time.");
    let done = run(&lines, question, inputs! { "request" => "Movie time.", "house" => &house })?;
    // pointed at: tv on
    // pointed at: lights off

    // It gets more interesting when the map changes. What was just done can go back on the map as
    // one more name. Its description says what it does, because the description is all the model
    // sees of it.
    let again: Thing = Arc::new(move || done.iter().for_each(|thing| thing()));
    lines.insert(
        "movie time",
        ("Do again what was done for 'Movie time.': TV on, lights off.", Some(again)),
    );

    house.set(common::house(json!({"tv": "off", "lights": "on"})).get()); // the next evening
    common::show_house(&house, "Let's watch a film.");
    run(&lines, question, inputs! { "request" => "Let's watch a film.", "house" => &house })?;
    // pointed at: movie time
    Ok(())
}

/// What a request wants of the house, by a word in it: the steps that bring it about, in the
/// order the stand-in takes them.
const WANTS: [(&str, &[&str]); 6] = [
    ("movie", &["tv on", "lights off"]),
    ("film", &["tv on", "lights off"]),
    ("night", &["tv off", "lights off", "lock locked"]),
    ("bed", &["tv off", "lights off", "lock locked"]),
    ("everything off", &["tv off", "lights off"]),
    ("lock", &["lock locked"]),
];

/// The stand-in decision model, which answers when the environment asks for no real one. It
/// points at the first step the request wants that the house does not yet show, and stops when
/// there is none. It does not know that an open door cannot be locked: it tries, and when the
/// last step was refused for that it closes the door. A name that was kept is pointed at when
/// its description names more than one of the steps still to take.
///
/// The Python tests its reference page with a list of names for each task, given in turn
/// (`lock locked`, `door closed`, `lock locked`, `finish`). These rules give the same names
/// from what each request shows, and so for the form's other requests too.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    let asked = inputs["task"]
        .as_str()
        .or(inputs["request"].as_str())
        .unwrap_or_default()
        .to_lowercase();
    let wanted = WANTS
        .iter()
        .find(|(word, _)| asked.contains(word))
        .map_or(&[][..], |(_, steps)| steps);
    let shows = |step: &&str| step.split_once(' ').is_some_and(|(device, to)| inputs["house"][device] == to);
    let unmet: Vec<&str> = wanted.iter().copied().filter(|step| !shows(step)).collect();
    if inputs["observation"]
        .as_str()
        .is_some_and(|observation| observation.contains("refused, the door is open"))
    {
        return "door closed".into(); // the house said what is in the way
    }
    // The names that were kept, and what each says it does: `good night` in the cell, `movie time` in the post.
    for kept in ["good night", "movie time"] {
        let says = question.criteria.as_ref().and_then(|criteria| criteria[kept].as_str());
        let says = says.unwrap_or_default().to_lowercase();
        if unmet.iter().filter(|step| says.contains(**step)).count() > 1 {
            return kept.into(); // one name that does several of the steps still to take
        }
    }
    let stop = if inputs["request"].is_string() { "stop" } else { "finish" };
    unmet.first().copied().unwrap_or(stop).into()
}
