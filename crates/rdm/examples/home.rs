//! The smart home. A decision model runs the house by pointing: each request asks what should happen
//! next, and its answer is one name on a map. The house's map names its rooms, and a room is a map of
//! its own: pointing at one asks the same question again, inside it, about the things in that room.
//! No request shows the house. The model learns how it is from what each step did, and from what
//! the house refuses. It writes nothing: it only ever points.
//!
//! On the site the house keeps its state from one request to the next. Here one run can take
//! several requests in turn, each put to the house as the one before left it.
//!
//! ```text
//! cargo run -p rdm --example home
//! cargo run -p rdm --example home -- "It is too hot in here."
//! cargo run -p rdm --example home -- "We are off to bed." "Morning!" "Set the alarm."   # in turn, to one house
//! cargo run -p rdm --example home -- --task 1 --then 2 --then 11                        # the same, by number
//! cargo run -p rdm --example home -- --help
//! ```

mod common;

use std::sync::Arc;

use common::{Form, Say, py};
use indexmap::IndexMap;
use rdm::systemone::Question;
use rdm::{Ask, Call, Json, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;

/// The requests the page offers. The first five are ones the small local model manages: a
/// routine, or one thing in one room. The rest want the hosted one, and the last four are
/// refused at first: by an open door, an open window, an oven left on.
const PRESETS: [&str; 11] = [
    "We are off to bed.",
    "Morning!",
    "Turn the TV on.",
    "Close the garage door.",
    "We are leaving for the weekend.",
    "Let's watch a film.",
    "I left the oven on.",
    "Set the thermostat to 20 degrees.",
    "Lock the front door.",
    "It is too hot in here.",
    "Set the alarm.",
];

/// What a thing can be set to. One not named here is on or off, and one with nothing to choose is
/// not the house's to set: what is in the garage doorway, how warm it is, the weather.
fn states(thing: &str) -> Vec<Json> {
    let of = |states: &[&str]| states.iter().map(|state| Json::from(*state)).collect();
    match thing {
        "door" | "window" => of(&["open", "closed"]),
        "lock" => of(&["locked", "unlocked"]),
        "alarm" => of(&["armed", "off"]),
        "mode" => of(&["heat", "cool", "off"]),
        "target" => (17..26).map(Json::from).collect(),
        "doorway" | "temp" | "outside" => Vec::new(),
        _ => of(&["on", "off"]),
    }
}

/// Why the house will not do it, if it will not.
fn refused(house: &Json, room: &str, thing: &str, to: &Json) -> Option<String> {
    let (entry, garage, thermostat) = (&house["entry"], &house["garage"], &house["thermostat"]);
    let mut rooms = house.as_object().into_iter().flatten();
    let window = rooms.find(|(_, things)| things["window"] == "open");
    let window = window.map(|(name, _)| format!("the {name} window is open"));
    let set = (thing, to.as_str().unwrap_or_default());
    if entry["alarm"] == "armed" && [("door", "open"), ("window", "open"), ("lock", "unlocked")].contains(&set) {
        return Some("the alarm is armed".to_owned());
    }
    if set == ("lock", "locked") && entry["door"] == "open" {
        return Some("the door is open".to_owned());
    }
    if (room, set) == ("entry", ("door", "open")) && entry["lock"] == "locked" {
        return Some("the door is locked".to_owned());
    }
    if (room, set) == ("garage", ("door", "closed")) && garage["doorway"] == "blocked" {
        return Some("something is in the doorway".to_owned());
    }
    if set == ("window", "open") && thermostat["mode"] != "off" {
        return Some(format!("the thermostat is on {}", py::str(&thermostat["mode"])));
    }
    if thing == "mode" && to != "off" {
        return window; // no heating or cooling out of an open window
    }
    if set == ("alarm", "armed") {
        let unsafe_to = [
            ("the front door is not locked", entry["lock"] == "unlocked"),
            ("the garage door is open", garage["door"] == "open"),
            ("the oven is on", house["kitchen"]["oven"] == "on"),
        ];
        let reason = unsafe_to.iter().find(|(_, so)| *so).map(|(reason, _)| reason.to_string());
        return reason.or(window);
    }
    None
}

/// Set one thing in one room to one state, if the house allows it.
fn switch(house: &Shared<Json>, room: &str, thing: &str, to: &Json) -> String {
    let done = house.update(|house| {
        let (why, state) = (refused(house, room, thing, to), py::str(to));
        if house[room][thing] == *to {
            format!("{room} {thing}: already {state}")
        } else if let Some(why) = why {
            format!("{room} {thing}: refused while {why}") // the house pushes back, and the next request sees it
        } else {
            house[room][thing] = to.clone();
            match room {
                // it takes time to get there
                "thermostat" => format!(
                    "{room} {thing}: now {state}, and the house is heading for {} degrees",
                    heading(house)
                ),
                _ => format!("{room} {thing}: now {state}"),
            }
        }
    });
    common::show_house(house, &done); // redraws the house on the page
    done
}

/// How warm the house will get: its target, as far as heating or cooling takes it there, or else the weather.
fn heading(house: &Json) -> i64 {
    let thermostat = &house["thermostat"];
    let degrees = |key: &str| thermostat[key].as_i64().unwrap_or_default();
    let (target, outside) = (degrees("target"), degrees("outside"));
    match thermostat["mode"].as_str() {
        Some("heat") => target.max(outside),
        Some("cool") => target.min(outside),
        _ => outside,
    }
}

/// Time passes, and the house gets one degree nearer to where it is heading.
fn tick(house: &mut Json) {
    let goal = heading(house);
    let temp = house["thermostat"]["temp"].as_i64().unwrap_or(goal);
    house["thermostat"]["temp"] = (temp + (goal - temp).signum()).into();
}

/// The single steps of each room, by the room and then by the step's name.
type Steps = IndexMap<String, IndexMap<String, Tool>>;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the page sets: the request, the house as it is now, and the routines it knows. And,
    // here only, the requests that follow it, each put to the house as the one before left it.
    let form = Form::new(
        "home",
        "The house",
        "Ask it for something. It keeps its state from one request to the next.",
    )
    .text("task", "Tell the house what to do", &PRESETS)
    .lines(
        "then",
        "Then more requests in turn, each to the house as the last left it: a text, or the number of one above",
        &[],
    )
    .house(json!({
        "entry": {"lights": "on", "door": "open", "lock": "unlocked", "alarm": "off"},
        "living room": {"lights": "on", "tv": "off", "fan": "on", "window": "closed"},
        "kitchen": {"lights": "on", "oven": "on", "coffee maker": "off", "dishwasher": "on"},
        "bedroom": {"lights": "off", "lamp": "on", "window": "open"},
        "bathroom": {"lights": "off", "fan": "off"},
        "garage": {"lights": "on", "door": "open", "doorway": "clear", "washer": "off", "dryer": "on"},
        "thermostat": {"mode": "off", "target": 22, "temp": 28, "outside": 28}
    }))
    .json(
        "routines",
        "The routines it knows",
        json!({
            "good night": {"about": "Going to bed: the rest of the house off, the doors shut, and the front door locked.", "steps": ["living room: tv off", "living room: fan off", "living room: lights off", "kitchen: oven off", "kitchen: coffee maker off", "kitchen: dishwasher off", "kitchen: lights off", "bathroom: fan off", "bathroom: lights off", "garage: washer off", "garage: dryer off", "garage: lights off", "garage: door closed", "entry: lights off", "entry: door closed", "entry: lock locked"]},
            "good morning": {"about": "Getting up: the front door unlocked, the kitchen lights and the coffee on.", "steps": ["entry: alarm off", "entry: lock unlocked", "kitchen: lights on", "kitchen: coffee maker on"]},
            "movie time": {"about": "Watching a film: the living room dimmed and the film on.", "steps": ["living room: tv on", "living room: lights off"]},
            "leaving": {"about": "Going out and leaving the house empty: everything off and shut, locked, and the alarm armed.", "steps": ["good night", "bedroom: lamp off", "bedroom: lights off", "bedroom: window closed", "living room: window closed", "thermostat: mode off", "entry: alarm armed"]}
        }),
    )
    .read();
    let preset = |task: &String| match task.parse::<usize>() {
        Ok(number) if (1..=PRESETS.len()).contains(&number) => PRESETS[number - 1].to_owned(),
        _ => task.clone(),
    };
    let tasks: Vec<String> = std::iter::once(form.text_of("task"))
        .chain(form.lines_of("then").iter().map(preset))
        .collect();
    let house = Shared::new(form.json_of("house"));
    let routines = Arc::new(form.json_of("routines"));
    let rooms = house.get().as_object().cloned().unwrap_or_default(); // the rooms, and the things in each, as the house starts

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let inside = operate
        .clone()
        .with_instructions("Operate one room of a smart home, one step at a time.")
        .input_described("before", "What was done in the house just before coming to this room.");

    // The single steps of each room: every state of every thing in it, each a name and a description.
    let single = |room: &String, thing: &String, to: Json| {
        let (name, house) = (format!("{thing} {}", py::str(&to)), house.clone());
        let description = format!("Set the {room} {thing} to {}.", py::str(&to));
        let (room, thing) = (room.clone(), thing.clone());
        let step = Tool::new(name.clone(), description, move |_call| Ok(switch(&house, &room, &thing, &to)));
        (name, step)
    };
    let things = |room: &Json| {
        room.as_object()
            .into_iter()
            .flatten()
            .map(|(thing, _)| thing.clone())
            .collect::<Vec<_>>()
    };
    let steps: Steps = rooms
        .iter()
        .map(|(room, inside)| {
            let each = things(inside)
                .into_iter()
                .flat_map(|thing| states(&thing).into_iter().map(move |to| (thing.clone(), to)));
            (room.clone(), each.map(|(thing, to)| single(room, &thing, to)).collect())
        })
        .collect();
    let steps = Arc::new(steps);

    // A room is one name on the house's map, and a map of its own: pointing at it asks again, inside it.
    let room = |name: &String| {
        let inside = Rdm::new(inside.clone())
            .tools_from(steps[name].values().cloned())
            .model(decision_model.clone());
        let nothing = format!("{name}: nothing was done");
        let enter = move |call: &Call| {
            let (task, observation): (String, String) = (call.arg("task")?, call.arg("observation")?);
            // One level down: the same request, and what led here.
            let done = call
                .on(&inside)
                .sub(inputs! { "task" => task, "before" => observation })?
                .steps("next");
            let mut results: Vec<String> = Vec::new();
            for (_, result) in done.iter().filter(|(step, _)| step != "finish") {
                if !results.contains(&py::str(result)) {
                    results.push(py::str(result));
                }
            }
            Ok(if results.is_empty() { nothing.clone() } else { results.join("; ") })
        };
        let things: Vec<String> = things(&rooms[name]).into_iter().filter(|thing| !states(thing).is_empty()).collect();
        Tool::new(name, format!("The {name}: {}.", things.join(", ")), enter)
    };

    /// A routine is several steps under one name. A step is "room: thing state", or another routine.
    fn routine(name: &str, routines: &Json, steps: &Steps, call: &Call) -> rdm::Result<String> {
        let mut done = Vec::new();
        for step in routines[name]["steps"].as_array().into_iter().flatten() {
            let step = step.as_str().unwrap_or_default();
            if routines.get(step).is_some() {
                done.push(routine(step, routines, steps, call)?);
                continue;
            }
            let single = step.split_once(": ").and_then(|(room, step)| steps.get(room)?.get(step));
            let single = single.ok_or_else(|| rdm::Error::tool(format!("The house has no step {step:?}.")))?;
            done.push(py::str(&single.call(call)?));
        }
        Ok(format!("{name}: done ({})", done.join("; ")))
    }

    // The house's map: its rooms, then its routines. The page adds a routine when it is asked to keep one.
    let known = |(name, each): (&String, &Json)| {
        let (name, routines, steps) = (name.clone(), routines.clone(), steps.clone());
        Tool::new(name.clone(), py::str(&each["about"]), move |call| {
            routine(&name, &routines, &steps, call)
        })
    };
    let known = routines.as_object().into_iter().flatten().map(known);
    let home = Rdm::new(operate)
        .tools_from(rooms.keys().map(room).chain(known))
        .model(decision_model.clone());

    for (at, task) in tasks.iter().enumerate() {
        if tasks.len() > 1 {
            // What the page shows in its form: which request this is.
            if at > 0 {
                py::print!();
            }
            py::print!(task);
        }
        common::show_house(&house, task);
        // The house is the run's state: it is there, and no request shows it. A state here is
        // plain data, so this is the house as the request finds it: the tools hold the house itself.
        let result = home.call_with(Ask::new(inputs! { "task" => task }).state(json!({"house": house.get()})))?;
        common::trace(&result);
        for (name, done) in result.steps("next") {
            if done.is_null() || done == "" {
                py::print!(name);
            } else {
                py::print!(format!("{name}: {}", py::str(&done)));
            }
        }

        // The request is done, and time passes until the house is as warm as it is going to get.
        while house.update(|house| house["thermostat"]["temp"].as_i64().is_some_and(|temp| temp != heading(house))) {
            let temp = house.update(|house| {
                tick(house);
                py::str(&house["thermostat"]["temp"])
            });
            common::show_house(&house, &format!("time passes: {temp} degrees inside"));
        }
    }
    Ok(())
}

/// A routine the house knows, by a word of the request.
const ROUTINES: [(&str, &str); 6] = [
    ("bed", "good night"),
    ("morning", "good morning"),
    ("weekend", "leaving"),
    ("leaving", "leaving"),
    ("film", "movie time"),
    ("movie", "movie time"),
];

/// One thing to set, by the words of the request: the room it is in, and the step there.
const WANTS: [(&[&str], &str, &str); 8] = [
    (&["alarm"], "entry", "alarm armed"),
    (&["hot"], "thermostat", "mode cool"),
    (&["cold"], "thermostat", "mode heat"),
    (&["garage"], "garage", "door closed"),
    (&["oven"], "kitchen", "oven off"),
    (&["tv", "off"], "living room", "tv off"),
    (&["tv"], "living room", "tv on"),
    (&["lock"], "entry", "lock locked"),
];

/// What the house says is in the way, and the step that puts it right: the room, and the step there.
const FIXES: [(&str, &str, &str); 4] = [
    ("refused while the front door is not locked", "entry", "lock locked"),
    ("refused while the door is open", "entry", "door closed"),
    ("refused while the garage door is open", "garage", "door closed"),
    ("refused while the oven is on", "kitchen", "oven off"),
];

/// The stand-in decision model, which answers when the environment asks for no real one: how
/// the site's own mock runs the house (`runsHouse` in its `test/notebook.mjs`), with a word for
/// each of the page's requests.
///
/// It is shown no house, as a model is not: it reads what the last step did. On the house's map
/// it points at a routine where the request is one, once, or at the room of the one thing the
/// request asks for. In a room it points at that thing's step. A refusal names what is in the
/// way, and it goes to put that right, in whichever room that is, and comes back. It stops when
/// the step is done, or was refused for a reason it knows no cure for.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    let text = |name: &str| inputs[name].as_str().unwrap_or_default();
    let (task, last) = (
        text("task").to_lowercase(),
        text("observation").split("; ").last().unwrap_or_default(),
    );
    let offered = question.labels();
    let words: Vec<&str> = task
        .split(|letter: char| !letter.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    let wanted = WANTS.iter().find(|(needs, _, _)| needs.iter().all(|word| words.contains(word)));
    let mut want = wanted.map(|(_, room, step)| (*room, step.to_string()));
    if let (None, Some(at @ 1..)) = (&want, words.iter().position(|word| *word == "degrees")) {
        want = Some(("thermostat", format!("target {}", words[at - 1])));
    }
    // The room to go to and the step to take there, for what the house said is in the way.
    let fixing = |said: &str, rooms: &[String]| {
        let fix = FIXES.iter().find(|(refusal, _, _)| said.contains(refusal));
        let window = || {
            rooms
                .iter()
                .find(|room| said.contains(&format!("refused while the {room} window is open")))
        };
        fix.map(|(_, room, step)| (room.to_string(), *step))
            .or_else(|| window().map(|room| (room.clone(), "window closed")))
    };
    let done = |said: &str| {
        let set = want.as_ref().and_then(|(_, step)| step.rsplit_once(' '));
        set.is_some_and(|(thing, to)| said.contains(&format!("{thing}: now {to}")) || said.contains(&format!("{thing}: already {to}")))
    };
    let on_offer = |name: &str| offered.iter().any(|offered| offered == name);

    if inputs.get("before").is_none() {
        // The house's map: a routine where the request is one, or the room to go to next.
        if let Some((_, routine)) = ROUTINES.iter().find(|(word, routine)| words.contains(word) && on_offer(routine)) {
            return if last.is_empty() { *routine } else { "finish" }.into();
        }
        if let Some((room, _)) = fixing(last, &offered).filter(|(room, _)| on_offer(room)) {
            return room.into();
        }
        return match &want {
            Some((room, _)) if !done(last) && !last.contains("refused") && on_offer(room) => (*room).into(),
            _ => "finish".into(),
        };
    }
    // A room's map. Its descriptions say which room it is: "Set the entry lights to on."
    let first = question
        .criteria
        .as_ref()
        .and_then(|criteria| criteria[offered[0].as_str()].as_str());
    let set = first.and_then(|first| first.strip_prefix("Set the ")?.rsplit_once(" to "));
    let room = set
        .and_then(|(set, _)| set.rsplit_once(' '))
        .map_or("", |(room, _)| room)
        .to_owned();
    let led = if last.is_empty() {
        text("before").split("; ").last().unwrap_or_default()
    } else {
        last
    };
    match (fixing(led, std::slice::from_ref(&room)), &want) {
        (Some((to, step)), _) if to == room && on_offer(step) => step.into(),
        (_, Some((to, step))) if *to == room && !last.contains("refused") && !done(last) && on_offer(step) => step.into(),
        _ => "finish".into(),
    }
}
