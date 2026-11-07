//! The shapes of the notebook examples, on a stand-in model: what each needs of the core is here.

mod common;

use common::{Say, client, offered, shown};
use indexmap::IndexMap;
use rdm::systemone::{Answer, Scripted};
use rdm::{Ask, Json, Rdm, Reason, Shared, Signature, Tool, inputs};
use serde_json::json;

type House = Shared<IndexMap<String, String>>;

fn house(devices: &[(&str, &str)]) -> House {
    Shared::new(
        devices
            .iter()
            .map(|(device, state)| (device.to_string(), state.to_string()))
            .collect(),
    )
}

/// Set one device to one state. The house pushes back: an open door cannot be locked.
fn switch(house: &House, device: &str, to: &str) -> String {
    let mut house = house.lock();
    if device == "lock" && to == "locked" && house["door"] == "open" {
        return "lock: refused, the door is open".to_owned();
    }
    house.insert(device.to_owned(), to.to_owned());
    format!("{device}: now {to}")
}

fn act(house: &House, model: impl rdm::DecisionModel + 'static) -> Rdm {
    let signature = Signature::new("Operate a smart home.")
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
    let house = house.clone();
    let switch = Tool::new("switch", "Set one device to one state.", move |call| {
        let (device, to): (String, String) = (call.arg("device")?, call.arg("to")?);
        Ok(switch(&house, &device, &to))
    });
    Rdm::new(signature).tool(switch).model(model)
}

/// Reads a task as the examples' decision model would: by its words.
fn reads(inputs: &Json, name: &str) -> Say {
    let task = inputs["task"].as_str().unwrap().to_lowercase();
    let device = ["tv", "lights", "door", "lock"].into_iter().find(|device| task.contains(device));
    match name {
        "split" => Say::Yes(task.contains(", ")),
        "act" | "switch" => Say::Yes(device.is_some() && !task.contains(", ")),
        "chat" => Say::Yes(task.ends_with('?') && !task.contains(", ")),
        "device" => Say::Pick(device.unwrap_or("tv").to_owned()),
        _ => Say::Pick(if task.contains("off") { "off" } else { "on" }.to_owned()),
    }
}

#[test]
fn act_one_request_and_what_its_answers_run() {
    let house = house(&[("tv", "off"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let model = client(|inputs, name, _| reads(inputs, name));
    let act = act(&house, model.clone());
    let result = act.call(inputs! { "task" => "Turn the TV on.", "house" => &house }).unwrap();
    assert_eq!(json!(result.observations), json!({"switch": "tv: now on"}));
    assert_eq!(
        (house.get()["tv"].as_str(), model.requests().len(), result.trajectory.len()),
        ("on", 1, 1)
    );
}

#[test]
fn route_a_writer_splits_and_each_part_is_routed_one_level_down() {
    let house = house(&[("tv", "off"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let model = client(|inputs, name, _| reads(inputs, name));
    let act = act(&house, model.clone());
    let route = Signature::new("Route a request to a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul("split", "Does `inputs.task` ask for more than one distinct thing?")
        .noul("act", "Is `inputs.task` one thing to do to the house?")
        .noul("chat", "Is `inputs.task` only a question to answer in conversation?");
    // Each part the language model writes is a request of its own, routed one level down.
    let split = Tool::new("split", "", |call| {
        let task: String = call.arg("task")?;
        call.spend(40); // what the writer used
        let parts = task
            .trim_end_matches('.')
            .split(", ")
            .map(|part| call.sub(inputs! { "task" => part, "house" => call.input("house")? }));
        Ok(parts
            .collect::<rdm::Result<Vec<_>>>()?
            .iter()
            .map(|part| part.termination_reason)
            .collect::<Vec<_>>())
    });
    let reply = Tool::new("chat", "", |_| Ok("It will rain."));
    // A module is a tool: `act` is another RDM, and joins the run as a sub-call.
    let route = Rdm::new(route)
        .tool(split)
        .tool(Tool::module("act", "", &act))
        .tool(reply)
        .model(model.clone());
    let task = "Turn the TV on, switch the lights off, will it rain?";
    let result = route.call(inputs! { "task" => task, "house" => &house }).unwrap();
    assert_eq!(
        json!(house.get()),
        json!({"tv": "on", "lights": "off", "door": "open", "lock": "unlocked"})
    );
    assert_eq!(result.termination_reason, Reason::Final);
    let run: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.tool_name.as_deref(), entry.inputs["task"].as_str().unwrap()))
        .collect();
    assert_eq!(
        run,
        [
            (0, None, task),
            (1, Some("split"), "Turn the TV on"),
            (2, Some("act"), "Turn the TV on"),
            (1, Some("split"), "switch the lights off"),
            (2, Some("act"), "switch the lights off"),
            (1, Some("split"), "will it rain?"),
        ]
    );
    // The house is live all the way down: the second part's requests show what the first part did.
    assert_eq!(
        shown(&model).iter().map(|inputs| inputs["house"]["tv"].clone()).collect::<Vec<_>>(),
        ["off", "off", "off", "on", "on", "on"]
    );
    assert_eq!(
        (result.tokens(), json!(result.trajectory[0].tool_tokens)),
        (40, json!({"split": 40}))
    );
}

#[test]
fn map_what_was_done_is_kept_as_one_more_name() {
    let house = house(&[("tv", "on"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let start = house.get();
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
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
            Ok(switch(&house, device, to))
        })
    });
    // Points at whatever is not yet as the night wants it, and at the routine once it is on the map.
    let model = client(|inputs, _, question| {
        let (house, labels) = (&inputs["house"], question.labels());
        if labels.contains(&"good night".to_owned()) && inputs["observation"] == "" {
            return "good night";
        }
        if house["lock"] == "unlocked" && !inputs["observation"].as_str().unwrap().contains("refused") {
            return "lock locked";
        }
        [
            ("door", "open", "door closed"),
            ("lock", "unlocked", "lock locked"),
            ("tv", "on", "tv off"),
            ("lights", "on", "lights off"),
        ]
        .into_iter()
        .find(|(device, state, _)| house[*device] == *state)
        .map_or("finish", |(_, _, step)| step)
    });
    let rdm = Rdm::new(operate).tools_from(steps).model(model.clone());

    // Nothing here says to close the door before locking it: the house refuses, and the next request sees that.
    let result = rdm
        .call(inputs! { "task" => "Good night: everything off, and lock up.", "house" => &house })
        .unwrap();
    assert_eq!(
        result.pointed("next"),
        ["lock locked", "door closed", "lock locked", "tv off", "lights off", "finish"]
    );
    let done: Vec<String> = result
        .steps("next")
        .into_iter()
        .filter(|(name, seen)| name != "finish" && !seen.as_str().unwrap().contains("refused"))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(done, ["door closed", "lock locked", "tv off", "lights off"]);

    // The map changes: a tool put on the module's map is offered from the next run. It takes
    // the steps that worked, in the order they were taken, through the run's own map.
    let routine = done.clone();
    let good_night = Tool::new("good night", format!("Good night ({}).", done.join(", ")), move |call| {
        routine.iter().map(|step| call.run(step)).collect::<rdm::Result<Vec<_>>>()
    });
    rdm.tools().insert(good_night);
    house.set(start);
    let result = rdm.call(inputs! { "task" => "Night!", "house" => &house }).unwrap();
    assert_eq!(
        offered(&model.requests().last().unwrap().clone(), "next")
            .iter()
            .rev()
            .take(2)
            .collect::<Vec<_>>(),
        ["finish", "good night"]
    );
    assert_eq!(result.pointed("next"), ["good night", "finish"]);
    assert_eq!(
        result.steps("next")[0].1,
        json!(["door: now closed", "lock: now locked", "tv: now off", "lights: now off"])
    );
    assert_eq!(
        json!(house.get()),
        json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"})
    );
}

#[test]
fn home_a_room_is_a_name_on_the_map_and_a_map_of_its_own() {
    let rooms = Shared::new(json!({"kitchen": {"oven": "on", "lights": "on"}, "bedroom": {"lamp": "on"}}));
    let operate = || {
        Signature::new("Operate a smart home, one step at a time.")
            .input("task")
            .input_described("observation", "What the last step did.")
            .tool("next", "What should happen next for `inputs.task`?")
    };
    // A stand-in that knows the house, and points at the first name it is offered that has something
    // still on. A real model learns how the house is from what each step did: no request shows it.
    let knows = rooms.clone();
    let model = client(move |_, _, question| {
        let rooms = knows.get();
        let wanted = |label: &String| match label.strip_suffix(" off") {
            Some(thing) => rooms.as_object().unwrap().values().any(|room| room[thing] == "on"),
            None => rooms[label.as_str()]
                .as_object()
                .is_some_and(|things| things.values().any(|state| state == "on")),
        };
        question.labels().into_iter().find(wanted).unwrap_or_else(|| "finish".to_owned())
    });
    let room = |name: &'static str, things: &[&'static str]| {
        let inside = operate()
            .with_instructions("Operate one room of a smart home, one step at a time.")
            .input_described("before", "What was done in the house just before coming to this room.");
        let steps = things.iter().map(|thing| {
            let (rooms, thing) = (rooms.clone(), *thing);
            Tool::new(format!("{thing} off"), format!("Set the {name} {thing} to off."), move |call| {
                // The house is the run's state as well: it is there, and no request shows it.
                assert!(call.state()["house"].is_string());
                rooms.update(|rooms| rooms[name][thing] = json!("off"));
                Ok(format!("{name} {thing} off"))
            })
        });
        let inside = Rdm::new(inside).tools_from(steps).model(model.clone());
        // A room is one name on the house's map, and a map of its own: pointing at it asks again, inside it.
        Tool::new(name, format!("The {name}: {}.", things.join(", ")), move |call| {
            let (task, before): (String, String) = (call.arg("task")?, call.arg("observation")?);
            let done = call.on(&inside).sub(inputs! { "task" => task, "before" => before })?.steps("next");
            let said: Vec<String> = done
                .into_iter()
                .filter(|(step, _)| step != "finish")
                .filter_map(|(_, result)| result.as_str().map(str::to_owned))
                .collect();
            Ok(if said.is_empty() {
                format!("{name}: nothing was done")
            } else {
                said.join("; ")
            })
        })
    };
    let home = Rdm::new(operate())
        .tool(room("kitchen", &["oven", "lights"]))
        .tool(room("bedroom", &["lamp"]))
        .model(model.clone());
    let result = home
        .call_with(Ask::new(inputs! { "task" => "We are off to bed." }).state(json!({"house": "the house"})))
        .unwrap();
    assert_eq!(
        result.observations["next"],
        json!([
            ["kitchen", "kitchen oven off; kitchen lights off"],
            ["bedroom", "bedroom lamp off"],
            ["finish", null]
        ])
    );
    assert_eq!(
        rooms.get(),
        json!({"kitchen": {"oven": "off", "lights": "off"}, "bedroom": {"lamp": "off"}})
    );
    // The house's turns at depth 0, and each room's own loop one level down, under the room that was pointed at.
    let run: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.iteration, entry.tool_name.as_deref()))
        .collect();
    assert_eq!(
        run,
        [
            (0, 0, None),
            (1, 0, Some("kitchen")),
            (1, 1, Some("oven off")),
            (1, 2, Some("lights off")),
            (0, 1, Some("kitchen")),
            (1, 0, Some("bedroom")),
            (1, 1, Some("lamp off")),
            (0, 2, Some("bedroom")),
        ]
    );
    assert_eq!(result.termination_reason, Reason::Final);
}

#[test]
fn hierarchy_every_option_likely_enough_is_followed_and_the_leaves_are_asked_among() {
    fn taxonomy(category: &str) -> Vec<&'static str> {
        match category {
            "All products" => vec!["Luggage & Bags", "Electronics", "Furniture"],
            "Luggage & Bags" => vec!["Backpacks", "Suitcases"],
            "Electronics" => vec!["Laptop Bags", "Laptops"],
            _ => vec![],
        }
    }
    let ask = |options: Vec<String>| {
        Signature::parse("listing -> category").unwrap().with_output(
            "category",
            rdm::Output::choice("Which category best matches the product in `inputs.listing`?", options),
        )
    };
    let all = |category: &str| taxonomy(category).into_iter().map(str::to_owned).collect::<Vec<_>>();
    // Runs on every answer. Follows each category at least half as likely as the likeliest, by
    // asking the module again about the categories under it. Returns the leaves reached.
    let category = Tool::new("category", "", move |call| {
        let rated = &call
            .answer("category")
            .and_then(|answer| answer.as_choice())
            .ok_or("no choice was made")?
            .probabilities;
        let most = rated.values().copied().fold(0.0, f64::max);
        let mut leaves = Vec::new();
        for (each, _) in rated.iter().filter(|(_, probability)| **probability >= most / 2.0) {
            if all(each).is_empty() {
                leaves.push(each.clone());
                continue;
            }
            let below = call.sub_with(Ask::new(inputs! { "listing" => call.input("listing")? }).signature(ask(all(each))))?;
            leaves.extend(below.observed::<Vec<String>>("category").unwrap_or_default());
        }
        Ok(leaves)
    });
    // Rates bags and electronics alike at the top, and one option clearly under each.
    let model = Scripted::new(|request| {
        let labels = request.questions["category"].labels();
        let rate = |label: &String| match label.as_str() {
            "Luggage & Bags" | "Electronics" => 0.45,
            "Backpacks" | "Laptop Bags" => 0.9,
            _ => 0.1,
        };
        [("category", Answer::choice(labels.iter().map(|label| (label.clone(), rate(label)))))]
    });
    let search = Rdm::new(ask(all("All products")))
        .tool(category)
        .max_depth(12)
        .max_predictor_calls(60)
        .model(model);
    let result = search
        .call(inputs! { "listing" => "Laptop backpack with a padded sleeve, 20 litres." })
        .unwrap();
    let found: Vec<String> = result.observed("category").unwrap();
    assert_eq!(
        (found.clone(), result.trajectory.len()),
        (vec!["Backpacks".to_owned(), "Laptop Bags".to_owned()], 3)
    );
    // More than one leaf came back: ask once more, among those, for the answer only.
    let best = search
        .call_with(
            Ask::new(inputs! { "listing" => "Laptop backpack." })
                .signature(ask(found))
                .max_depth(0),
        )
        .unwrap();
    assert_eq!(
        (best.chosen("category"), best.observations.len(), best.termination_reason),
        (Some("Backpacks"), 0, Reason::Final)
    );
}
