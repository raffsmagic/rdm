//! What goes out: every request of the notebook's cells, set beside what DSPy sends for the same cell.
//!
//! The files in `fixtures/` are the requests the Python module made for each cell of the
//! notebook, run on a stand-in decision model and written a line each as `{"state",
//! "questions"}`. Each cell is ported here as it would be written in Rust, on a stand-in that
//! answers the same, and every request must be the same text: the same keys in the same order,
//! the same instructions, the same options. A model shown anything else is another model's input.

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use common::client;
use indexmap::IndexMap;
use rdm::systemone::Scripted;
use rdm::{Ask, Json, Output, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;

/// What a stand-in was sent, a line for each request, as the fixtures are written.
fn sent(model: &Scripted) -> Vec<String> {
    let requests = model.requests();
    let written = requests
        .iter()
        .map(|request| json!({"state": request.state, "questions": request.questions}).to_string());
    written.collect()
}

/// The requests of a fixture, each as text with nothing between its parts: the order of the keys is kept.
fn captured(lines: &str) -> Vec<String> {
    let requests = lines.lines().filter(|line| !line.trim().is_empty());
    requests
        .map(|line| serde_json::from_str::<Json>(line).expect("a fixture is JSON").to_string())
        .collect()
}

type House = Shared<IndexMap<String, String>>;

fn house(devices: &[(&str, &str)]) -> House {
    Shared::new(
        devices
            .iter()
            .map(|(device, state)| (device.to_string(), state.to_string()))
            .collect(),
    )
}

/// The house pushes back: an open door cannot be locked.
fn switch(house: &House, device: &str, to: &str) -> String {
    let mut house = house.lock();
    if device == "lock" && to == "locked" && house["door"] == "open" {
        return "lock: refused, the door is open".to_owned();
    }
    house.insert(device.to_owned(), to.to_owned());
    format!("{device}: now {to}")
}

#[test]
fn the_act_cell_sends_what_dspy_sends() {
    let house = house(&[("tv", "off"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
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
    let home = house.clone();
    let set = Tool::new("switch", "Set one device to one state.", move |call| {
        let (device, to): (String, String) = (call.arg("device")?, call.arg("to")?);
        Ok(switch(&home, &device, &to))
    });
    let model = client(|_, name, _| match name {
        "switch" => common::Say::Yes(true),
        "device" => "tv".into(),
        _ => "on".into(),
    });
    let rdm = Rdm::new(act).tool(set).model(model.clone());
    let result = rdm.call(inputs! { "task" => "Turn the TV on.", "house" => &house }).unwrap();
    assert_eq!(json!(result.observations), json!({"switch": "tv: now on"}));
    assert_eq!(sent(&model), captured(include_str!("fixtures/act.jsonl")));
}

#[test]
fn the_triage_cell_sends_what_dspy_sends() {
    let triage = Signature::new("Triage a message from a customer.")
        .input("message")
        .noul("refund", "Is the customer in `inputs.message` asking for a refund?")
        .choice(
            "team",
            "Which team should handle `inputs.message`?",
            ["billing", "technical support", "sales", "account management"],
        )
        .noul("upset", "Is the customer in `inputs.message` upset?");
    let refund = Tool::new("refund", "Open a refund with a team.", |call| {
        let team: String = call.arg("team")?;
        Ok(format!("refund opened with {team}"))
    });
    let model = client(|_, name, _| if name == "team" { "billing".into() } else { common::Say::Yes(true) });
    let rdm = Rdm::new(triage).tool(refund).model(model.clone());
    let message = "I was charged twice for the same order and nobody answers my emails. I want my money back now.";
    let result = rdm.call(inputs! { "message" => message }).unwrap();
    assert_eq!(json!(result.observations), json!({"refund": "refund opened with billing"}));
    assert_eq!(sent(&model), captured(include_str!("fixtures/triage.jsonl")));
}

#[test]
fn the_map_cell_sends_what_dspy_sends_at_every_turn_of_three_runs() {
    let house = house(&[("tv", "on"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let start = house.get();
    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");
    let lines = [
        ("tv", "on"),
        ("tv", "off"),
        ("lights", "on"),
        ("lights", "off"),
        ("door", "closed"),
        ("lock", "locked"),
    ];
    let steps = lines.map(|(device, to)| {
        let house = house.clone();
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
            Ok(switch(&house, device, to))
        })
    });
    // What the stand-in pointed at when the cell was captured, in turn.
    let script = ["lock locked", "door closed", "lock locked", "finish"]
        .into_iter()
        .chain(["tv off", "lights off", "door closed", "lock locked", "finish"])
        .chain(["good night", "finish"]);
    let script = Mutex::new(script.map(str::to_owned).collect::<VecDeque<_>>());
    let model = client(move |_, _, _| script.lock().unwrap().pop_front().expect("the script has run out"));
    let rdm = Rdm::new(operate).tools_from(steps).model(model.clone());

    let (first, routine, again) = ("Lock the front door.", "Good night: everything off, and lock up.", "Night!");
    let result = rdm.call(inputs! { "task" => first, "house" => &house }).unwrap();
    assert_eq!(result.pointed("next"), ["lock locked", "door closed", "lock locked", "finish"]);

    house.set(start.clone());
    let result = rdm.call(inputs! { "task" => routine, "house" => &house }).unwrap();
    let done: Vec<String> = result
        .steps("next")
        .into_iter()
        .filter(|(name, observed)| name != "finish" && !observed.as_str().unwrap_or_default().contains("refused"))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(done, ["tv off", "lights off", "door closed", "lock locked"]);

    // What was done is kept as one more name on the map, and the next run is offered it.
    let kept = done.clone();
    let good_night = Tool::new("good night", format!("{routine} ({}).", done.join(", ")), move |call| {
        kept.iter().map(|name| call.run(name)).collect::<rdm::Result<Vec<_>>>()
    });
    rdm.tools().insert(good_night);

    house.set(start);
    let result = rdm.call(inputs! { "task" => again, "house" => &house }).unwrap();
    assert_eq!(result.pointed("next"), ["good night", "finish"]);
    assert_eq!(
        json!(house.get()),
        json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"})
    );
    let (ours, theirs) = (sent(&model), captured(include_str!("fixtures/map.jsonl")));
    assert_eq!(ours.len(), 11);
    for (turn, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(ours, theirs, "request {turn}");
    }
    assert_eq!(ours.len(), theirs.len());
}

fn taxonomy(category: &str) -> Option<Vec<&'static str>> {
    Some(match category {
        "Products" => vec!["Animals & Pet Supplies", "Furniture", "Kitchen"],
        "Animals & Pet Supplies" => vec!["Cat Supplies", "Dog Supplies"],
        "Cat Supplies" => vec!["Cat Beds", "Cat Toys", "Cat Window Perches"],
        "Dog Supplies" => vec!["Dog Beds", "Dog Toys", "Dog Leashes"],
        "Furniture" => vec!["Shelves", "Beds", "Sofas"],
        "Kitchen" => vec!["Cookware", "Tableware", "Kitchen Knives"],
        _ => return None,
    })
}

/// As the cell writes it: the string form, then the output given its options and its question.
fn under(category: &str) -> Signature {
    Signature::parse("listing -> category").unwrap().with_output(
        "category",
        Output::choice(
            "Which category best matches the product in `inputs.listing`?",
            taxonomy(category).unwrap_or_default(),
        ),
    )
}

/// The cell's tool: puts the question again under the category chosen, down to one that has nothing under it.
fn descend() -> Tool {
    Tool::new("category", "Puts the question under the category chosen.", |call| {
        let (category, listing): (String, String) = (call.arg("category")?, call.arg("listing")?);
        if taxonomy(&category).is_none() {
            return Ok(json!(category));
        }
        let below = call.sub_with(Ask::new(inputs! { "listing" => listing }).signature(under(&category)))?;
        // As the cell reads it, and as a port would: there is nothing under this key at a leaf.
        Ok(below.observations["category"].clone())
    })
}

/// A stand-in that chooses its way down to the window perches.
fn perches() -> Arc<Scripted> {
    let chosen = ["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"];
    client(move |_, _, question| {
        let labels = question.labels();
        labels
            .into_iter()
            .find(|label| chosen.contains(&label.as_str()))
            .expect("one of them is on offer")
    })
}

const LISTING: &str = "A padded shelf that suction-cups to a window, so a cat can sun itself.";

#[test]
fn the_choose_cell_sends_what_dspy_sends_instructions_of_the_string_form_included() {
    let model = perches();
    let classify = Rdm::new(under("Products")).tool(descend()).model(model.clone());
    let result = classify.call(inputs! { "listing" => LISTING }).unwrap();
    assert_eq!(result.chosen("category"), Some("Animals & Pet Supplies"));
    assert_eq!(result.observations["category"], "Cat Window Perches");
    assert_eq!(sent(&model), captured(include_str!("fixtures/choose.jsonl")));
    // The instructions are DSPy's for a signature written as a string, and are sent at every level.
    let instructions = "Given the fields `listing`, produce the fields `category`.";
    assert!(model.requests().iter().all(|request| request.state["instructions"] == instructions));
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn the_choose_cell_at_the_depth_limit_ends_as_it_does_in_python() {
    // One level short of the leaves: the last request is answered and its tool is not run, so
    // the tool above it reads an observation that is not there. In Python that is a KeyError,
    // and the failure is the tool's observation. Here it is a panic, and it is the same.
    let model = perches();
    let classify = Rdm::new(under("Products")).tool(descend()).max_depth(2).model(model.clone());
    let result = classify.call(inputs! { "listing" => LISTING }).unwrap();
    assert_eq!((result.termination_reason, result.trajectory.len()), (rdm::Reason::MaxDepth, 3));
    assert_eq!(model.requests().len(), 3);
    let seen = result.observations["category"].as_str().unwrap();
    assert!(seen.starts_with("Execution error in category: "), "{seen}");
}

/// `{heading: [the headings and passages directly under it]}`, made by plain code.
fn outline(markdown: &str, title: &str) -> IndexMap<String, Vec<String>> {
    let mut under: IndexMap<String, Vec<String>> = IndexMap::from([(title.to_owned(), Vec::new())]);
    let mut path = vec![title.to_owned()];
    for block in markdown.trim().split("\n\n") {
        let depth = block.len() - block.trim_start_matches('#').len();
        let mut block = block.to_owned();
        if depth > 0 {
            block = block.trim_start_matches(['#', ' ']).to_owned();
            path.truncate(depth);
            path.push(block.clone());
            under.insert(block.clone(), Vec::new());
        }
        let parent = if depth > 0 { &path[path.len() - 2] } else { &path[path.len() - 1] };
        let parent = parent.clone();
        under.entry(parent).or_default().push(block);
    }
    under
}

/// Every heading below an entry: all that a request shows of what is under it.
fn headings(under: &IndexMap<String, Vec<String>>, entry: &str) -> Vec<String> {
    let mut below = Vec::new();
    for each in under.get(entry).into_iter().flatten().filter(|each| under.contains_key(*each)) {
        below.push(each.clone());
        below.extend(headings(under, each));
    }
    below
}

#[test]
fn the_outline_cell_sends_what_dspy_sends_and_never_the_handbook() {
    let handbook = include_str!("fixtures/handbook.md");
    let under = outline(handbook, "Handbook");
    let navigate = Signature::new("Search a handbook for the answer to a question, one entry at a time.")
        .input_described("entry", "A heading of the handbook, or a passage under one.")
        .input_with("contains", "list[str]", "The headings under `inputs.entry`. Empty for a passage.")
        .input("question")
        .noul(
            "explore",
            "Does `inputs.entry` state the answer to `inputs.question`, or does `inputs.entry` or any heading listed in `inputs.contains` name the topic that `inputs.question` asks about?",
        );
    let explore = Tool::new("explore", "Asks the same question about each entry under this one.", |call| {
        let (entry, question): (String, String) = (call.arg("entry")?, call.arg("question")?);
        // What the run was given as state: no request shows it.
        let outline: IndexMap<String, Vec<String>> = serde_json::from_value(call.state()["outline"].clone())?;
        let Some(entries) = outline.get(&entry) else {
            return Ok(vec![entry]); // a leaf: a passage, and the model said yes to it
        };
        let mut found = Vec::new();
        for each in entries {
            let below = call.sub(inputs! { "entry" => each, "contains" => headings(&outline, each), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("explore").unwrap_or_default());
        }
        Ok(found)
    });
    let model = client(|inputs, _, _| {
        let entry = inputs["entry"].as_str().unwrap_or_default().to_lowercase();
        entry == "handbook" || entry.contains("remote") || entry.contains("abroad") || entry.contains("another country")
    });
    let walk = Rdm::new(navigate).tool(explore).max_depth(10).model(model.clone());
    let question = "Can I work from another country?";
    let ask = Ask::new(inputs! { "entry" => "Handbook", "contains" => headings(&under, "Handbook"), "question" => question })
        .state(json!({ "outline": under }));
    let result = walk.call_with(ask).unwrap();
    let found = result.observed::<Vec<String>>("explore").unwrap();
    assert!(found.len() == 1 && found[0].starts_with("Working from another country is limited to 20 working days"));
    let (ours, theirs) = (sent(&model), captured(include_str!("fixtures/outline.jsonl")));
    for (at, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(ours, theirs, "request {at}");
    }
    assert_eq!((ours.len(), theirs.len()), (9, 9));
}

#[test]
fn the_bodies_the_reports_captured_from_the_wheel_are_sent_as_they_are() {
    // Report 06, section 6.2: the halving search, a single yes or no.
    let search = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");
    let model = client(|_, _, _| false);
    let rdm = Rdm::new(search).model(model.clone());
    rdm.call(inputs! { "text" => "line one\nline two", "question" => "Q?" }).unwrap();
    let body = r#"{"state": {"instructions": "Search a text for the answer to a question.",
           "input_fields": "1. `text` (str): \n2. `question` (str):",
           "inputs": {"text": "line one\nline two", "question": "Q?"}},
 "questions": {"look": {"instructions": "Is the answer to `inputs.question` stated in `inputs.text`?", "type": "noul"}}}"#;
    assert_eq!(sent(&model), [serde_json::from_str::<Json>(body).unwrap().to_string()]);

    // A pointer, the first turn and the second of the same call.
    let house = house(&[("tv", "off"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");
    let steps = [("tv", "on"), ("door", "closed"), ("lock", "locked")].map(|(device, to)| {
        let house = house.clone();
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
            Ok(switch(&house, device, to))
        })
    });
    let model = client(|inputs, _, _| if inputs["observation"] == "" { "tv on" } else { "finish" });
    let rdm = Rdm::new(operate).tools_from(steps).model(model.clone());
    rdm.call(inputs! { "task" => "Turn the TV on.", "house" => &house }).unwrap();
    let turn = |tv: &str, observation: &str| {
        json!({
            "state": {
                "instructions": "Operate a smart home, one step at a time.",
                "input_fields": "1. `task` (str): \n2. `house` (dict): \n3. `observation` (str): What the last step did.",
                "inputs": {
                    "task": "Turn the TV on.",
                    "house": {"tv": tv, "lights": "on", "door": "open", "lock": "unlocked"},
                    "observation": observation,
                },
            },
            "questions": {"next": {
                "instructions": "What should happen next for `inputs.task`?",
                "type": "choice",
                "criteria": {
                    "tv on": "Set the tv to on.",
                    "door closed": "Set the door to closed.",
                    "lock locked": "Set the lock to locked.",
                    "finish": "Stop: nothing more should be done.",
                },
            }},
        })
        .to_string()
    };
    assert_eq!(sent(&model), [turn("off", ""), turn("on", "tv: now on")]);

    // A signature written as a string, with a choice whose options have no descriptions.
    let model = client(|_, _, _| "A > B");
    let ask = Signature::parse("listing -> category").unwrap().with_output(
        "category",
        Output::choice("Which category best matches the product in `inputs.listing`?", ["A > B", "A > C"]),
    );
    Rdm::new(ask).model(model.clone()).call(inputs! { "listing" => "x" }).unwrap();
    let body = r#"{"state": {"instructions": "Given the fields `listing`, produce the fields `category`.",
           "input_fields": "1. `listing` (str):",
           "inputs": {"listing": "x"}},
 "questions": {"category": {"instructions": "Which category best matches the product in `inputs.listing`?", "type": "choice",
                            "criteria": {"A > B": null, "A > C": null}}}}"#;
    assert_eq!(sent(&model), [serde_json::from_str::<Json>(body).unwrap().to_string()]);
}

#[test]
fn a_signature_written_as_a_string_has_the_instructions_dspy_gives_it() {
    // dspy/signatures/signature.py: every field, ticked, in the order written.
    let written = Signature::parse("task, house: dict -> switch, device: Literal['tv', 'lights']").unwrap();
    assert_eq!(
        written.instructions(),
        "Given the fields `task`, `house`, produce the fields `switch`, `device`."
    );
    // They are the signature's from then on, whatever is put on it or taken off it.
    let edited = written
        .with_output("to", Output::choice("Which state?", ["on", "off"]))
        .without("switch");
    assert_eq!(
        edited.instructions(),
        "Given the fields `task`, `house`, produce the fields `switch`, `device`."
    );
    // A signature that is built is given its instructions, and may be given none.
    assert_eq!(Signature::new("").input("task").noul("switch", "Switch?").instructions(), "");
}
