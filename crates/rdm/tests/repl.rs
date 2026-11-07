//! Repl: a REPL where what a language model writes can be pointed at by a decision model.
//!
//! - One REPL: what is written stays, and can be pointed at
//! - A Repl in a program: saved, loaded, copied, and used from any thread
//! - Several REPLs: environments that are isolated, and routed to by name
//! - The functions of the host, and what written code is held to
//! - The notebook's cells: a starter map, and a REPL a writer writes into
#![cfg(feature = "repl")]

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{client, offered, scripted, shown};
use indexmap::IndexMap;
use rdm::repl::{Draft, Functions, Limits, Repl, Written};
use rdm::{Error, Json, Prediction, Rdm, Scripted, Shared, Signature, Source, Tool, as_observation, inputs};
use serde_json::{Map, json};

type World = Shared<IndexMap<String, i64>>;

/// A heater.
fn world() -> World {
    Shared::new(IndexMap::from([("heat".to_owned(), 21)]))
}

fn heat(world: &World) -> i64 {
    world.lock()["heat"]
}

/// A REPL with the two functions that exist for a heater inside it.
fn environment(world: &World) -> Repl {
    let (cooled, read) = (world.clone(), world.clone());
    Repl::new()
        .function("cooler", "Make it one degree cooler, once.", move |_| {
            let mut state = cooled.lock();
            state["heat"] -= 1;
            Ok(format!("heat is {}", state["heat"]))
        })
        .function("read", "The state, as JSON text.", move |_| {
            Ok(as_observation(&json!(*read.lock())))
        })
}

fn operate() -> Signature {
    Signature::new("Operate a heater, one step at a time.")
        .input("request")
        .input_typed("world", "dict")
        .input("observation")
        .tool("next", "What should happen next for `inputs.request`?")
}

const COOL_THREE: &str = "def cool_three():\n    for _ in range(3):\n        cooler()\n    return 'cooled by three: ' + read()";
const WRITTEN: &str = "`cool_three` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Make it three degrees cooler.";
const WRITE: &str = "Have a new tool written in code: nothing listed does what is still needed.";

fn wrote() -> Draft {
    Draft::new("cool_three", "Make it three degrees cooler.", COOL_THREE)
}

fn cool_three() -> Written {
    Written::new("Make it three degrees cooler.", COOL_THREE)
}

/// A tool that writes: something gives a function in one call, as a language model would, and
/// the tool defines it in the REPL. What it was shown each time is kept.
fn writer<const N: usize>(repl: &Repl, name: &str, drafts: [Draft; N]) -> (Tool, Shared<Vec<Json>>) {
    let (drafts, asked) = (Mutex::new(VecDeque::from(drafts)), Shared::new(Vec::new()));
    let seen = asked.clone();
    let write = repl.writer(name, WRITE, move |call| {
        seen.lock().push(Json::Object(call.shown().clone()));
        Ok(drafts.lock().unwrap().pop_front().expect("the writer has nothing more to give"))
    });
    (write, asked)
}

/// A module over a writer and the REPL it writes into.
fn house<const N: usize>(repl: &Repl, drafts: [Draft; N]) -> (Rdm, Shared<Vec<Json>>) {
    let (write, asked) = writer(repl, "write", drafts);
    (Rdm::new(operate()).tool(write).source(repl.clone()), asked)
}

fn offers(model: &Scripted) -> Vec<Vec<String>> {
    model.requests().iter().map(|request| offered(request, "next")).collect()
}

fn observed(model: &Scripted) -> Vec<String> {
    let observation = |inputs: &Json| inputs["observation"].as_str().unwrap_or_default().to_owned();
    shown(model).iter().map(observation).collect()
}

fn names(repl: &Repl) -> Vec<String> {
    repl.offers().iter().map(|tool| tool.name().to_owned()).collect()
}

fn failure(result: rdm::Result<String>) -> String {
    match result {
        Err(Error::Tool(why)) => why,
        other => panic!("expected a refusal, and got {other:?}"),
    }
}

fn step(result: &Prediction, turn: usize) -> (String, String) {
    let (name, observation) = result.steps("next")[turn].clone();
    (
        name,
        observation.as_str().expect("a written function's observation is text").to_owned(),
    )
}

// One REPL: what is written stays, and can be pointed at

#[test]
fn what_is_written_is_defined_in_the_repl_and_is_on_the_map() {
    let world = world();
    let repl = environment(&world);
    let (rdm, asked) = house(&repl, [wrote()]);
    let model = scripted(["write", "cool_three", "finish", "cool_three", "finish"]);
    let rdm = rdm.model(model.clone());
    let result = rdm
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    // The decision model pointed at `write`, a tool like any other, which called `define`.
    assert_eq!(asked.lock().len(), 1);
    assert_eq!(asked.lock()[0]["request"], "Three degrees cooler.");
    // What was defined is on the map from the next request: the map holds what the REPL offers then.
    assert_eq!(
        offers(&model)[..2],
        [vec!["write", "finish"], vec!["write", "cool_three", "finish"]]
    );
    assert_eq!(observed(&model)[1], WRITTEN);
    // The decision model pointed at the name, and the REPL ran it, with the functions it was given: the loop a decision model cannot make.
    assert_eq!(heat(&world), 18);
    assert_eq!(
        step(&result, 1),
        ("cool_three".to_owned(), "cooled by three: {\"heat\": 18}".to_owned())
    );
    assert_eq!(repl.written(), Functions::from([("cool_three".to_owned(), cool_three())]));
    // The change is recorded in the trajectory, on the request whose tool made it.
    let changes: Vec<_> = result.trajectory.iter().map(|entry| entry.changes.clone()).collect();
    assert_eq!(
        changes,
        [Some(vec![json!({"by": "repl", "did": "define", "name": "cool_three"})]), None, None]
    );
    assert_eq!(
        serde_json::to_string(&changes[0]).unwrap(),
        r#"[{"by":"repl","did":"define","name":"cool_three"}]"#
    );
    // `show` is for whoever writes next: what is written already, with its source.
    assert_eq!(repl.show(), format!("cool_three: Make it three degrees cooler.\n{COOL_THREE}"));
    // It is kept: the next run is offered it from its first request, and nothing is written again.
    rdm.call(inputs! { "request" => "And three more.", "world" => &world }).unwrap();
    assert_eq!(offers(&model)[3], ["write", "cool_three", "finish"]);
    assert_eq!(heat(&world), 15);
    assert_eq!(asked.lock().len(), 1);
    // `remove` takes a function off the map and forgets it.
    assert_eq!(repl.remove("cool_three"), "`cool_three` is no longer on the list.");
    assert!(repl.written().is_empty());
    assert_eq!(
        repl.remove("cool_three"),
        "There is no function `cool_three`: the functions are []."
    );
    assert!(names(&repl).is_empty());
}

#[test]
fn a_function_that_fails_is_taken_off_the_map_and_can_be_written_again() {
    let world = world();
    let broken = Draft::new(
        "cool_three",
        "Make it three degrees cooler.",
        "def cool_three():\n    return cooler() + str(1 // 0)",
    );
    let repl = environment(&world);
    let (rdm, asked) = house(&repl, [broken.clone(), wrote()]);
    let model = scripted(["write", "cool_three", "write", "cool_three", "finish"]);
    rdm.model(model.clone())
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    let (offers, observed) = (offers(&model), observed(&model));
    // A function that fails when it is run comes off the map, so it is not pointed at again.
    assert!(offers[1].contains(&"cool_three".to_owned()) && !offers[2].contains(&"cool_three".to_owned()));
    assert_eq!(
        observed[2],
        "`cool_three` failed and was taken off the list. ZeroDivisionError: division by zero (line 2 of `cool_three`)"
    );
    // The writer is given the error, as the request's observation, and what it writes next takes the name.
    assert!(asked.lock()[1]["observation"].as_str().unwrap().contains("ZeroDivisionError"));
    assert_eq!(heat(&world), 17);
    assert_eq!(repl.written()["cool_three"], cool_three());

    // Its source is kept, marked as failed, for whoever writes it again.
    let repl = environment(&world);
    let (rdm, _) = house(&repl, [broken.clone()]);
    let result = rdm
        .model(scripted(["write", "cool_three", "finish"]))
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    let written = repl.written();
    assert!(written["cool_three"].failed.as_ref().unwrap().contains("ZeroDivisionError"));
    assert!(names(&repl).is_empty());
    assert_eq!(
        result.trajectory[1].changes,
        Some(vec![json!({"by": "repl", "did": "fail", "name": "cool_three"})])
    );
    assert_eq!(
        repl.show(),
        format!(
            "cool_three: Make it three degrees cooler.\nIt failed and is off the list: ZeroDivisionError: division by zero (line 2 of `cool_three`)\n{}",
            broken.code
        )
    );
    // What is saved says so too, in the place a saved map has for it.
    assert_eq!(
        serde_json::to_value(&written).unwrap()["cool_three"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["description", "code", "failed"]
    );
}

#[test]
fn writing_only_defines_and_what_is_refused_changes_nothing() {
    let world = world();
    let repl = environment(&world);
    assert_eq!(
        repl.define("cool_three", "Make it three degrees cooler.", COOL_THREE).unwrap(),
        WRITTEN
    );
    let one_def = "The source must be one `def cool_three(...)` and nothing beside it, with no decorator, and plain values as \
                   defaults: writing a function defines it and runs nothing.";
    // Source that would run something when it is written: a call beside the function, a decorator, a default that calls.
    let refused = [
        format!("{COOL_THREE}\ncool_three()"),
        format!("@cooler\n{COOL_THREE}"),
        "def cool_three(now=cooler()):\n    return now".to_owned(),
        "x = 1".to_owned(),
        "async def cool_three():\n    return 1".to_owned(),
        "def cool_two():\n    return 2".to_owned(),
        String::new(),
    ];
    for code in &refused {
        assert_eq!(failure(repl.define("cool_three", "Something else.", code)), one_def, "{code}");
    }
    assert_eq!(
        failure(repl.define("cool_three", "Something else.", "def cool_three(:\n    return")),
        "What was written is not Python. Expected a parameter or the end of the parameter list (line 1)"
    );
    assert_eq!(
        failure(repl.define("cool_three", "Something else.", "```js\nfunction cool_three() {}\n```")),
        "What was written is not Python. Expected Python code but got ```js fence. Write Python code, not js."
    );
    // A name the REPL's own functions have: the function would take the place of one, for everything written before.
    assert_eq!(
        failure(repl.define("cooler", "Cool twice.", "def cooler():\n    return 'not the tool'")),
        "`cooler` is already the name of a tool. Give the function another name."
    );
    assert_eq!(
        failure(repl.define("cool three", "Cool.", COOL_THREE)),
        "`cool three` cannot be the name of a function."
    );
    // `finish` is on every list, and the REPL uses two names itself.
    assert_eq!(
        failure(repl.define("finish", "Stop.", "def finish():\n    return 1")),
        "`finish` is already the name of a tool. Give the function another name."
    );
    assert_eq!(
        failure(repl.define("print", "Print.", "def print():\n    return 1")),
        "`print` is a name the REPL uses. Give the function another name."
    );
    // Nothing ran while it was written, and the function written first is still the one the name has.
    assert_eq!(repl.written(), Functions::from([("cool_three".to_owned(), cool_three())]));
    assert_eq!(heat(&world), 21);
    assert_eq!(repl.execute("print(cool_three())").unwrap(), "cooled by three: {\"heat\": 18}");
    // A fence round the source, as a language model writes code, is taken off, and the name and description are trimmed.
    let fenced = format!("```python\n{COOL_THREE}\n```");
    assert_eq!(
        repl.define(" cool_three ", " Make it three degrees cooler. ", &fenced).unwrap(),
        WRITTEN
    );
    assert_eq!(repl.written()["cool_three"].code, COOL_THREE);
    // In a run, a name that a tool on the map has is refused too, and the writer is told.
    let (rdm, _) = house(&repl, [Draft::new("write", "Write.", "def write():\n    return 1")]);
    let model = scripted(["write", "finish"]);
    rdm.model(model.clone())
        .call(inputs! { "request" => "Write.", "world" => &world })
        .unwrap();
    assert_eq!(
        observed(&model)[1],
        "Nothing was put on the list. `write` is already the name of a tool. Give the function another name."
    );
    // So is the name of an input: where a function runs, each input is a variable under its name.
    let (rdm, _) = house(&repl, [Draft::new("world", "The world.", "def world():\n    return 1")]);
    let model = scripted(["write", "finish"]);
    rdm.model(model.clone())
        .call(inputs! { "request" => "Write.", "world" => &world })
        .unwrap();
    assert_eq!(
        observed(&model)[1],
        "Nothing was put on the list. `world` is already the name of an input of the request. Give the function another name."
    );
}

#[test]
fn a_function_that_cannot_be_defined_leaves_the_one_the_name_had() {
    let repl = Repl::new();
    repl.define("count", "Count to three.", "def count():\n    return [1, 2, 3]")
        .unwrap();
    // Monty has no generators: the source is one function, and defining it fails.
    let why = failure(repl.define("count", "Count to three.", "def count():\n    yield 1"));
    assert!(why.starts_with("What was written failed. NotImplementedError"), "{why}");
    assert_eq!(repl.execute("print(count())").unwrap(), "[1, 2, 3]");
    assert_eq!(repl.written()["count"].code, "def count():\n    return [1, 2, 3]");
}

#[test]
fn what_is_written_again_keeps_its_place_and_its_source_is_kept_as_it_is_run() {
    let repl = Repl::new();
    // The source is kept without the space round it, as it is without a fence.
    repl.define("first", "Say one.", "\n\ndef first():\n    return 1\n\n").unwrap();
    repl.define("second", "Say two.", "def second():\n    return 2").unwrap();
    assert_eq!(repl.written()["first"].code, "def first():\n    return 1");
    // What there is, is said to whoever asks for what there is not.
    assert_eq!(
        repl.remove("third"),
        "There is no function `third`: the functions are ['first', 'second']."
    );
    // A function written again takes the name where it stands, and is the one that runs.
    repl.define("first", "Say three.", "def first():\n    return 3").unwrap();
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["first", "second"]);
    assert_eq!(repl.written()["first"], Written::new("Say three.", "def first():\n    return 3"));
    assert_eq!(repl.execute("print(first())").unwrap(), "3");
    // A word of Python's own is a name to `isidentifier`, and no source defines it.
    let why = failure(repl.define("class", "A class.", "def class():\n    return 1"));
    assert!(why.starts_with("What was written is not Python. "), "{why}");
    assert_eq!(
        failure(repl.define("", "Nothing.", "def nothing():\n    return 1")),
        "`` cannot be the name of a function."
    );
    // A default is a plain value as `ast.Constant` is one: `-1` is a minus applied to `1`, and is refused.
    let why = failure(repl.define("below", "Below.", "def below(by=-1):\n    return by"));
    assert!(
        why.starts_with("The source must be one `def below(...)` and nothing beside it"),
        "{why}"
    );
    let every = "def every(a, b=1, *rest, c, d='x', e=None, f=True, g=2.5, **more):\n    return a";
    repl.define("every", "Every kind of parameter.", every).unwrap();
}

#[test]
fn a_written_function_sees_the_inputs_of_the_request_that_pointed_at_it() {
    let world = world();
    let repl = environment(&world);
    repl.define("echo", "Say what was asked.", "def echo():\n    return 'asked: ' + request")
        .unwrap();
    let rdm = Rdm::new(operate())
        .source(repl.clone())
        .model(scripted(["echo", "finish", "echo", "finish"]));
    let first = rdm.call(inputs! { "request" => "one thing", "world" => &world }).unwrap();
    // The request's inputs are variables in the REPL each time a function is run: not those of the run it was written in.
    let second = rdm.call(inputs! { "request" => "another thing", "world" => &world }).unwrap();
    assert_eq!(step(&first, 0), ("echo".to_owned(), "asked: one thing".to_owned()));
    assert_eq!(step(&second, 0), ("echo".to_owned(), "asked: another thing".to_owned()));
    // They are copies: a function changes the world through the functions it is given, and not by changing a variable.
    repl.define(
        "melt",
        "Set the heat.",
        "def melt():\n    world['heat'] = 99\n    return world['heat']",
    )
    .unwrap();
    let result = rdm
        .clone()
        .model(scripted(["melt", "finish"]))
        .call(inputs! { "request" => "melt", "world" => &world })
        .unwrap();
    assert_eq!(step(&result, 0).1, "99");
    assert_eq!(heat(&world), 21);
}

/// A decision model that points at these names in order, and answers any other question with one option.
fn pointing<const N: usize>(names: [&str; N], chosen: &'static str) -> Arc<Scripted> {
    let script = Mutex::new(VecDeque::from(names.map(str::to_owned)));
    client(move |_, name, _| match name {
        "next" => script.lock().unwrap().pop_front().expect("the script has run out"),
        _ => chosen.to_owned(),
    })
}

#[test]
fn a_written_function_takes_its_parameters_as_any_tool_does() {
    let operate_by = operate().choice("degrees", "By how many degrees does `inputs.request` ask?", ["1", "2", "3"]);
    let world = world();
    let repl = environment(&world);
    // A choice is answered with a label, which is text: the function makes a number of it.
    let cool_by = "def cool_by(degrees, request, times=1):\n    for _ in range(int(degrees) * times):\n        cooler()\n    return request + ' ' + read()";
    repl.define("cool_by", "Make it cooler by the degrees asked for.", cool_by).unwrap();
    let rdm = Rdm::new(operate_by).source(repl.clone());
    let result = rdm
        .clone()
        .model(pointing(["cool_by", "finish"], "3"))
        .call(inputs! { "request" => "Three cooler.", "world" => &world })
        .unwrap();
    // `degrees` is an answer of the request that pointed at the function, and `request` an input of it: each
    // is given by name, as to any tool. `times` names neither and has a default, so it is left alone.
    assert_eq!(heat(&world), 18);
    assert_eq!(step(&result, 0), ("cool_by".to_owned(), "Three cooler. {\"heat\": 18}".to_owned()));
    // So one function serves every number: what it is told is a question, answered by the decision model.
    rdm.clone()
        .model(pointing(["cool_by", "finish"], "2"))
        .call(inputs! { "request" => "Two cooler.", "world" => &world })
        .unwrap();
    assert_eq!(heat(&world), 16);
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["cool_by"]);
    // A parameter that names no answer or input is said, as an observation: a language model named it and can write it again.
    repl.define("cool_some", "Make it cooler.", "def cool_some(amount, *, unit):\n    return amount")
        .unwrap();
    let result = rdm
        .model(pointing(["cool_some", "finish"], "1"))
        .call(inputs! { "request" => "Cooler.", "world" => &world })
        .unwrap();
    assert_eq!(
        step(&result, 0).1,
        "`cool_some` was not run: it takes ['amount', 'unit'], which name no answer or input of the request. \
         A parameter is filled by name from ['degrees', 'next', 'observation', 'request', 'world']."
    );
    // It was not run, so it has not failed: it is on the list still.
    assert_eq!(names(&repl), ["cool_by", "cool_some"]);
}

#[test]
fn a_parameter_is_given_whatever_the_request_has_under_its_name() {
    let repl = Repl::new();
    // An answer that is a yes, a choice that chose none, an input that is not text, a parameter that has a
    // default and names an input, and two that take whatever is left: nothing is left, so they are empty.
    let say = "def say(request, urgent, team='support', extra='x', *rest, **more):\n    print('printed')\n    return [request, urgent, team, extra, rest, more]";
    repl.define("say", "Say what was given.", say).unwrap();
    repl.define("nothing", "Return nothing.", "def nothing():\n    pass").unwrap();
    let signature = Signature::new("Say.")
        .input("request")
        .input_typed("extra", "dict")
        .input("observation")
        .noul("urgent", "Is `inputs.request` urgent?")
        .choice_or_none("team", "Which team is `inputs.request` for?", ["billing", "tech"], "No team.")
        .tool("next", "What next?");
    let model = client(|inputs, name, _| match (name, inputs["observation"].as_str()) {
        ("urgent", _) => common::Say::Yes(true),
        ("team", _) => "None".into(),
        (_, Some("")) => "say".into(),
        (_, Some("None")) => "finish".into(),
        _ => "nothing".into(),
    });
    let result = Rdm::new(signature)
        .source(repl)
        .model(model)
        .call(inputs! { "request" => "now", "extra" => json!({"a": [1, 2.5, null]}) })
        .unwrap();
    // What a function prints is observed with what it returns, as the REPL printed them.
    assert_eq!(step(&result, 0).1, "printed\n['now', True, None, {'a': [1, 2.5, None]}, (), {}]");
    // A function that returns nothing is observed as Python prints nothing.
    assert_eq!(step(&result, 1).1, "None");
}

#[test]
fn two_repls_share_one_map() {
    let world = world();
    let note = Draft::new("note", "Note what was asked.", "def note():\n    return 'noted: ' + request");
    let (heating, notes) = (environment(&world), environment(&world).named("notes"));
    // Each has a tool that writes into it, under a name of its own. What both offer is on the one map.
    let (write_heating, _) = writer(&heating, "write_heating", [wrote()]);
    let (write_note, _) = writer(&notes, "write_note", [note]);
    let model = scripted(["write_heating", "write_note", "cool_three", "note", "finish"]);
    let rdm = Rdm::new(operate())
        .tool(write_heating)
        .tool(write_note)
        .source(heating.clone())
        .source(notes.clone())
        .model(model.clone());
    let result = rdm
        .call(inputs! { "request" => "Cooler, and note it.", "world" => &world })
        .unwrap();
    assert_eq!(
        offers(&model).last().unwrap(),
        &["write_heating", "write_note", "cool_three", "note", "finish"]
    );
    assert_eq!(heating.written().keys().collect::<Vec<_>>(), ["cool_three"]);
    assert_eq!(notes.written().keys().collect::<Vec<_>>(), ["note"]);
    assert_eq!(heat(&world), 18);
    assert_eq!(step(&result, 3), ("note".to_owned(), "noted: Cooler, and note it.".to_owned()));
    // Each change says whose it is.
    assert_eq!(
        result.trajectory[1].changes,
        Some(vec![json!({"by": "notes", "did": "define", "name": "note"})])
    );
    // A name that two of them offer is a mistake of the program, and is said.
    notes.define("cool_three", "Make it three degrees cooler.", COOL_THREE).unwrap();
    let error = rdm.call(inputs! { "request" => "Again.", "world" => &world }).unwrap_err();
    assert_eq!(
        error,
        Error::Usage("['cool_three'] are offered by `Repl` and are already names on the map.".to_owned())
    );
}

#[test]
fn a_writer_is_shown_what_is_there_and_takes_a_function_away() {
    let world = world();
    let repl = environment(&world);
    repl.define("cool_three", "Make it three degrees cooler.", COOL_THREE).unwrap();
    assert_eq!(Repl::new().show(), "No function has been written.");
    let (showing, removing) = (repl.clone(), repl.clone());
    let look = Tool::new("look", "Read what is written.", move |call| Ok(showing.show_in(call)));
    let forget = Tool::new("forget", "Take what is written away.", move |call| {
        Ok(removing.remove_in(call, "cool_three"))
    });
    let model = scripted(["look", "forget", "finish"]);
    let rdm = Rdm::new(operate())
        .tool(look)
        .tool(forget)
        .source(repl.clone())
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => "Tidy up.", "world" => &world }).unwrap();
    // In a run, `show` lists the other names on the map as well, which a function cannot take.
    assert_eq!(
        step(&result, 0).1,
        format!(
            "cool_three: Make it three degrees cooler.\n{COOL_THREE}\n\nOther names on the list, which a function cannot take: look, forget"
        )
    );
    assert_eq!(step(&result, 1).1, "`cool_three` is no longer on the list.");
    assert_eq!(
        result.trajectory[1].changes,
        Some(vec![json!({"by": "repl", "did": "remove", "name": "cool_three"})])
    );
    // What was removed is off the map from the next request.
    assert_eq!(
        offers(&model),
        [
            vec!["look", "forget", "cool_three", "finish"],
            vec!["look", "forget", "cool_three", "finish"],
            vec!["look", "forget", "finish"]
        ]
    );
}

#[test]
fn a_tool_that_gives_a_function_as_data_is_a_writer() {
    // What gives the function is a tool of its own: a signature a language model completes, say. Here it is a
    // function, which gives an object with no source first, and reports the tokens a model would have used.
    let gave = Mutex::new(VecDeque::from([
        json!({"name": "cool_three", "description": "Make it three degrees cooler."}),
        json!({"name": "cool_three", "description": "Make it three degrees cooler.", "code": format!("```python\n{COOL_THREE}\n```"), "reasoning": "A loop."}),
    ]));
    let gives = Tool::new("write", WRITE, move |call| {
        let _request: String = call.arg("request")?;
        call.spend(7);
        Ok(gave.lock().unwrap().pop_front().expect("nothing more to give"))
    });
    let world = world();
    let repl = environment(&world);
    let write = repl.writer_from(&gives);
    // It is on the map as the tool it was made of was to be.
    assert_eq!((write.name(), write.description()), ("write", WRITE));
    let model = scripted(["write", "write", "cool_three", "finish"]);
    let result = Rdm::new(operate())
        .tool(write)
        .source(repl.clone())
        .model(model.clone())
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    // What was given and is not a function is the tool's failure, said as any tool's is: whoever writes can write again.
    assert_eq!(
        observed(&model)[1..],
        [
            "Execution error in write: What was given is not a function's name, description and code: missing field `code`.",
            WRITTEN,
            "cooled by three: {\"heat\": 18}"
        ]
    );
    assert_eq!(repl.written(), Functions::from([("cool_three".to_owned(), cool_three())]));
    // The tool it was made of ran for the request that pointed at the writer: its tokens are that request's.
    assert_eq!(result.trajectory[0].tool_tokens.as_ref().unwrap()["write"], 7);
    assert_eq!(
        result.trajectory[1].changes,
        Some(vec![json!({"by": "repl", "did": "define", "name": "cool_three"})])
    );
    // A function is read from any data that has the three: a struct of the program's own as well as JSON.
    #[derive(serde::Serialize)]
    struct Wrote {
        name: &'static str,
        description: &'static str,
        code: &'static str,
    }
    let wrote = Wrote {
        name: "cool_three",
        description: "Make it three degrees cooler.",
        code: COOL_THREE,
    };
    assert_eq!(Draft::read(&wrote).unwrap(), self::wrote());
    assert_eq!(
        Draft::read(&json!("def cool_three(): pass")).unwrap_err(),
        Error::Tool(
            "What was given is not a function's name, description and code: invalid type: string \"def cool_three(): pass\", expected struct Draft."
                .to_owned()
        )
    );
}

#[test]
fn a_name_a_tool_took_away_can_be_written_again_at_once() {
    // A run's map is made when a request is. A tool that takes a function away and writes it again does both
    // before the run's next request, while the map still holds the name: the name is the REPL's own to give again.
    let repl = Repl::new();
    repl.define("hello", "Say hello.", "def hello():\n    return 'hello'").unwrap();
    let writing = repl.clone();
    let rewrite = Tool::new("rewrite", "Write it again.", move |call| {
        let gone = writing.remove_in(call, "hello");
        let shown = writing.show_in(call);
        let done = writing.define_in(call, "hello", "Say hello, louder.", "def hello():\n    return 'HELLO'")?;
        Ok(json!([gone, shown, done]))
    });
    let signature = Signature::new("Greet.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let model = scripted(["rewrite", "hello", "finish"]);
    let result = Rdm::new(signature)
        .tool(rewrite)
        .source(repl.clone())
        .model(model.clone())
        .call(inputs! { "request" => "Louder." })
        .unwrap();
    assert_eq!(
        result.steps("next")[0].1,
        json!([
            "`hello` is no longer on the list.",
            // It is not among the names a function cannot take.
            "Other names on the list, which a function cannot take: rewrite",
            "`hello` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Say hello, louder."
        ])
    );
    assert_eq!(offers(&model)[1], ["rewrite", "hello", "finish"]);
    assert_eq!(step(&result, 1).1, "HELLO");
    assert_eq!(
        result.trajectory[0].changes,
        Some(vec![
            json!({"by": "repl", "did": "remove", "name": "hello"}),
            json!({"by": "repl", "did": "define", "name": "hello"})
        ])
    );
    // A name that is another tool's is no freer for it: taking away what is not there frees nothing.
    let writing = repl.clone();
    let take = Tool::new("take", "Take a tool's name.", move |call| {
        let gone = writing.remove_in(call, "take");
        Ok(json!([
            gone,
            writing
                .define_in(call, "take", "Take.", "def take():\n    return 1")
                .unwrap_err()
                .to_string()
        ]))
    });
    let signature = Signature::new("Take.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let result = Rdm::new(signature)
        .tool(take)
        .source(repl.clone())
        .model(scripted(["take", "finish"]))
        .call(inputs! { "request" => "Take." })
        .unwrap();
    assert_eq!(
        result.steps("next")[0].1,
        json!([
            "There is no function `take`: the functions are ['hello'].",
            "`take` is already the name of a tool. Give the function another name."
        ])
    );
}

#[test]
fn a_name_written_into_two_repls_before_the_next_request_is_said_at_that_request() {
    // The Python's `define` reads every source of the module again, so the second of these is refused where
    // it is written. Here a tool knows the run's map as the last request made it, which holds neither yet: both
    // are written, and the mistake is said one request later, as any name that two sources offer is.
    let (one, two) = (Repl::new().named("one"), Repl::new().named("two"));
    let (first, second) = (one.clone(), two.clone());
    let both = Tool::new("both", "Write into both.", move |call| {
        let code = "def hello():\n    return 'hello'";
        first.define_in(call, "hello", "Say hello.", code)?;
        second.define_in(call, "hello", "Say hello.", code)
    });
    let signature = Signature::new("Greet.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let error = Rdm::new(signature)
        .tool(both)
        .source(one)
        .source(two)
        .model(scripted(["both", "hello", "finish"]))
        .call(inputs! { "request" => "Hello." })
        .unwrap_err();
    assert_eq!(
        error,
        Error::Usage("['hello'] are offered by `Repl` and are already names on the map.".to_owned())
    );
}

// A Repl in a program: saved, loaded, copied, and used from any thread

#[test]
fn what_was_written_is_plain_data_that_is_saved_and_loaded() {
    let world = world();
    let repl = environment(&world);
    repl.define("cool_three", "Make it three degrees cooler.", COOL_THREE).unwrap();
    let saved = serde_json::to_string(&repl.written()).unwrap();
    assert_eq!(
        saved,
        format!(
            r#"{{"cool_three":{{"description":"Make it three degrees cooler.","code":{}}}}}"#,
            json!(COOL_THREE)
        )
    );
    // A REPL built the same way and loaded has the functions, defined in itself, and nothing that was written in it before.
    let restored = environment(&world);
    restored
        .define("stale", "Something written before the load.", "def stale():\n    return 'stale'")
        .unwrap();
    restored.execute("kept = 7").unwrap();
    restored.load(serde_json::from_str::<Functions>(&saved).unwrap());
    assert_eq!(restored.written(), repl.written());
    let model = scripted(["cool_three", "finish"]);
    Rdm::new(operate())
        .source(restored.clone())
        .model(model.clone())
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    assert_eq!(offers(&model)[0], ["cool_three", "finish"]);
    assert_eq!(heat(&world), 18);
    for gone in ["print(stale())", "print(kept)"] {
        assert!(failure(restored.execute(gone)).starts_with("NameError"));
    }
    // The source is plain data too: another REPL can be given it.
    let given = environment(&world).with_written(serde_json::from_str::<Functions>(&saved).unwrap());
    assert_eq!(given.execute("print(cool_three())").unwrap(), "cooled by three: {\"heat\": 15}");
}

#[test]
fn a_copy_of_the_module_has_a_repl_of_its_own() {
    let world = world();
    let repl = environment(&world);
    let (rdm, _) = house(&repl, [wrote()]);
    repl.define("first", "Say first.", "def first():\n    return 'first'").unwrap();
    // A copy of the module starts with the same functions, and its REPL is another one.
    let twin = rdm.copy();
    let copied = repl.copies().pop().expect("the copy of the module copied the REPL");
    assert_eq!(copied.written(), repl.written());
    copied.define("cool_three", "Make it three degrees cooler.", COOL_THREE).unwrap();
    let model = scripted(["cool_three", "finish"]);
    twin.model(model.clone())
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    assert_eq!(offers(&model)[0], ["write", "first", "cool_three", "finish"]);
    assert_eq!(heat(&world), 18);
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["first"]);
    assert_eq!(names(&repl), ["first"]);
    // Nothing else an environment holds is shared either.
    repl.execute("kept = 7").unwrap();
    assert!(failure(copied.execute("print(kept)")).starts_with("NameError"));
    // A clone, though, is the same REPL: that is how a tool and a module hold one between them.
    assert_eq!(repl.clone().execute("print(kept)").unwrap(), "7");
    // A copy that nothing holds is not kept.
    drop((copied, model));
    assert!(repl.copies().is_empty());
}

#[test]
fn a_copy_of_the_module_writes_into_its_own_repl() {
    let world = world();
    let repl = environment(&world);
    let hello = Draft::new("hello", "Say hello.", "def hello():\n    return 'hello'");
    let (rdm, _) = house(&repl, [wrote(), hello]);
    // Tools that hold the REPL they were made with, as a writer does: one that shows, one that takes away.
    let (showing, forgetting) = (repl.clone(), repl.clone());
    let rdm = rdm
        .tool(Tool::new("show", "Show what was written.", move |call| Ok(showing.show_in(call))))
        .tool(Tool::new("forget", "Forget the first.", move |call| {
            Ok(forgetting.remove_in(call, "first"))
        }));
    repl.define("first", "Say first.", "def first():\n    return 'first'").unwrap();

    // A copy of the module has the same tools, and a REPL of its own. What its tools write, show
    // and take away in a run of the copy is in the copy's REPL, where the copy's next request finds it.
    let twin = rdm.copy();
    let copied = repl.copies().pop().expect("the copy of the module copied the REPL");
    let model = scripted(["write", "show", "cool_three", "forget", "finish"]);
    let result = twin
        .clone()
        .model(model.clone())
        .call(inputs! { "request" => "Three degrees cooler.", "world" => &world })
        .unwrap();
    assert_eq!(step(&result, 0).1, WRITTEN);
    assert!(step(&result, 1).1.contains("cool_three: Make it three degrees cooler."));
    assert_eq!(step(&result, 3).1, "`first` is no longer on the list.");
    assert_eq!(
        offers(&model),
        [
            vec!["write", "show", "forget", "first", "finish"],
            vec!["write", "show", "forget", "first", "cool_three", "finish"],
            vec!["write", "show", "forget", "first", "cool_three", "finish"],
            vec!["write", "show", "forget", "first", "cool_three", "finish"],
            vec!["write", "show", "forget", "cool_three", "finish"],
        ]
    );
    assert_eq!(heat(&world), 18);
    assert_eq!(names(&copied), ["cool_three"]);
    // The original was not run, and is as it was.
    assert_eq!(names(&repl), ["first"]);
    let model = scripted(["finish"]);
    rdm.clone()
        .model(model.clone())
        .call(inputs! { "request" => "Nothing.", "world" => &world })
        .unwrap();
    assert_eq!(offers(&model)[0], ["write", "show", "forget", "first", "finish"]);

    // So it is with a copy of the copy.
    let model = scripted(["write", "finish"]);
    let third = twin.copy().model(model.clone());
    third.call(inputs! { "request" => "Say hello.", "world" => &world }).unwrap();
    assert_eq!(offers(&model)[1], ["write", "show", "forget", "cool_three", "hello", "finish"]);
    let last = copied.copies().pop().expect("the copy of the copy copied its REPL");
    assert_eq!(
        (names(&last), names(&copied), names(&repl)),
        (
            vec!["cool_three".to_owned(), "hello".to_owned()],
            vec!["cool_three".to_owned()],
            vec!["first".to_owned()]
        )
    );
    // Outside a run there is no module to ask: a handle writes into the REPL it is.
    repl.define("second", "Say second.", "def second():\n    return 'second'").unwrap();
    assert_eq!(
        (names(&repl), names(&copied)),
        (vec!["first".to_owned(), "second".to_owned()], vec!["cool_three".to_owned()])
    );
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn a_repl_runs_one_thing_at_a_time_whatever_thread_asks() {
    let world = world();
    let (heater, heating) = (environment(&world), environment(&world).named("heating"));
    heater.execute("def double(n):\n    return n * 2").unwrap();
    // A module's tools and batches run on many threads: all of them reach the same REPL, each in its turn.
    let doubled: Vec<String> = std::thread::scope(|threads| {
        let asked: Vec<_> = (0..4)
            .map(|n| {
                let heater = &heater;
                threads.spawn(move || heater.execute(&format!("print(double({n}))")).unwrap())
            })
            .collect();
        asked.into_iter().map(|thread| thread.join().unwrap()).collect()
    });
    assert_eq!(doubled, ["0", "2", "4", "6"]);
    let other = heater.clone();
    std::thread::spawn(move || other.execute("kept = 7").unwrap()).join().unwrap();
    assert_eq!(heater.execute("print(kept)").unwrap(), "7");
    // Each REPL is its own, whatever the REPLs are called.
    assert!(failure(heating.execute("print(kept)")).starts_with("NameError"));
}

#[test]
fn functions_pointed_at_by_one_request_each_wait_their_turn() {
    // Two outputs each named after a written function: both are selected at once, and run one after the other.
    let world = world();
    let repl = environment(&world);
    for name in ["first", "second"] {
        repl.define(name, "Cool once.", &format!("def {name}():\n    return cooler()"))
            .unwrap();
    }
    let both = Signature::new("Cool twice.")
        .input("request")
        .noul("first", "Does `inputs.request` ask for it?")
        .noul("second", "Does `inputs.request` ask for it?");
    let result = Rdm::new(both)
        .source(repl)
        .model(client(|_, _, _| true))
        .call(inputs! { "request" => "Both." })
        .unwrap();
    let mut said: Vec<&str> = ["first", "second"]
        .iter()
        .map(|name| result.observations[*name].as_str().unwrap())
        .collect();
    said.sort();
    assert_eq!(said, ["heat is 19", "heat is 20"]);
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn code_a_repl_is_running_cannot_call_back_into_it_and_can_reach_another_environment() {
    let said = Shared::new(Vec::new());
    let repl = Repl::new().limits(Limits {
        wait: Duration::from_millis(200),
        ..Limits::default()
    });
    let room = repl.spawn("room");
    // Ask the REPL for something from the function it called.
    let (again, heard) = (repl.clone(), said.clone());
    let repl = repl.function("again", "Ask the REPL from inside it.", move |_| {
        match again.execute("print(inner())") {
            Err(error) => {
                heard.lock().push(error.to_string());
                Ok("refused".to_owned())
            }
            ran => ran,
        }
    });
    // Ask it from another thread, and wait for that thread.
    let (elsewhere, heard) = (repl.clone(), said.clone());
    let repl = repl.function("elsewhere", "Ask the REPL from another thread.", move |_| {
        let asked = elsewhere.clone();
        match std::thread::spawn(move || asked.execute("print(1)")).join().unwrap() {
            Err(error) => {
                heard.lock().push(error.to_string());
                Ok("refused".to_owned())
            }
            ran => ran,
        }
    });
    // Run a function written in another environment.
    let reached = room.clone();
    let repl = repl.function("other", "Run what another environment holds.", move |_| {
        reached.execute("print(inner())")
    });
    for environment in [&repl, &room] {
        let code = format!("def inner():\n    return 'inner ran in {}'", environment.name());
        environment.define("inner", "Say so.", &code).unwrap();
    }
    // A REPL runs one thing at a time. Code it is running cannot call back into it through a function of the
    // host: on the function's thread that is known at once, and on another it is known once it has waited.
    assert_eq!(repl.execute("print(again())").unwrap(), "refused");
    assert_eq!(repl.execute("print(elsewhere())").unwrap(), "refused");
    assert_eq!(
        *said.lock(),
        [
            "The REPL `repl` is running the code that asked for this, and runs one thing at a time: \
             a function written here cannot be run from inside one that is running.",
            "The REPL `repl` runs one thing at a time, and was still running something else after 200ms.",
        ]
    );
    // One written function calls another directly, and another environment, with its own interpreter, can be reached.
    assert_eq!(
        repl.execute("def outer():\n    return 'outer, then ' + inner()\nprint(outer())")
            .unwrap(),
        "outer, then inner ran in repl"
    );
    assert_eq!(repl.execute("print(other())").unwrap(), "inner ran in room");
    // A function that writes from inside the REPL is told the same, as what it wrote failing.
    let writing = repl.clone();
    let repl = repl.function("write", "Write from inside.", move |_| {
        writing.define("late", "Late.", "def late():\n    return 1")
    });
    let why = failure(repl.execute("write()"));
    assert!(why.starts_with("RuntimeError: What was written failed. The REPL `repl` is running the code that asked for this"));
}

#[test]
fn a_request_made_under_a_function_that_is_running_cannot_run_one_of_its_repl() {
    // A tool of the module is a function in the REPL. A written function calls it, and the tool
    // asks the module again: those requests, one level down, point at a function of the same REPL.
    // One is asked on the thread the REPL is running on, and two at once on threads of their own.
    let below = |prediction: &Prediction| prediction.steps("next")[0].1.clone();
    let again = Tool::new("again", "Ask again, one level down.", move |call| {
        let one = call.sub(inputs! { "request" => "below" })?;
        let two = call.batch([inputs! { "request" => "below, left" }, inputs! { "request" => "below, right" }])?;
        Ok(json!([below(&one), below(&two[0]), below(&two[1])]))
    });
    let repl = Repl::new().tool(&again);
    repl.define("outer", "Ask again.", "def outer():\n    return again()").unwrap();
    repl.define("inner", "Say so.", "def inner():\n    return 'inner ran'").unwrap();
    let signature = Signature::new("Ask.")
        .input("request")
        .input("observation")
        .tool("next", "What should happen next for `inputs.request`?");
    let model = client(|inputs, _, _| match (inputs["request"].as_str(), inputs["observation"].as_str()) {
        (Some("above"), Some("")) => "outer",
        (Some(_), Some("")) => "inner",
        _ => "finish",
    });
    let started = std::time::Instant::now();
    let result = Rdm::new(signature)
        .source(repl.clone())
        .model(model)
        .call(inputs! { "request" => "above" })
        .unwrap();
    // Each is refused at once, and told why: none of them waited for the REPL to be free.
    let refused = "Execution error in inner: The REPL `repl` is running the code that asked for this, and runs one thing at a time: \
                   a function written here cannot be run from inside one that is running.";
    assert_eq!(step(&result, 0).1, format!("['{refused}', '{refused}', '{refused}']"));
    assert!(started.elapsed() < Duration::from_secs(10));
    // Neither function failed, so both are on the list still.
    assert_eq!(names(&repl), ["outer", "inner"]);
    assert_eq!(result.trajectory.iter().filter(|entry| entry.depth == 1).count(), 6);
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn a_request_another_tool_made_while_a_function_runs_waits_its_turn() {
    // One request selects two things at once: a written function that takes a while, and a tool that asks the
    // module again. The request below points at another function of the same REPL. It is under the request the
    // REPL is running for, and no code the REPL is running asked for it: it waits its turn, and runs.
    let meeting = common::Meeting::of(2);
    let (held, asking) = (meeting.clone(), meeting.clone());
    let repl = Repl::new().function("hold", "Wait a while.", move |_| {
        // Until the other tool is about to ask, and then long enough for what it asks to reach the REPL.
        let met = held.meet();
        std::thread::sleep(Duration::from_millis(300));
        Ok(met)
    });
    repl.define(
        "slow",
        "Take a while.",
        "def slow():\n    return 'slow ran' if hold() else 'nothing asked'",
    )
    .unwrap();
    repl.define("inner", "Say so.", "def inner():\n    return 'inner ran'").unwrap();
    let ask = Tool::new("ask", "Ask again, one level down.", move |call| {
        asking.meet();
        let below = call.sub(inputs! { "request" => "below" })?;
        Ok(below.observation("inner").cloned())
    });
    let signature = Signature::new("Ask.")
        .input("request")
        .noul("slow", "Does `inputs.request` ask for what is slow?")
        .noul("ask", "Does `inputs.request` ask to be asked again?")
        .noul("inner", "Does `inputs.request` ask for what is inner?");
    // Above, a yes to `slow` and to `ask`. Below, a yes to `inner`.
    let model = client(|inputs, name, _| (inputs["request"] == "above") != (name == "inner"));
    let result = Rdm::new(signature)
        .tool(ask)
        .source(repl.clone())
        .model(model)
        .call(inputs! { "request" => "above" })
        .unwrap();
    assert_eq!(result.observations["slow"], "slow ran");
    assert_eq!(result.observations["ask"], "inner ran");
    assert_eq!(names(&repl), ["slow", "inner"]);
}

#[test]
fn runs_at_once_share_one_repl() {
    // Several runs of a module at once, each on a thread of its own, point at functions of the one REPL.
    let world = world();
    let repl = environment(&world);
    repl.define("cool", "Cool once.", "def cool():\n    return cooler()").unwrap();
    let model = client(|inputs, _, _| if inputs["observation"] == "" { "cool" } else { "finish" });
    let rdm = Rdm::new(operate()).source(repl).model(model);
    let asks = (0..8).map(|run| inputs! { "request" => format!("Cooler, {run}."), "world" => json!({}) });
    let results = rdm.batch(asks);
    let mut said: Vec<String> = results.into_iter().map(|result| step(&result.unwrap(), 0).1).collect();
    said.sort();
    let mut expected: Vec<String> = (13..21).map(|heat| format!("heat is {heat}")).collect();
    expected.sort();
    assert_eq!(said, expected);
    assert_eq!(heat(&world), 13);
}

// Several REPLs: environments that are isolated, and routed to by name

#[test]
fn environments_are_isolated_and_a_name_on_the_map_routes_to_one() {
    let world = world();
    let lights_off = Draft::new("lights_off", "Turn the lights off.", "def lights_off():\n    return 'lights off'");
    let repl = environment(&world);
    let (heating, lights) = (repl.spawn("heating"), repl.spawn("lights"));
    let model = scripted([
        "heating",
        "write",
        "cool_three",
        "finish",
        "lights",
        "write",
        "lights_off",
        "finish",
        "finish",
    ]);
    // Each environment is a module over its own map. On the root's map a name stands for a whole environment.
    let room = |environment: &Repl, wrote: Draft, description: &str| {
        let (module, _) = house(environment, [wrote]);
        Tool::module(environment.name(), description, &module.model(model.clone()))
    };
    let root = Rdm::new(operate())
        .tool(room(&heating, wrote(), "Anything about the temperature."))
        .tool(room(&lights, lights_off, "Anything about the lights."))
        .model(model.clone());
    let result = root
        .call(inputs! { "request" => "Three degrees cooler, and the lights off.", "world" => &world })
        .unwrap();
    // The root was offered the environments, and pointing at one put the request to it, one level down.
    assert_eq!(offers(&model)[0], ["heating", "lights", "finish"]);
    assert_eq!(heat(&world), 18);
    let walked: Vec<(usize, Json)> = result.trajectory.iter().map(|entry| (entry.depth, entry.value("next"))).collect();
    let expected = [
        (0, "heating"),
        (1, "write"),
        (1, "cool_three"),
        (1, "finish"),
        (0, "lights"),
        (1, "write"),
        (1, "lights_off"),
        (1, "finish"),
        (0, "finish"),
    ];
    assert_eq!(walked, expected.map(|(depth, name)| (depth, json!(name))));
    // Each has its own interpreter, and what was written in one is there and nowhere else: not in
    // another's REPL, and not on another's map.
    assert_eq!(heating.written().keys().collect::<Vec<_>>(), ["cool_three"]);
    assert_eq!(lights.written().keys().collect::<Vec<_>>(), ["lights_off"]);
    assert!(repl.written().is_empty());
    assert_eq!(heating.execute("print(cool_three())").unwrap(), "cooled by three: {\"heat\": 15}");
    assert_eq!(lights.execute("print(lights_off())").unwrap(), "lights off");
    for (room, other) in [(&lights, "cool_three"), (&heating, "lights_off"), (&repl, "cool_three")] {
        assert!(failure(room.execute(&format!("print({other}())"))).starts_with("NameError"));
    }
    assert_eq!(offers(&model)[5], ["write", "finish"]);
    // So is anything else an environment holds.
    heating.execute("kept = 7").unwrap();
    assert_eq!(heating.execute("print(kept)").unwrap(), "7");
    assert!(failure(lights.execute("print(kept)")).starts_with("NameError"));
    // They are given the same functions, which act on one world.
    assert_eq!(lights.execute("print(cooler())").unwrap(), "heat is 14");
    assert_eq!(lights.describe(), repl.describe());
}

// The functions of the host, and what written code is held to

#[test]
fn a_function_of_the_host_is_called_as_it_was_declared() {
    let calls = Shared::new(Vec::new());
    let (switched, waited, logged) = (calls.clone(), calls.clone(), calls.clone());
    let repl = Repl::new()
        .function("switch(device, to)", "Set a device to a state.", move |args| {
            let (device, to): (String, String) = (args.get("device")?, args.get("to")?);
            switched.lock().push(format!("switch {device} {to}"));
            Ok(json!({"device": device, "now": to, "was": [1, 2.5, null, true]}))
        })
        .function("wait(seconds=1, *, unit='s')", "Wait a while.", move |args| {
            let (seconds, unit): (Option<f64>, Option<String>) = (args.opt("seconds")?, args.opt("unit")?);
            waited.lock().push(format!("wait {seconds:?} {unit:?}"));
            Ok(())
        })
        .function("log(*lines, **marks)", "Keep some lines.", move |args| {
            logged.lock().push(format!("log {:?} {:?}", args.rest(), args.named()));
            Ok(args.rest().len())
        })
        .function("now", "The hour.", |_| Ok(7));
    // What a writer is told it may call: each as it is called, with what it does.
    assert_eq!(
        repl.describe(),
        "switch(device, to): Set a device to a state.\nwait(seconds=1, *, unit='s'): Wait a while.\n\
         log(*lines, **marks): Keep some lines.\nnow(): The hour."
    );
    assert_eq!(repl.functions(), ["switch", "wait", "log", "now"]);
    // By position or by keyword, an argument is read by the name it was declared under. What is returned is Python data.
    assert_eq!(
        repl.execute("print(switch('tv', to='off'))").unwrap(),
        "{'device': 'tv', 'now': 'off', 'was': [1, 2.5, None, True]}"
    );
    assert_eq!(
        repl.execute("print(wait(), wait(2.5), wait(unit='h'), log('a', ('b', 1), kept=True), now() + 1)")
            .unwrap(),
        "None None None 2 8"
    );
    assert_eq!(
        *calls.lock(),
        [
            "switch tv off",
            "wait None None",
            "wait Some(2.5) None",
            "wait None Some(\"h\")",
            "log [String(\"a\"), Array [String(\"b\"), Number(1)]] {\"kept\": Bool(true)}",
        ]
    );
    // Code that calls one as it was not declared is told so, as Python tells it.
    let refused = [
        ("switch('tv')", "TypeError: switch() missing required argument: 'to'"),
        ("switch()", "TypeError: switch() missing required arguments: 'device', 'to'"),
        (
            "switch('tv', 'off', 'now')",
            "TypeError: switch() takes 2 positional arguments but 3 were given",
        ),
        (
            "switch('tv', device='tv')",
            "TypeError: switch() got multiple values for argument 'device'",
        ),
        (
            "switch('tv', 'off', soon=True)",
            "TypeError: switch() got an unexpected keyword argument 'soon'",
        ),
        ("wait(1, 'h')", "TypeError: wait() takes 1 positional argument but 2 were given"),
        ("now(1)", "TypeError: now() takes 0 positional arguments but 1 was given"),
        (
            "switch(1, 2)",
            "RuntimeError: switch() was given `device` as a type it is not: invalid type: integer `1`, expected a string.",
        ),
        ("nothing()", "NameError: name 'nothing' is not defined"),
    ];
    for (code, why) in refused {
        assert_eq!(failure(repl.execute(code)), why, "{code}");
    }
    assert_eq!(calls.lock().len(), 5);
    // The name of a function is a value too.
    assert_eq!(repl.execute("f = now\nprint(f())").unwrap(), "7");
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn a_function_of_the_host_that_fails_is_an_error_the_code_can_catch() {
    let repl = Repl::new()
        .function("jammed", "A lock that is jammed.", |_| -> rdm::Result<()> {
            Err("the lock is jammed".into())
        })
        .function("broken", "A lock that is broken.", |_| -> rdm::Result<()> {
            panic!("the lock fell off")
        })
        .function("misused", "A module used wrongly.", |_| -> rdm::Result<()> {
            Err(Error::usage("no such input"))
        });
    assert_eq!(failure(repl.execute("jammed()")), "RuntimeError: the lock is jammed");
    assert_eq!(failure(repl.execute("broken()")), "RuntimeError: the lock fell off");
    let caught = "def careful():\n    try:\n        jammed()\n    except RuntimeError as error:\n        return 'caught: ' + str(error)";
    repl.define("careful", "Try the lock.", caught).unwrap();
    assert_eq!(repl.execute("print(careful())").unwrap(), "caught: the lock is jammed");
    // A written function it fails in has failed, where it failed.
    repl.define("careless", "Try the lock.", "def careless():\n    x = 1\n    jammed()")
        .unwrap();
    let signature = Signature::new("Try.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let rdm = Rdm::new(signature).source(repl.clone());
    let result = rdm
        .clone()
        .model(scripted(["careless", "finish"]))
        .call(inputs! { "request" => "Try." })
        .unwrap();
    assert_eq!(
        step(&result, 0).1,
        "`careless` failed and was taken off the list. RuntimeError: the lock is jammed (line 3 of `careless`)"
    );
    // A module used wrongly is the caller's error wherever it happens, and the code cannot catch it.
    let everything = "def reckless():\n    try:\n        misused()\n    except BaseException:\n        return 'caught'";
    repl.define("reckless", "Catch everything.", everything).unwrap();
    assert_eq!(repl.execute("print(reckless())"), Err(Error::Usage("no such input".to_owned())));
    let error = rdm
        .model(scripted(["reckless", "finish"]))
        .call(inputs! { "request" => "Try." })
        .unwrap_err();
    assert_eq!(error, Error::Usage("no such input".to_owned()));
    // It ended the run, and is nothing the function did wrong: the function is on the list, and the REPL goes on.
    assert_eq!(names(&repl), ["careful", "reckless"]);
    assert_eq!(repl.execute("print(careful())").unwrap(), "caught: the lock is jammed");
}

#[test]
fn a_tool_of_the_module_is_a_function_in_the_repl() {
    let world = world();
    let cooled = world.clone();
    // A tool reads what it reads from the request: here an answer of it.
    let cooler = Tool::new("cooler", "Make it cooler by the degrees asked for.", move |call| {
        let degrees: String = call.arg("degrees")?;
        let mut state = cooled.lock();
        state["heat"] -= degrees.parse::<i64>().map_err(|error| error.to_string())?;
        Ok(format!("heat is {}", state["heat"]))
    });
    let repl = Repl::new().tool(&cooler);
    assert_eq!(repl.describe(), "cooler(): Make it cooler by the degrees asked for.");
    repl.define("twice", "Make it cooler, twice.", "def twice():\n    cooler()\n    return cooler()")
        .unwrap();
    let signature = operate().choice("degrees", "By how many degrees does `inputs.request` ask?", ["1", "2", "3"]);
    let rdm = Rdm::new(signature).tool(cooler).source(repl.clone());
    let result = rdm
        .clone()
        .model(pointing(["twice", "finish"], "2"))
        .call(inputs! { "request" => "Cooler.", "world" => &world })
        .unwrap();
    // It ran for the request that pointed at the function, twice.
    assert_eq!(step(&result, 0).1, "heat is 17");
    // It takes nothing of its own, and with no request there is nothing for it to run for.
    assert_eq!(
        failure(repl.execute("cooler(2)")),
        "TypeError: cooler() takes 0 positional arguments but 1 was given"
    );
    assert_eq!(
        failure(repl.execute("cooler()")),
        "RuntimeError: `cooler` is a tool of a module: it runs for a request, and no request is running this code."
    );
    // The name is taken, as any function's of the host is.
    assert_eq!(
        failure(repl.define("cooler", "Cool.", "def cooler():\n    return 1")),
        "`cooler` is already the name of a tool. Give the function another name."
    );
    // A tool whose name Python cannot say cannot be called, so a writer is not told of it until it has one that can.
    let step = Tool::new("tv on", "Turn the TV on.", |_| Ok("tv: now on"));
    assert_eq!(Repl::new().tool(&step).describe(), "");
    let repl = Repl::new().tool(&step.named("tv_on"));
    assert_eq!(repl.describe(), "tv_on(): Turn the TV on.");
}

#[test]
fn an_input_does_not_take_the_name_of_a_function() {
    // The request has an input called `heat`, and so has the REPL a function.
    let repl = Repl::new().function("heat", "How warm it is.", |_| Ok(21));
    repl.define(
        "warm",
        "Say how warm it is.",
        "def warm():\n    return f'{heat()} degrees, asked: {request}'",
    )
    .unwrap();
    repl.define("note", "A note.", "def note():\n    return 'a note'").unwrap();
    let signature = Signature::new("Read.")
        .input("request")
        .input("heat")
        .input("note")
        .input("print")
        .input("observation")
        .tool("next", "What next?");
    let rdm = Rdm::new(signature).source(repl.clone());
    let ask = || inputs! { "request" => "how warm?", "heat" => "unknown", "note" => "none", "print" => "nothing" };
    let result = rdm.clone().model(scripted(["warm", "note", "finish"])).call(ask()).unwrap();
    // The functions kept their names: an input that has the name of one is not a variable.
    assert_eq!(step(&result, 0).1, "21 degrees, asked: how warm?");
    assert_eq!(step(&result, 1).1, "a note");
    assert_eq!(repl.execute("print(heat(), note())").unwrap(), "21 a note");
}

fn limited(limits: Limits) -> Repl {
    let stuck = Shared::new(21);
    let now = stuck.clone();
    Repl::new()
        .function("heat", "How warm it is.", move |_| Ok(now.get()))
        .function("cooler", "A cooler that does nothing.", |_| Ok(()))
        .limits(limits)
}

#[test]
fn written_code_cannot_hang_the_host() {
    let repl = limited(Limits {
        time: Duration::from_millis(100),
        calls: 50,
        recursion: 50,
        memory: 1024 * 1024,
        output: 1000,
        ..Limits::default()
    });
    repl.define("hello", "Say hello.", "def hello():\n    return 'hello'").unwrap();
    repl.execute("kept = 7").unwrap();
    // A loop that does not end is ended by the time.
    let started = std::time::Instant::now();
    let why = failure(repl.execute("while True:\n    pass"));
    assert!(why.starts_with("TimeoutError: "), "{why}");
    assert!(why.ends_with(". The REPL was started again: it holds the functions that were written and nothing else."));
    assert!(started.elapsed() < Duration::from_secs(5));
    // What the interpreter held is not trusted after that: it is made again, with what was written.
    assert_eq!(repl.execute("print(hello())").unwrap(), "hello");
    assert!(failure(repl.execute("print(kept)")).starts_with("NameError"));
    // A loop over a function of the host that changes nothing spends no time of its own: the calls are counted.
    assert_eq!(
        failure(repl.execute("while heat() > 18:\n    cooler()")),
        "RuntimeError: the code called the functions it is given more than 50 times"
    );
    let caught = "try:\n    while True:\n        cooler()\nexcept BaseException:\n    print('caught')";
    assert!(failure(repl.execute(caught)).starts_with("RuntimeError: the code called"));
    // So is whatever else code stops for: a file it asks for, and is refused, over and over.
    let asking = "while True:\n    try:\n        open('x')\n    except PermissionError:\n        pass";
    assert!(failure(repl.execute(asking)).starts_with("RuntimeError: the code called"));
    // A value too large to hold, a recursion that does not return, and more output than is read.
    assert!(failure(repl.execute("x = 'x' * 10 ** 9")).starts_with("MemoryError"));
    assert!(failure(repl.execute("def down(n):\n    return down(n + 1)\ndown(0)")).starts_with("RecursionError"));
    assert!(failure(repl.execute("for _ in range(100):\n    print('x' * 100)")).starts_with("MemoryError"));
    // Nothing of the machine is reached.
    for code in [
        "open('/etc/passwd').read()",
        "import os\nos.getenv('HOME')",
        "import time\ntime.sleep(1)",
        "from pathlib import Path\nPath('/etc/passwd').read_text()",
    ] {
        assert_eq!(
            failure(repl.execute(code)),
            "PermissionError: the REPL has no access to the operating system",
            "{code}"
        );
    }
    assert!(failure(repl.execute("import socket")).starts_with("ModuleNotFoundError"));
    // After all of it the REPL is as it was written.
    assert_eq!(repl.execute("print(hello())").unwrap(), "hello");
}

#[test]
fn what_code_is_held_to_is_counted_for_each_run_and_not_while_the_host_works() {
    let repl = Repl::new()
        .function("tick", "Count one.", |_| Ok(1))
        .function("slow", "Take a while.", |_| {
            std::thread::sleep(Duration::from_millis(300));
            Ok(1)
        })
        .limits(Limits {
            time: Duration::from_millis(100),
            calls: 50,
            ..Limits::default()
        });
    // The calls of one run are not those of the next: a REPL that has run a long while is not used up.
    for _ in 0..5 {
        let counted = repl.execute("n = 0\nfor _ in range(30):\n    n += tick()\nprint(n)");
        assert_eq!(counted.unwrap(), "30");
    }
    // The time is the code's own. What a function of the host takes, a tool that asks a model say, is not counted.
    assert_eq!(repl.execute("print(slow() + slow())").unwrap(), "2");
}

#[test]
fn a_function_that_does_not_end_has_failed() {
    let repl = limited(Limits {
        time: Duration::from_millis(100),
        ..Limits::default()
    });
    repl.define("spin", "Spin.", "def spin():\n    while True:\n        pass").unwrap();
    repl.define("hello", "Say hello.", "def hello():\n    return 'hello'").unwrap();
    let signature = Signature::new("Spin.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let model = scripted(["spin", "hello", "finish"]);
    let result = Rdm::new(signature)
        .source(repl.clone())
        .model(model.clone())
        .call(inputs! { "request" => "Spin." })
        .unwrap();
    assert!(
        step(&result, 0)
            .1
            .starts_with("`spin` failed and was taken off the list. TimeoutError: ")
    );
    // The run goes on, with the functions that work.
    assert_eq!(offers(&model)[1], ["hello", "finish"]);
    assert_eq!(step(&result, 1).1, "hello");
    assert!(repl.written()["spin"].failed.is_some());
}

/// Every way a source can be deep and not nested: each link is one more level of what the
/// parser makes of it, with no bracket opened and left open.
const LINKS: [&str; 10] = [".a", " + 1", "()", "[0]", " - x", " * 2", " @ x", " | x", " << 1", " ** 2"];

/// On a thread whose stack is the size a tool's thread has: what overflows it ends the process.
fn on_a_tool_thread(does: impl FnOnce() + Send + 'static) {
    common::on_a_stack(8 * 1024 * 1024, does);
}

#[test]
fn a_source_that_is_one_long_chain_is_refused_and_ends_nothing() {
    on_a_tool_thread(|| {
        let repl = Repl::new();
        repl.define("hello", "Say hello.", "def hello():\n    return 'hello'").unwrap();
        for link in LINKS {
            // A function a writer wrote: 30,000 links, and no bracket inside another.
            let code = format!("def f():\n    x = 0\n    return x{}", link.repeat(30_000));
            assert_eq!(
                failure(repl.define("f", "A thing.", &code)),
                "What was written is not Python. The source is too deeply nested.",
                "{link}"
            );
            // Code that is run, ten times as long.
            let code = format!("x = 0\nx{}", link.repeat(300_000));
            assert_eq!(failure(repl.execute(&code)), "SyntaxError: Source is too deeply nested", "{link}");
        }
        // A chain on what a bracket closed, which is as deep as the two together.
        let code = format!("x = 0\n(x{}){}", ".a".repeat(30_000), ".a".repeat(30_000));
        assert_eq!(failure(repl.execute(&code)), "SyntaxError: Source is too deeply nested");
        // Nothing was written, and what was there is there.
        assert_eq!(names(&repl), ["hello"]);
        assert_eq!(repl.execute("print(hello())").unwrap(), "hello");
    });
}

#[test]
fn a_saved_function_that_is_one_long_chain_has_failed_and_the_others_run() {
    on_a_tool_thread(|| {
        let deep = format!("def deep():\n    x = 0\n    return x{}", ".a".repeat(100_000));
        let saved = Functions::from([
            ("deep".to_owned(), Written::new("Go deep.", deep)),
            ("hello".to_owned(), Written::new("Say hello.", "def hello():\n    return 'hello'")),
        ]);
        let repl = Repl::new().with_written(saved);
        let signature = Signature::new("Go.")
            .input("request")
            .input("observation")
            .tool("next", "What next?");
        let model = scripted(["deep", "hello", "finish"]);
        let result = Rdm::new(signature)
            .source(repl.clone())
            .model(model.clone())
            .call(inputs! { "request" => "Go." })
            .unwrap();
        assert_eq!(
            step(&result, 0).1,
            "`deep` failed and was taken off the list. SyntaxError: Source is too deeply nested"
        );
        assert_eq!(step(&result, 1).1, "hello");
    });
}

#[test]
fn a_source_that_is_long_and_not_deep_is_run() {
    // What is long is not for that reason deep: ten thousand of a thing, one beside the other.
    let repl = Repl::new();
    let items: Vec<String> = (0..10_000).map(|n| format!("-{n} + len('ab'.strip())")).collect();
    let code = format!("def total():\n    return sum([{}])", items.join(", "));
    repl.define("total", "Add them up.", &code).unwrap();
    assert_eq!(repl.execute("print(total())").unwrap(), (20_000 - 49_995_000i64).to_string());
    let lines: Vec<String> = (0..10_000).map(|n| format!("x = x + {n}")).collect();
    assert_eq!(repl.execute(&format!("x = 0\n{}\nprint(x)", lines.join("\n"))).unwrap(), "49995000");
    let entries: Vec<String> = (0..5_000).map(|n| format!("'k{n}': [{n}, ({n},)]")).collect();
    assert_eq!(
        repl.execute(&format!("d = {{{}}}\nprint(len(d))", entries.join(", "))).unwrap(),
        "5000"
    );
}

#[test]
fn a_saved_function_that_cannot_be_defined_has_failed_and_the_others_run() {
    // A saved map is plain data, and whatever is in it is defined when the REPL starts.
    let saved = Functions::from([
        ("count".to_owned(), Written::new("Count.", "def count():\n    yield 1")),
        ("spin".to_owned(), Written::new("Spin.", "while True:\n    pass")),
        ("hello".to_owned(), Written::new("Say hello.", "def hello():\n    return 'hello'")),
    ]);
    let repl = Repl::new()
        .limits(Limits {
            time: Duration::from_millis(100),
            ..Limits::default()
        })
        .with_written(saved);
    let signature = Signature::new("Count.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let model = scripted(["count", "hello", "finish"]);
    let result = Rdm::new(signature)
        .source(repl.clone())
        .model(model.clone())
        .call(inputs! { "request" => "Count." })
        .unwrap();
    assert_eq!(
        offers(&model),
        [
            vec!["count", "spin", "hello", "finish"],
            vec!["hello", "finish"],
            vec!["hello", "finish"]
        ]
    );
    assert!(
        step(&result, 0)
            .1
            .starts_with("`count` failed and was taken off the list. NotImplementedError")
    );
    assert_eq!(step(&result, 1).1, "hello");
    let written = repl.written();
    assert!(written["spin"].failed.as_ref().unwrap().starts_with("TimeoutError"));
    assert!(written["hello"].failed.is_none());
}

// The notebook's cells

type House = Shared<Map<String, Json>>;

/// The house of the notebook, and the functions it has for a REPL.
fn home(start: Json) -> (House, Repl) {
    let house: House = Shared::new(start.as_object().expect("a house is an object").clone());
    let (switched, cooled, warmed, read) = (house.clone(), house.clone(), house.clone(), house.clone());
    let by = |house: &House, degrees: i64| {
        let mut house = house.lock();
        let heat = house["heat"].as_i64().unwrap_or_default() + degrees;
        house.insert("heat".to_owned(), json!(heat));
        format!("heat: now {heat}")
    };
    let repl = Repl::new()
        .function(
            "switch(device, to)",
            "Set the tv or the lights to on or off, the door to open or closed, or the lock to locked or unlocked.",
            move |args| {
                let (device, to): (String, String) = (args.get("device")?, args.get("to")?);
                let mut house = switched.lock();
                if device == "lock" && to == "locked" && house["door"] == "open" {
                    return Ok("lock: refused, the door is open".to_owned());
                }
                house.insert(device.clone(), json!(to));
                Ok(format!("{device}: now {to}"))
            },
        )
        .function("cooler", "Make it one degree cooler.", move |_| Ok(by(&cooled, -1)))
        .function("warmer", "Make it one degree warmer.", move |_| Ok(by(&warmed, 1)))
        .function("heat", "How warm it is, in degrees.", move |_| Ok(read.lock()["heat"].clone()));
    (house, repl)
}

fn operate_home() -> Signature {
    Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?")
}

const STARTER: &str = include_str!("fixtures/starter_map.json");

#[test]
fn the_starter_map_is_the_one_the_notebook_loads() {
    // The fixture is a copy of the map the notebook's cell reads. Where RDM_SITE_STARTER_MAP names that file, it is the same.
    if let Ok(original) = std::fs::read_to_string(std::env::var("RDM_SITE_STARTER_MAP").unwrap_or_default()) {
        assert_eq!(STARTER, original);
    }
    // Read and written again it is the same data, key for key and in the same order.
    let starter: Json = serde_json::from_str(STARTER).unwrap();
    let written: Functions = serde_json::from_value(starter["written"].clone()).unwrap();
    let (_, repl) = home(json!({"heat": 21}));
    let repl = repl.with_written(written);
    assert_eq!(
        serde_json::to_string(&repl.written()).unwrap(),
        serde_json::to_string(&starter["written"]).unwrap()
    );
}

#[test]
fn starter_a_map_written_before_is_pointed_at_from_the_first_request() {
    let starter: Json = serde_json::from_str(STARTER).unwrap();
    assert_eq!(starter["made_by"]["decision_model"], "~typesafe/jev-latest");
    let written: Functions = serde_json::from_value(starter["written"].clone()).unwrap();
    assert_eq!(written.keys().collect::<Vec<_>>(), ["good_night", "movie_time", "good_morning"]);

    // The functions run in a REPL of this program's own, against this program's house.
    let (house, repl) = home(json!({"tv": "on", "lights": "on", "door": "open", "lock": "unlocked", "heat": 21}));
    let repl = repl.with_written(written);
    // No language model is asked anything: the decision model has only to point.
    let model = scripted(["movie_time", "finish", "good_night", "finish", "good_morning", "finish"]);
    let rdm = Rdm::new(operate_home()).source(repl.clone()).model(model.clone());

    let tasks = ["Let's watch a film.", "We are off to bed.", "Morning!"];
    let mut pointed = Vec::new();
    let mut houses = Vec::new();
    for task in tasks {
        let result = rdm.call(inputs! { "task" => task, "house" => &house }).unwrap();
        pointed.push(result.pointed("next"));
        houses.push(json!(*house.lock()));
    }
    assert_eq!(
        pointed,
        [
            vec!["movie_time", "finish"],
            vec!["good_night", "finish"],
            vec!["good_morning", "finish"]
        ]
    );
    // They were on the map from the first request, each with the description it was saved with.
    let first = &model.requests()[0];
    assert_eq!(offered(first, "next"), ["good_night", "movie_time", "good_morning", "finish"]);
    assert_eq!(
        first.questions["next"].criteria.as_ref().unwrap()["movie_time"],
        "Turns on the TV and turns off the lights for movie time."
    );
    // What each returned is what the next request showed.
    assert_eq!(
        observed(&model),
        [
            "",
            "Turned the TV on and the lights off.",
            "",
            "Turned off the TV and lights, closed and locked the door, and set the heat to 18 degrees.",
            "",
            "Turned the lights on, unlocked the door, and brought the heat up to 21 degrees.",
        ]
    );
    assert_eq!(
        houses,
        [
            json!({"tv": "on", "lights": "off", "door": "open", "lock": "unlocked", "heat": 21}),
            json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked", "heat": 18}),
            json!({"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked", "heat": 21}),
        ]
    );
    // The house ends as the cell prints it.
    assert_eq!(
        as_observation(&json!(*house.lock())),
        r#"{"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked", "heat": 21}"#
    );
    // Nothing was written, and nothing failed.
    assert_eq!(serde_json::to_value(repl.written()).unwrap(), starter["written"]);
}

#[test]
fn repl_what_a_writer_defines_is_pointed_at_from_the_next_request() {
    let task = "Bring the heat down to 18 degrees.";
    let (house, repl) = home(json!({"tv": "off", "lights": "on", "door": "closed", "lock": "locked", "heat": 21}));
    // The writer's side that needs no language model: what it is told it may call.
    assert!(
        repl.describe()
            .ends_with("cooler(): Make it one degree cooler.\nwarmer(): Make it one degree warmer.\nheat(): How warm it is, in degrees.")
    );
    // In the cell a language model writes. Here a function does, and calls `define` as the model would.
    let writing = repl.clone();
    let write = Tool::new(
        "write",
        "Have a new tool written in code and put on this list: nothing listed does what is still needed.",
        move |call| {
            let code = "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return f\"The heat is now {heat()} degrees.\"";
            writing.define_in(call, "cool_to_18", "Bring the heat down to 18 degrees.", code)
        },
    );
    let model = scripted(["write", "cool_to_18", "finish"]);
    let cool = Rdm::new(operate_home()).tool(write).source(repl.clone()).model(model.clone());
    let result = cool.call(inputs! { "task" => task, "house" => &house }).unwrap();

    assert_eq!(result.pointed("next"), ["write", "cool_to_18", "finish"]);
    assert_eq!(house.lock()["heat"], 18);
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["cool_to_18"]);
    // The function was on the map from the request after the one that wrote it.
    assert_eq!(
        offers(&model),
        [
            vec!["write", "finish"],
            vec!["write", "cool_to_18", "finish"],
            vec!["write", "cool_to_18", "finish"]
        ]
    );
    assert_eq!(
        observed(&model),
        [
            "",
            "`cool_to_18` was written and is now on the list. It has not been run, so nothing has changed yet. \
             It does this: Bring the heat down to 18 degrees.",
            "The heat is now 18 degrees.",
        ]
    );
    // Writing changed nothing: the house was as it started when the function was pointed at.
    assert_eq!(shown(&model)[1]["house"]["heat"], 21);
    assert_eq!(shown(&model)[2]["house"]["heat"], 18);
    assert_eq!(
        result.trajectory[0].changes,
        Some(vec![json!({"by": "repl", "did": "define", "name": "cool_to_18"})])
    );
}
