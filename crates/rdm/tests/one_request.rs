//! One request: answered once, and its answers select tools.

mod common;

use std::collections::HashSet;

use common::{Meeting, Say, asked, client, shown};
use rdm::systemone::{Kind, Params};
use rdm::{Opt, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;

fn triage() -> Signature {
    Signature::new("Triage a support ticket.")
        .input("ticket")
        .noul("refund", "Is the customer asking for a refund?")
        .noul("escalate", "Does the ticket need an engineer?")
        .score("frustration", "How frustrated is the customer?", ["calm", "frustrated", "angry"])
}

#[test]
fn a_request_is_answered_once_and_an_output_named_after_a_tool_calls_it() {
    let called = Shared::new(Vec::new());
    let (for_refund, for_escalate) = (called.clone(), called.clone());
    let refund = Tool::new("refund", "Start a refund.", move |call| {
        for_refund.lock().push(("refund", call.arg::<String>("ticket")?));
        Ok("refund started")
    });
    let escalate = Tool::new("escalate", "Page an engineer.", move |call| {
        for_escalate.lock().push(("escalate", call.arg::<String>("ticket")?));
        Ok(())
    });
    let model = client(|_, name, _| match name {
        "refund" => Say::Yes(true),
        "escalate" => Say::Yes(false),
        _ => Say::Level(1),
    });
    let rdm = Rdm::new(triage()).tool(refund).tool(escalate).model(model.clone());
    let result = rdm.call(inputs! { "ticket" => "I was charged twice." }).unwrap();

    // One request: the state and every question, as a System One model takes them. Nothing is asked twice.
    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].state["inputs"], json!({"ticket": "I was charged twice."}));
    let kinds: Vec<(&str, Kind)> = requests[0]
        .questions
        .iter()
        .map(|(name, question)| (name.as_str(), question.kind))
        .collect();
    assert_eq!(
        kinds,
        [("refund", Kind::Noul), ("escalate", Kind::Noul), ("frustration", Kind::Score)]
    );
    // The answers come back typed, with their probabilities, and the yes called its tool.
    let refund = result.noul("refund").unwrap();
    assert_eq!((refund.value, refund.probability, result.yes("escalate")), (true, 0.9, false));
    assert_eq!(called.get(), [("refund", "I was charged twice.".to_owned())]);
    // What the tool returned comes back to the caller, and the call says it did everything it selected.
    assert_eq!(json!(result.observations), json!({"refund": "refund started"}));
    assert_eq!(result.termination_reason, "final");
    // A score is a judgment and calls nothing, whatever it is named.
    assert_eq!(result.score("frustration").unwrap().value, 1.0);
    let escalate = result.noul("escalate").unwrap();
    assert_eq!(
        json!(result.trajectory),
        json!([{
            "depth": 0, "iteration": 0, "parent": null, "tool_name": null,
            "inputs": {"ticket": "I was charged twice."},
            "tokens": 0,
            "model": "jev-test",
            "outputs": {
                "refund": {"value": true, "confidence": refund.confidence, "probability": 0.9},
                "escalate": {"value": false, "confidence": escalate.confidence, "probability": 0.1},
                "frustration": {"value": 1.0, "confidence": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "level": 1},
            },
            "observations": {"refund": "refund started"},
        }])
    );
    // The entry has the keys the Python module gives it, in its order, and no others.
    let entry = json!(result.trajectory[0]);
    let keys: Vec<&str> = entry.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "depth",
            "iteration",
            "parent",
            "tool_name",
            "inputs",
            "tokens",
            "model",
            "outputs",
            "observations"
        ]
    );
    assert_eq!(rdm.tools().names(), ["refund", "escalate"]);
}

#[test]
fn with_inputs_that_fit_it_is_one_request_and_its_answers() {
    let model = client(|_, name, _| match name {
        "refund" => Say::P(0.8),
        "escalate" => Say::P(0.3),
        _ => Say::Level(2),
    });
    let inputs = inputs! { "ticket" => "I was charged twice." };
    let ours = Rdm::new(triage()).model(model.clone()).call(inputs.clone()).unwrap();
    // The same request as the signature alone makes, once: the module is a predict until something is selected.
    let alone = triage().request(&inputs.snapshot().unwrap()).unwrap();
    assert_eq!(model.requests(), [alone]);
    assert_eq!((ours.probability("refund"), ours.yes("escalate")), (Some(0.8), false));
    assert_eq!((ours.score("frustration").unwrap().level, ours.trajectory.len()), (2, 1));
    assert!(ours.observations.is_empty());
}

#[test]
fn the_state_on_the_wire_is_three_keys_and_the_inputs_are_a_numbered_string() {
    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("request")
        .input_with("devices", "dict", "Every device and its state now.")
        .input_described("observation", "What the last step did, if there was one.")
        .tool("next", "What should happen next for `inputs.request`?");
    let model = client(|_, _, _| "finish");
    let tv = Tool::new("tv on", "Set the tv to on.", |_| Ok("tv: now on"));
    let lights = Tool::new("lights off", "", |_| Ok("lights: now off"));
    let rdm = Rdm::new(operate).tool(tv).tool(lights).model(model.clone());
    rdm.call(inputs! { "request" => "Movie time.", "devices" => json!({"tv": "off", "lights": "on"}) })
        .unwrap();
    let request = &model.requests()[0];
    assert_eq!(request.model, "");
    assert_eq!(
        request.state,
        json!({
            "instructions": "Operate a smart home, one step at a time.",
            "input_fields": "1. `request` (str): \n2. `devices` (dict): Every device and its state now.\n3. `observation` (str): What the last step did, if there was one.",
            "inputs": {"request": "Movie time.", "devices": {"tv": "off", "lights": "on"}, "observation": ""},
        })
    );
    let keys: Vec<&String> = request.state.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["instructions", "input_fields", "inputs"]);
    // The pointer is a choice over the tools in the order of the map, `finish` last. An empty description is sent as null.
    assert_eq!(
        json!(request.questions),
        json!({"next": {
            "instructions": "What should happen next for `inputs.request`?",
            "type": "choice",
            "criteria": {"tv on": "Set the tv to on.", "lights off": null, "finish": "Stop: nothing more should be done."},
        }})
    );
    let order: Vec<String> = request.questions["next"].labels();
    assert_eq!(order, ["tv on", "lights off", "finish"]);
}

#[test]
fn a_plain_list_of_labels_is_a_choice_whose_criteria_are_all_null() {
    let act = Signature::new("Operate a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul("switch", "Does `inputs.task` ask for one device to be set to another state?")
        .choice("device", "Which device does `inputs.task` mean?", ["tv", "lights", "door", "lock"])
        .choice("team", "Which team?", [("billing", "Payment issue"), ("sales", "")]);
    let questions = act.questions().unwrap();
    assert_eq!(
        json!(questions["switch"]),
        json!({"instructions": "Does `inputs.task` ask for one device to be set to another state?", "type": "noul"})
    );
    assert_eq!(
        questions["device"].criteria,
        Some(json!({"tv": null, "lights": null, "door": null, "lock": null}))
    );
    assert_eq!(questions["team"].criteria, Some(json!({"billing": "Payment issue", "sales": null})));
    assert_eq!(act.input_fields(), "1. `task` (str): \n2. `house` (dict):");
    // A question with nothing to ask is refused before it is sent.
    let unasked = Signature::parse("question -> found: bool").unwrap();
    assert!(unasked.questions().unwrap_err().to_string().contains("requires a description"));
}

#[test]
fn arguments_come_from_the_other_answers() {
    let home = Signature::new("Carry out a request to a smart home.")
        .input("request")
        .noul("lights", "Is the request a command to the lights?")
        .choice(
            "action",
            "What should be done to the lights?",
            [("on", "turn them on"), ("off", "turn them off")],
        )
        .choice(
            "room",
            "Which lights does the request mean?",
            [("kitchen", ""), ("bedroom", ""), ("house", "every room")],
        )
        .choice(
            "thermostat",
            "How should the temperature change?",
            [
                Opt::new("warmer", ""),
                Opt::new("cooler", ""),
                Opt::none("The request is not about temperature."),
            ],
        );
    let done = Shared::new(Vec::new());
    let (by_lights, by_thermostat) = (done.clone(), done.clone());
    let lights = Tool::new("lights", "", move |call| {
        let (action, room): (String, String) = (call.arg("action")?, call.arg("room")?);
        by_lights.lock().push(format!("lights {action} in {room}"));
        Ok(())
    });
    let thermostat = Tool::new("thermostat", "", move |call| {
        by_thermostat
            .lock()
            .push(format!("thermostat {}", call.arg::<String>("thermostat")?));
        Ok(())
    });
    let answers = Shared::new(json!({"lights": true, "action": "off", "room": "house", "thermostat": "None"}));
    let said = answers.clone();
    let model = client(move |_, name, _| match &said.lock()[name] {
        rdm::Json::Bool(yes) => Say::Yes(*yes),
        other => Say::Pick(other.as_str().unwrap().to_owned()),
    });
    let rdm = Rdm::new(home).tool(lights).tool(thermostat).model(model.clone());
    let result = rdm
        .call(inputs! { "request" => "Turn off all of the lights in the house" })
        .unwrap();
    // Every question was asked at once, before it was known which would matter. The tool the
    // answers selected took its arguments from the answers of the same name.
    assert_eq!(done.get(), ["lights off in house"]);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(
        model.requests()[0].questions["thermostat"].criteria.as_ref().unwrap()["None"],
        "The request is not about temperature."
    );
    // Choosing none of them is an answer with no value, and calls nothing.
    assert_eq!((result.value("thermostat"), result.chosen("thermostat")), (json!(null), None));
    assert_eq!(result.choice("thermostat").unwrap().value, "None");
    assert_eq!(result.chosen("room"), Some("house"));
    // A choice calls the tool of its name, which takes the chosen option by that name.
    answers.update(|answers| {
        answers["lights"] = json!(false);
        answers["thermostat"] = json!("warmer");
    });
    rdm.call(inputs! { "request" => "It is freezing in here" }).unwrap();
    assert_eq!(done.get(), ["lights off in house", "thermostat warmer"]);
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn every_tool_that_is_selected_runs_and_they_run_at_once() {
    let names = ["external_search", "internal_search", "llm", "child"];
    let next = Signature::new("Decide how to look for an answer.")
        .input("query")
        .noul("external_search", "Would a web search help?")
        .noul("internal_search", "Would the internal wiki help?")
        .noul("llm", "Would rewording the query help?")
        .noul("child", "Should the query be broken into parts?");
    let meeting = Meeting::of(4);
    let threads = Shared::new(HashSet::new());
    let make = |name: &'static str| {
        let (meeting, threads) = (meeting.clone(), threads.clone());
        Tool::new(name, "", move |call| {
            // Each waits for the other three: this only returns if all four are running together.
            if !meeting.meet() {
                return Err("the others never came".into());
            }
            threads.lock().insert(std::thread::current().id());
            Ok(format!("{name} found nothing for {}", call.arg::<String>("query")?))
        })
    };
    let rdm = Rdm::new(next).tools_from(names.map(make)).model(client(|_, _, _| true));
    let result = rdm.call(inputs! { "query" => "q" }).unwrap();
    // Four yeses, four calls, and their results in the order of the outputs that selected them.
    assert_eq!((threads.lock().len(), result.trajectory.len()), (4, 1));
    let expected: Vec<(String, String)> = names
        .iter()
        .map(|name| (name.to_string(), format!("{name} found nothing for q")))
        .collect();
    let observed: Vec<(String, String)> = result
        .observations
        .iter()
        .map(|(name, seen)| (name.clone(), seen.as_str().unwrap().to_owned()))
        .collect();
    assert_eq!(observed, expected);
}

#[test]
fn a_score_never_calls_a_tool_and_a_no_does_not() {
    let ran = Shared::new(0);
    let counted = ran.clone();
    let frustration = Tool::new("frustration", "", move |_| {
        counted.update(|ran| *ran += 1);
        Ok(())
    });
    let rdm = Rdm::new(triage()).tool(frustration).model(client(
        |_, name, _| if name == "frustration" { Say::Level(2) } else { Say::Yes(false) },
    ));
    let result = rdm.call(inputs! { "ticket" => "t" }).unwrap();
    assert_eq!(
        (ran.get(), result.observations.len(), result.termination_reason.as_str()),
        (0, 0, "final")
    );
}

#[test]
fn thresholds_are_parameters_of_the_module_by_output_name() {
    let refund = Tool::new("refund", "", |_| Ok("refund started"));
    let model = client(|_, name, _| match name {
        "refund" => Say::P(0.6),
        "escalate" => Say::P(0.2),
        _ => Say::Level(0),
    });
    let rdm = Rdm::new(triage()).tool(refund).model(model.clone());
    assert_eq!(
        json!(rdm.call(inputs! { "ticket" => "t" }).unwrap().observations),
        json!({"refund": "refund started"})
    );
    // The model said 0.6. The threshold is applied here, and the tool is no longer called.
    rdm.set_params("refund", Params::threshold(0.7));
    let result = rdm.call(inputs! { "ticket" => "t" }).unwrap();
    assert_eq!(
        (result.probability("refund"), result.yes("refund"), result.observations.len()),
        (Some(0.6), false, 0)
    );
    assert!(!json!(model.requests()).to_string().contains("threshold"));
    assert_eq!(rdm.params_of("refund"), Some(Params::threshold(0.7)));
    // A score's cuts move the level, not the value.
    let rdm = rdm.params("frustration", Params::cuts([0.1, 0.2]));
    let score = rdm.call(inputs! { "ticket" => "t" }).unwrap().score("frustration").unwrap().clone();
    assert_eq!((score.value, score.level), (0.0, 0));
    // A number that cannot be one is a wrong use of the module, not a failure of the model.
    rdm.set_params("refund", Params::threshold(1.5));
    assert!(rdm.call(inputs! { "ticket" => "t" }).unwrap_err().is_usage());
    assert_eq!(rdm.unset_params("refund"), Some(Params::threshold(1.5)));
    assert!(rdm.call(inputs! { "ticket" => "t" }).unwrap().yes("refund"));
}

#[test]
fn what_cannot_be_used_is_refused_when_the_module_is_built_or_called() {
    for signature in [
        "state, question -> found: bool",
        "question -> trajectory: bool",
        "question -> observations: bool",
    ] {
        let rdm = Rdm::new(Signature::parse(signature).unwrap());
        assert!(rdm.check().unwrap_err().to_string().contains("reserved by RDM"));
        assert!(rdm.call(inputs! { "question" => "q" }).unwrap_err().is_usage());
    }
    let noop = || Tool::new("descend", "", |_| Ok(()));
    let twice = Rdm::new(triage()).tool(noop()).tool(noop());
    assert!(twice.check().unwrap_err().to_string().contains("Duplicate tool name 'descend'"));
    assert!(
        Rdm::new(triage())
            .max_predictor_calls(0)
            .check()
            .unwrap_err()
            .to_string()
            .contains("at least 1")
    );
    assert!(
        Rdm::new(triage())
            .max_iters(0)
            .check()
            .unwrap_err()
            .to_string()
            .contains("at least 1")
    );
    let rdm = Rdm::new(triage()).model(client(|_, _, _| true));
    assert!(rdm.check().is_ok());
    assert!(rdm.call(inputs! {}).unwrap_err().to_string().contains("The call has inputs []"));
    let error = rdm.call(inputs! { "ticket" => "t", "extra" => 1 }).unwrap_err();
    assert!(
        error.is_usage()
            && error
                .to_string()
                .contains("The call has inputs ['extra', 'ticket']; its signature takes ['ticket']")
    );
    // A module with no model says so when it has something to ask.
    let error = Rdm::new(triage()).call(inputs! { "ticket" => "t" }).unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("no decision model"));
}

#[test]
fn a_signature_can_be_written_and_edited() {
    let written = Signature::parse("task, house: dict, notes: list[str] -> switch, device: Literal['tv', \"lights\"], next: Tool").unwrap();
    assert_eq!(
        written.input_fields(),
        "1. `task` (str): \n2. `house` (dict): \n3. `notes` (list[str]):"
    );
    assert_eq!(written.outputs().keys().collect::<Vec<_>>(), ["switch", "device", "next"]);
    assert_eq!(
        written
            .output("device")
            .unwrap()
            .options()
            .unwrap()
            .iter()
            .map(|option| option.label.as_str())
            .collect::<Vec<_>>(),
        ["tv", "lights"]
    );
    assert_eq!(written.pointer(), Some("next"));
    assert!(Signature::parse("no arrow").is_err() && Signature::parse("a -> b: float").is_err());

    // What a fit does to a signature: other options for an output, an output taken away, one put after the others.
    let edited = written
        .clone()
        .with_options("device", [("door", "The front door.")])
        .without("next")
        .noul("needed", "Is it needed?");
    assert_eq!(edited.outputs().keys().collect::<Vec<_>>(), ["switch", "device", "needed"]);
    assert_eq!(
        edited.output("device").unwrap().options().unwrap()[0].description,
        "The front door."
    );
    assert_eq!(edited.clone().with_instructions("Other.").instructions(), "Other.");
    // A mark is a fit's own, and travels with the output.
    let marked = edited.with_output("needed", rdm::Output::noul("Is it needed?").mark("rdm_fit", true));
    assert_eq!(marked.output("needed").unwrap().marked("rdm_fit"), Some(&json!(true)));
    assert_eq!(asked(&client(|_, _, _| true)).len(), 0);
    assert!(shown(&client(|_, _, _| true)).is_empty());
}

#[test]
fn an_input_is_shown_though_an_output_of_its_name_is_answered_without_asking() {
    // Where a listing is, and where it goes next: one name for the input and the output.
    let walk = |options: &[&str]| {
        Signature::new("Classify a listing.")
            .input("listing")
            .input("category")
            .choice(
                "category",
                "Which category under `inputs.category` is `inputs.listing`?",
                options.to_vec(),
            )
            .noul("sure", "Is `inputs.listing` in `inputs.category`?")
    };
    let given = json!({"listing": "a cat bed", "category": "Animals"});
    for options in [&["Beds"][..], &["Beds", "Bowls"]] {
        let model = client(|_, name, _| match name {
            "category" => Say::Pick("Beds".to_owned()),
            _ => Say::Yes(true),
        });
        let rdm = Rdm::new(walk(options)).model(model.clone());
        let result = rdm.call(inputs! { "listing" => "a cat bed", "category" => "Animals" }).unwrap();
        // With one option the category is not asked. What the request shows is the same either way,
        // and it is what the trajectory says it showed.
        let request = &model.requests()[0];
        assert_eq!(request.questions.contains_key("category"), options.len() > 1);
        assert_eq!(request.state["inputs"], given, "{options:?}");
        assert_eq!(
            request.state["input_fields"], "1. `listing` (str): \n2. `category` (str):",
            "{options:?}"
        );
        assert_eq!(result.trajectory[0].inputs, given);
        assert_eq!(result.choice("category").unwrap().value, "Beds");
    }
}
