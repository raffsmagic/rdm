//! Behaviour the Python module's tests name, or its reviews found, set down one at a time.

mod common;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{Say, client, scripted};
use rdm::systemone::{Answer, Kind, Params, Question, Scripted};
use rdm::{Ask, Call, Entry, Error, Inputs, Json, Opt, Rdm, Reason, Shared, Signature, Source, Tool, inputs};
use serde_json::json;

/// Says yes to every noul, and points at or chooses what `pick` says.
fn yes_and(pick: &'static str) -> Arc<Scripted> {
    client(move |_: &Json, _: &str, question: &Question| match question.kind {
        Kind::Noul => Say::Yes(true),
        Kind::Choice => Say::Pick(pick.to_owned()),
        Kind::Score => Say::Level(0),
    })
}

fn steps() -> Signature {
    Signature::new("Take steps.").input("observation").tool("next", "What next?")
}

#[test]
fn a_copy_keeps_the_parameters_and_the_tools_and_is_a_module_of_its_own() {
    let triage = Signature::new("Triage a support ticket.")
        .input("ticket")
        .noul("refund", "Is the customer asking for a refund?");
    let rdm = Rdm::new(triage)
        .tool(Tool::new("refund", "", |_| Ok("refund started")))
        .max_depth(7)
        .model(Scripted::new(|_| [("refund", Answer::noul(0.6))]));
    rdm.set_params("refund", Params::threshold(0.8));
    let copy = rdm.copy();
    assert_eq!(copy.params_of("refund"), Some(Params::threshold(0.8)));
    assert_eq!(copy.tools().names(), ["refund"]);
    // The copy decides with the threshold it was copied with.
    let result = copy.call(inputs! { "ticket" => "t" }).unwrap();
    assert!(!result.yes("refund") && result.observations.is_empty());
    // Changing the copy leaves the original as it was, parameters and tools.
    copy.unset_params("refund");
    copy.tools().remove("refund");
    assert_eq!(rdm.params_of("refund"), Some(Params::threshold(0.8)));
    assert_eq!(rdm.tools().names(), ["refund"]);
    assert!(copy.call(inputs! { "ticket" => "t" }).unwrap().yes("refund"));
    assert!(!rdm.call(inputs! { "ticket" => "t" }).unwrap().yes("refund"));
}

#[test]
fn a_tool_is_told_which_turn_of_the_loop_it_is_in_and_where_its_request_is() {
    let step = Tool::new("step", "", |call| Ok(format!("turn {} at {}", call.iteration(), call.index())));
    let rdm = Rdm::new(steps()).tool(step).model(scripted(["step", "step", "finish"]));
    let result = rdm.call(inputs! {}).unwrap();
    assert_eq!(
        result.observations["next"],
        json!([["step", "turn 0 at 0"], ["step", "turn 1 at 1"], ["finish", null]])
    );
}

#[test]
fn a_tool_named_by_an_output_beside_the_pointer_runs_at_every_turn() {
    // Every turn asks every question again, so a tool an output names runs at every turn. What
    // the call returns for it is the first turn's; each turn's own is on its entry.
    let logged = Shared::new(0);
    let counted = logged.clone();
    let log = Tool::new("log", "", move |_| Ok(counted.update(|n| std::mem::replace(n, *n + 1))));
    let step = Tool::new("step", "", |call| Ok(format!("turn {}", call.iteration())));
    let script = Mutex::new(VecDeque::from(["step", "step", "finish"]));
    let model = client(move |_, name, _| match name {
        "log" => Say::Yes(true),
        _ => Say::Pick(script.lock().unwrap().pop_front().expect("the script has run out").to_owned()),
    });
    let rdm = Rdm::new(steps().noul("log", "Should this be logged?"))
        .tool(log)
        .tool(step)
        .model(model);
    let result = rdm.call(inputs! {}).unwrap();
    assert_eq!(logged.get(), 3);
    assert_eq!(result.observations["log"], 0);
    assert_eq!(
        result.trajectory.iter().map(|entry| json!(entry.observations)).collect::<Vec<_>>(),
        [
            json!({"log": 0, "next": "turn 0"}),
            json!({"log": 1, "next": "turn 1"}),
            json!({"log": 2})
        ]
    );
}

#[test]
fn the_same_state_under_another_question_is_another_request() {
    // What makes a request a repeat is all of it: the instructions, each question as it is
    // worded, and each input's type, as well as the options, the module and the state.
    let wait =
        |instructions: &str, question: &str, door: &str| Signature::new(instructions).input_typed("door", door).noul("wait", question);
    let others = [
        wait("Wait.", "Is `inputs.door` locked?", "str"),
        wait("Wait longer.", "Is `inputs.door` shut?", "str"),
        wait("Wait.", "Is `inputs.door` shut?", "int"),
    ];
    for other in others {
        let again = Tool::new("wait", "", move |call| {
            call.sub_with(Ask::new(inputs! { "door" => 1 }).signature(other.clone()).answers_only())
        });
        let model = yes_and("");
        let result = Rdm::new(wait("Wait.", "Is `inputs.door` shut?", "str"))
            .tool(again)
            .model(model.clone())
            .call(inputs! { "door" => 1 })
            .unwrap();
        assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::Final));
    }
    // And the same of all of them is the same request: it is not asked again.
    let same = wait("Wait.", "Is `inputs.door` shut?", "str");
    let again = Tool::new("wait", "", move |call| {
        call.sub_with(Ask::new(inputs! { "door" => 1 }).signature(same.clone()).answers_only())
    });
    let model = yes_and("");
    let result = Rdm::new(wait("Wait.", "Is `inputs.door` shut?", "str"))
        .tool(again)
        .model(model.clone())
        .call(inputs! { "door" => 1 })
        .unwrap();
    assert_eq!((model.requests().len(), result.termination_reason), (1, Reason::Repeat));
}

#[test]
fn a_pointer_asked_for_its_answer_only_points_and_nothing_is_run() {
    let ran = Shared::new(0);
    let counted = ran.clone();
    let step = Tool::new("step", "", move |_| Ok(counted.update(|n| std::mem::replace(n, *n + 1))));
    let rdm = Rdm::new(steps()).tool(step).model(yes_and("step"));
    let result = rdm.call_with(Ask::new(inputs! {}).max_depth(0)).unwrap();
    assert_eq!(
        (
            ran.get(),
            result.chosen("next"),
            result.observations["next"].clone(),
            result.termination_reason
        ),
        (0, Some("step"), json!([]), Reason::Final)
    );
}

#[test]
fn an_observation_the_caller_gives_is_what_the_first_turn_shows() {
    let model = scripted(["finish"]);
    Rdm::new(steps())
        .tool(Tool::new("step", "", |_| Ok(())))
        .model(model.clone())
        .call(inputs! { "observation" => "the door was open" })
        .unwrap();
    assert_eq!(model.requests()[0].state["inputs"]["observation"], "the door was open");
}

#[test]
fn a_call_keeps_to_the_limit_of_the_run_it_is_in_whatever_it_asks_for() {
    let count = Signature::new("Count.").input_typed("n", "int").noul("again", "Again?");
    let again = Tool::new("again", "", |call| {
        let n: i64 = call.arg("n")?;
        Ok(call.sub_with(Ask::new(inputs! { "n" => n + 1 }).max_depth(5))?.termination_reason)
    });
    let model = yes_and("");
    let rdm = Rdm::new(count).tool(again).max_depth(1).model(model.clone());
    let result = rdm.call(inputs! { "n" => 0 }).unwrap();
    assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::MaxDepth));
    assert_eq!(result.observations["again"], "max_depth");
    // Nor at the call that starts the run.
    let result = rdm.call_with(Ask::new(inputs! { "n" => 0 }).max_depth(9)).unwrap();
    assert_eq!((result.trajectory.len(), result.termination_reason), (2, Reason::MaxDepth));
}

/// Another module with a loop of its own, which allows itself `own` levels, called from a tool
/// with `asked` levels allowed for the call. Returns how often its tool ran, and the run.
fn ticked(own: usize, asked: usize) -> (usize, rdm::Prediction) {
    let ticks = Arc::new(AtomicUsize::new(0));
    let counted = ticks.clone();
    let tick = Tool::new("tick", "", move |_| {
        Ok(format!("tick {}", counted.fetch_add(1, Ordering::SeqCst) + 1))
    });
    let signature = Signature::new("Tick.")
        .input_typed("n", "int")
        .input("observation")
        .tool("next", "Next?");
    let inner = Rdm::new(signature).tool(tick).max_depth(own).max_iters(5).model(yes_and("tick"));
    let count = Tool::new("count", "", move |call| {
        let n: i64 = call.arg("n")?;
        let ask = Ask::new(inputs! { "n" => n });
        // `usize::MAX` stands for a call that says nothing of its depth.
        let ask = if asked == usize::MAX { ask } else { ask.max_depth(asked) };
        Ok(call.on(&inner).sub_with(ask)?.termination_reason)
    });
    let outer = Rdm::new(Signature::new("Begin.").input_typed("n", "int").noul("count", "Start?"))
        .tool(count)
        .model(yes_and(""));
    let result = outer.call(inputs! { "n" => 1 }).unwrap();
    (ticks.load(Ordering::SeqCst), result)
}

#[test]
fn the_depth_a_call_asked_for_is_its_limit_at_every_turn_of_its_loop() {
    // The rule: a call has one limit. What it asked for is in the place of the module's own,
    // and holds for the first turn of its loop and for every turn after it. Only another
    // module's loop, asked for more levels than the module allows itself, can tell: the Python
    // module keeps what was asked for the first turn and goes back to the module's own for the
    // rest, so there its tool runs once and the second turn is a leaf.
    let (ticks, result) = ticked(0, 1);
    assert_eq!((ticks, result.termination_reason), (5, Reason::MaxIters));
    assert_eq!(result.observations["count"], "max_iters");
    let entries: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.iteration, entry.skipped))
        .collect();
    let turn = |iteration: usize| (1, iteration, None);
    assert_eq!(
        entries,
        [
            (0, 0, None),
            turn(0),
            turn(1),
            turn(2),
            turn(3),
            turn(4),
            (1, 5, Some(Reason::MaxIters))
        ]
    );
    // Asked for fewer levels than it has, or for more, it is the same: every turn runs its tool.
    for asked in [1, 9] {
        let (ticks, result) = ticked(2, asked);
        assert_eq!((ticks, result.termination_reason), (5, Reason::MaxIters), "asked for {asked}");
        assert_eq!(result.trajectory.len(), 7);
    }
    // Asked for none, it is answered at its first turn and runs nothing, which is no limit reached.
    let (ticks, result) = ticked(2, 0);
    assert_eq!((ticks, result.termination_reason, result.trajectory.len()), (0, Reason::Final, 2));
    // And with no more said than its inputs, the module's own limit is the one: here it runs no tools.
    let (ticks, result) = ticked(0, usize::MAX);
    assert_eq!(
        (ticks, result.termination_reason, result.trajectory.len()),
        (0, Reason::MaxDepth, 2)
    );
}

#[test]
fn a_batch_shows_each_of_its_requests_the_state_as_it_was_when_the_batch_was_made() {
    // Two sub-calls of one batch, asked one after the other on one thread, and the first one's
    // tool changes what both show. The second was made when the first was: it is asked about
    // the world as it was then, and its entry says so. What a request is asked, what is
    // recorded of it, and what makes it a repeat are one reading. (The Python module records
    // the same reading, and shows its model the world as the first one's tool left it.)
    let world = Shared::new(json!({"t": 0}));
    let model = client(|inputs, name, _| name == "fan" || inputs["world"]["t"] == 0);
    let changed = world.clone();
    let bump = Tool::new("bump", "", move |_| {
        Ok(changed.update(|world| {
            world["t"] = json!(world["t"].as_i64().unwrap_or(0) + 1);
            world["t"].clone()
        }))
    });
    let fan = Tool::new("fan", "", |call| {
        let look = Signature::new("Look.")
            .input("who")
            .input_typed("world", "dict")
            .noul("bump", "Is `inputs.world.t` zero?");
        let world = call.input("world")?;
        let asks = ["first", "second"].map(|who| Ask::new(inputs! { "who" => who, "world" => &world }).signature(look.clone()));
        let below = call.batch(asks)?;
        Ok(below.iter().map(|prediction| prediction.observations.clone()).collect::<Vec<_>>())
    });
    let fan_out = Signature::new("Fan.")
        .input("who")
        .input_typed("world", "dict")
        .noul("fan", "Fan out?");
    let rdm = Rdm::new(fan_out).tool(fan).tool(bump).threads(1).model(model.clone());
    let result = rdm.call(inputs! { "who" => "root", "world" => &world }).unwrap();
    let shown: Vec<(Json, Json)> = common::shown(&model)
        .iter()
        .map(|inputs| (inputs["who"].clone(), inputs["world"]["t"].clone()))
        .collect();
    assert_eq!(
        shown,
        [(json!("root"), json!(0)), (json!("first"), json!(0)), (json!("second"), json!(0))]
    );
    let recorded: Vec<(Json, Json)> = result
        .trajectory
        .iter()
        .map(|entry| (entry.inputs["world"]["t"].clone(), entry.value("bump")))
        .collect();
    assert_eq!(recorded[1..], [(json!(0), json!(true)), (json!(0), json!(true))]);
    assert_eq!(result.observations["fan"], json!([{"bump": 1}, {"bump": 2}]));
    assert_eq!(world.get(), json!({"t": 2}));
    // Made one at a time, each is asked about the world as the one before left it.
    world.set(json!({"t": 0}));
    let one_by_one = Tool::new("fan", "", |call| {
        let look = Signature::new("Look.")
            .input("who")
            .input_typed("world", "dict")
            .noul("bump", "Is `inputs.world.t` zero?");
        let mut seen = Vec::new();
        for who in ["first", "second"] {
            let ask = Ask::new(inputs! { "who" => who, "world" => call.input("world")? }).signature(look.clone());
            seen.push(call.sub_with(ask)?.observations);
        }
        Ok(seen)
    });
    rdm.tools().insert(one_by_one);
    let result = rdm.call(inputs! { "who" => "root", "world" => &world }).unwrap();
    assert_eq!(result.observations["fan"], json!([{"bump": 1}, {}]));
    assert_eq!(result.trajectory[2].inputs["world"], json!({"t": 1}));
}

#[test]
fn a_signature_given_to_a_call_may_not_take_a_reserved_name() {
    let own = Signature::new("Ask.").input("question").noul("found", "Found?");
    let other = Signature::new("Ask.").input("question").noul("trajectory", "Found?");
    let model = yes_and("");
    let rdm = Rdm::new(own).model(model.clone());
    let error = rdm.call_with(Ask::new(inputs! { "question" => "q" }).signature(other)).unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("no field may be named any of"));
    assert!(model.requests().is_empty());
}

#[test]
fn a_pointer_a_limit_stopped_has_no_answer_and_an_empty_list_of_steps() {
    let begin = Signature::new("Begin.").input("task").noul("begin", "Begin?");
    let start = Tool::new("begin", "", |call| call.sub_with(Ask::new(inputs! {}).signature(steps())));
    let rdm = Rdm::new(begin)
        .tool(start)
        .tool(Tool::new("step", "", |_| Ok(())))
        .max_predictor_calls(1)
        .model(yes_and("step"));
    let result = rdm.call(inputs! { "task" => "t" }).unwrap();
    assert_eq!(
        result.observations["begin"],
        json!({"next": null, "observations": {"next": []}, "termination_reason": "max_predictor_calls"})
    );
    assert_eq!(result.termination_reason, Reason::MaxPredictorCalls);
}

#[test]
fn a_fit_is_not_given_a_request_that_has_nothing_to_ask() {
    let seen = Shared::new(0);
    let noted = seen.clone();
    let fit = move |signature: Signature, inputs: Inputs, _: &Call| {
        noted.update(|n| *n += 1);
        Ok((signature, inputs))
    };
    let one = Signature::new("Leave.").input("room").choice("go", "Which way?", ["north"]);
    let model = yes_and("north");
    let rdm = Rdm::new(one).fit_fn(fit).model(model.clone());
    let result = rdm.call(inputs! { "room" => "hall" }).unwrap();
    assert_eq!((seen.get(), model.requests().len(), result.chosen("go")), (0, 0, Some("north")));
}

#[test]
fn a_fit_that_fails_ends_the_run_at_the_root_and_is_an_observation_under_a_tool() {
    // A fit is not a tool: nothing makes an observation of what it fails with where it is given
    // the run's first request.
    let broken = |_: Signature, _: Inputs, _: &Call| -> rdm::Result<(Signature, Inputs)> { Err("the fit broke".into()) };
    let ask = Signature::new("Ask.").input_typed("n", "int").noul("again", "Again?");
    let rdm = Rdm::new(ask.clone()).fit_fn(broken).model(yes_and(""));
    assert_eq!(rdm.call(inputs! { "n" => 0 }).unwrap_err(), Error::Tool("the fit broke".into()));
    // Under a tool it is a failure inside that tool, and so that tool's observation.
    let deeper = |signature: Signature, inputs: Inputs, call: &Call| -> rdm::Result<(Signature, Inputs)> {
        if call.depth() == 1 {
            return Err("the fit broke".into());
        }
        Ok((signature, inputs))
    };
    let again = Tool::new("again", "", |call| call.sub(inputs! { "n" => 1 }));
    let result = Rdm::new(ask)
        .tool(again)
        .fit_fn(deeper)
        .model(yes_and(""))
        .call(inputs! { "n" => 0 })
        .unwrap();
    assert_eq!(result.observations["again"], "Execution error in again: the fit broke");
}

#[test]
fn weights_set_for_the_pointer_are_not_applied_to_it() {
    // Parameters are for an answer space, and the pointer's is the map as it is at each request:
    // weights given under its name were fitted to no one set of tools, and are left out.
    let rdm = Rdm::new(steps())
        .tool(Tool::new("step", "", |_| Ok("stepped")))
        .max_iters(2)
        .model(Scripted::new(|_| [("next", Answer::choice([("step", 0.6), ("finish", 0.4)]))]));
    rdm.set_params("next", Params::weights([("step", 0.0), ("finish", 1.0)]));
    let result = rdm.call(inputs! {}).unwrap();
    assert_eq!(result.chosen("next"), Some("step"));
    assert_eq!(
        (result.pointed("next"), result.termination_reason),
        (vec!["step".to_owned(); 2], Reason::MaxIters)
    );
}

#[test]
fn an_output_with_one_option_needs_no_description() {
    // It is not asked, so there is no question to word.
    let one = Signature::parse("room -> go: Literal['north']").unwrap();
    let model = yes_and("north");
    let result = Rdm::new(one).model(model.clone()).call(inputs! { "room" => "hall" }).unwrap();
    assert_eq!((result.chosen("go"), model.requests().len()), (Some("north"), 0));
}

#[test]
fn choosing_none_reads_as_null_from_the_trajectory_as_it_does_from_the_prediction() {
    let home = Signature::new("Carry out a request.").input("request").choice(
        "thermostat",
        "How should the temperature change?",
        [Opt::new("warmer", ""), Opt::none("The request is not about temperature.")],
    );
    let rdm = Rdm::new(home.clone()).model(yes_and("None"));
    let result = rdm.call(inputs! { "request" => "Lights off." }).unwrap();
    assert_eq!((result.value("thermostat"), result.chosen("thermostat")), (json!(null), None));
    let entry = &result.trajectory[0];
    assert_eq!(entry.value("thermostat"), json!(null));
    assert!(entry.chose_none("thermostat"));
    // Written, the value is null, as the Python module writes it: a reader of the run can tell
    // choosing none from an option that happens to be labelled "None".
    let written = json!(result);
    let expected = json!({"confidence": 1.0, "value": null, "probabilities": {"warmer": 0.0, "None": 1.0}});
    assert_eq!(written["trajectory"][0]["outputs"]["thermostat"], expected);
    assert_eq!(written["thermostat"], expected);
    // The probabilities are still there, by label.
    assert_eq!(
        entry.outputs.as_ref().unwrap()["thermostat"].as_choice().unwrap().probabilities["None"],
        1.0
    );
    // Read back, it is the same entry, and writes the same.
    let read: Entry = serde_json::from_value(json!(entry)).unwrap();
    assert_eq!(&read, entry);
    assert_eq!(json!(read), json!(entry));
    // An option that is only labelled "None" is a label like any other.
    let labelled = Signature::new("Carry out a request.").input("request").choice(
        "thermostat",
        "How should the temperature change?",
        ["warmer", "None"],
    );
    let result = Rdm::new(labelled)
        .model(yes_and("None"))
        .call(inputs! { "request" => "r" })
        .unwrap();
    assert_eq!(
        (result.value("thermostat"), result.trajectory[0].value("thermostat")),
        (json!("None"), json!("None"))
    );
    assert_eq!(json!(result)["trajectory"][0]["outputs"]["thermostat"]["value"], "None");
    // A sub-call's prediction, which reaches the tool above as plain data, says null too, and a
    // repeat of a request that chose none still does.
    let again = Tool::new("again", "", |call| call.sub(inputs! { "request" => "Lights off." }));
    let result = Rdm::new(home.clone().noul("again", "Ask again?"))
        .tool(again)
        .model(yes_and("None"))
        .call(inputs! { "request" => "Lights off." })
        .unwrap();
    assert_eq!(result.termination_reason, Reason::Repeat);
    assert!(result.trajectory[1].repeat && result.trajectory[1].chose_none("thermostat"));
    assert_eq!(result.observations["again"]["thermostat"]["value"], json!(null));
    let ask_home = Tool::new("thermostat", "", move |call| {
        call.sub_with(Ask::new(inputs! { "request" => "Lights off." }).signature(home.clone()))
    });
    let asked = Signature::new("Ask.")
        .input("request")
        .noul("thermostat", "Ask about the temperature?");
    let result = Rdm::new(asked)
        .tool(ask_home)
        .model(yes_and("None"))
        .call(inputs! { "request" => "r" })
        .unwrap();
    assert_eq!(result.observations["thermostat"]["thermostat"]["value"], json!(null));
    assert!(result.trajectory[1].chose_none("thermostat") && !result.trajectory[0].chose_none("thermostat"));
}

#[test]
fn what_a_tool_returned_is_shown_as_python_writes_json() {
    // python3 -c 'import json; print(json.dumps([1e-05, 1.5e-07, 1e16, 1.5e300, 0.1, 100.0]))'
    let shown = |value: Json| rdm::as_observation(&value);
    assert_eq!(
        shown(json!([1e-05, 1.5e-07, 1e16, 1.5e300, 0.1, 100.0])),
        "[1e-05, 1.5e-07, 1e+16, 1.5e+300, 0.1, 100.0]"
    );
    // The exponent begins below 0.0001 and at 1e16, as Python's does.
    assert_eq!(
        shown(json!([0.0001, 0.00012345, 0.00009999, 1e-10])),
        "[0.0001, 0.00012345, 9.999e-05, 1e-10]"
    );
    assert_eq!(
        shown(json!([1e15, 9999999999999998.0, 1.2345e16, 1e22, 1.7976931348623157e308])),
        "[1000000000000000.0, 9999999999999998.0, 1.2345e+16, 1e+22, 1.7976931348623157e+308]"
    );
    assert_eq!(
        shown(json!([0.30000000000000004, -0.0, -2.5, -1e-7, 5e-324, 123456.789])),
        "[0.30000000000000004, -0.0, -2.5, -1e-07, 5e-324, 123456.789]"
    );
    // Whole numbers are written as they are, and everything else as before.
    assert_eq!(
        shown(json!({"n": 3, "big": 18446744073709551615u64, "low": -7, "ok": true, "none": null, "é": ["a", 1.5]})),
        r#"{"n": 3, "big": 18446744073709551615, "low": -7, "ok": true, "none": null, "é": ["a", 1.5]}"#
    );
    // And it is what the next turn is shown.
    let model = scripted(["weigh", "finish"]);
    let rdm = Rdm::new(steps())
        .tool(Tool::new("weigh", "", |_| Ok(json!({"grams": 0.00002}))))
        .model(model.clone());
    rdm.call(inputs! {}).unwrap();
    assert_eq!(model.requests()[1].state["inputs"]["observation"], r#"{"grams": 2e-05}"#);
}

thread_local! {
    /// Whether the next reading of the shelf on this thread is a slow one.
    static SLOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A shelf of names, each a tool, which one thread reads slowly: it comes back some time
/// later with what was on the shelf when it looked.
struct Shelf {
    names: Shared<Vec<String>>,
    seen: Shared<Vec<Vec<String>>>,
}

impl Source for Shelf {
    fn offers(&self) -> Vec<Tool> {
        let names = self.names.get();
        if SLOW.replace(false) {
            std::thread::sleep(Duration::from_millis(300));
        }
        let tool = |name: String| {
            let seen = self.seen.clone();
            Tool::new(name, "", move |call| {
                if call.arg::<String>("who")? == "second" {
                    // Long after this request's own reading of the shelf put `new` on the map.
                    std::thread::sleep(Duration::from_millis(500));
                    seen.lock().push(call.tools().names());
                }
                Ok("done")
            })
        };
        names.into_iter().map(tool).collect()
    }
}

#[test]
fn a_source_read_slowly_does_not_take_a_newer_name_off_the_map() {
    // Two sub-calls at once. The first reads the shelf and is slow to come back with what it
    // saw. Meanwhile the shelf gains a name, and the second reads it. What the first read is
    // older: put on the map after the second's, it would take the new name off again.
    let names = Shared::new(vec!["old".to_owned()]);
    let seen: Shared<Vec<Vec<String>>> = Shared::default();
    let adds = names.clone();
    let model = client(move |inputs, name, _| {
        match inputs["who"].as_str() {
            // The first sub-call reads the shelf next, and that reading is the slow one.
            Some("first") => SLOW.set(true),
            // The shelf gains a name while it does, and the second sub-call reads it at once.
            Some("second") => {
                std::thread::sleep(Duration::from_millis(100));
                let mut names = adds.lock();
                if !names.contains(&"new".to_owned()) {
                    names.push("new".to_owned());
                }
            }
            _ => {}
        }
        if name == "next" {
            Say::Pick("old".to_owned())
        } else {
            Say::Yes(true)
        }
    });
    let fan = Tool::new("fan", "", |call| {
        let point = Signature::new("Point.").input("who").tool("next", "next?");
        // The first only answers: its one reading of the shelf is the slow one.
        let idle = Signature::new("Answer.").input("who").noul("idle", "idle?");
        let asks = [
            Ask::new(inputs! { "who" => "first" }).signature(idle),
            Ask::new(inputs! { "who" => "second" }).signature(point),
        ];
        Ok(call.batch(asks)?.len())
    });
    let shelf = Shelf {
        names: names.clone(),
        seen: seen.clone(),
    };
    let rdm = Rdm::new(Signature::new("Fan out.").input("who").noul("fan", "fan?"))
        .tool(fan)
        .source(shelf)
        .max_iters(1)
        .model(model);
    rdm.call(inputs! { "who" => "root" }).unwrap();
    assert_eq!(names.get(), ["old", "new"]);
    // The tool the second pointed at reads the run's map well after both readings were put on it.
    assert_eq!(seen.get(), [["fan", "old", "new"]]);
}
