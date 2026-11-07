//! A map: an output typed as a tool points at one of the module's tools, turn after turn.

mod common;

use common::{Meeting, client, offered, scripted, shown};
use indexmap::IndexMap;
use rdm::{Json, Rdm, Reason, Shared, Signature, Source, Tool, inputs};
use serde_json::json;

type Devices = Shared<IndexMap<String, String>>;

/// A house that pushes back: it will not lock an open door.
fn house() -> Devices {
    let devices = [("tv", "off"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")];
    Shared::new(
        devices
            .iter()
            .map(|(device, state)| (device.to_string(), state.to_string()))
            .collect(),
    )
}

fn set(house: &Devices, device: &str, state: &str) -> String {
    let mut devices = house.lock();
    if device == "lock" && state == "locked" && devices["door"] == "open" {
        return "lock: refused, the door is open".to_owned();
    }
    devices.insert(device.to_owned(), state.to_owned());
    format!("{device}: now {state}")
}

fn operate() -> Signature {
    Signature::new("Operate a smart home, one step at a time.")
        .input("request")
        .input_with("devices", "dict", "Every device and its state now.")
        .input_described("observation", "What the last step did, if there was one.")
        .tool("next", "What should happen next for `inputs.request`?")
}

/// Every step the house can take, each a tool with a name and a description.
fn lines(house: &Devices) -> Vec<Tool> {
    let steps = [
        ("tv on", "tv", "on"),
        ("lights off", "lights", "off"),
        ("close door", "door", "closed"),
        ("lock door", "lock", "locked"),
    ];
    let step = |(name, device, state): (&'static str, &'static str, &'static str)| {
        let house = house.clone();
        Tool::new(name, format!("Set the {device} to {state}."), move |_| {
            Ok(set(&house, device, state))
        })
    };
    steps.into_iter().map(step).collect()
}

fn pointed(result: &rdm::Prediction) -> Vec<String> {
    result.pointed("next")
}

#[test]
fn an_output_typed_as_a_tool_points_at_a_tool_turn_after_turn_until_finish() {
    let house = house();
    let model = scripted(["tv on", "lights off", "finish"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&house)).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Movie time.", "devices" => &house }).unwrap();
    // Each request offers the module's tools as options, each with its description, and `finish`.
    let requests = model.requests();
    assert_eq!(
        offered(&requests[0], "next"),
        ["tv on", "lights off", "close door", "lock door", "finish"]
    );
    assert_eq!(
        requests[0].questions["next"].criteria.as_ref().unwrap()["tv on"],
        "Set the tv to on."
    );
    // The tool pointed at runs, and the same question is asked again about the house as it is now.
    assert_eq!(
        shown(&model)
            .iter()
            .map(|inputs| inputs["devices"]["tv"].clone())
            .collect::<Vec<_>>(),
        ["off", "on", "on"]
    );
    assert_eq!(
        shown(&model).iter().map(|inputs| inputs["observation"].clone()).collect::<Vec<_>>(),
        ["", "tv: now on", "lights: now off"]
    );
    assert_eq!((house.get()["tv"].as_str(), house.get()["lights"].as_str()), ("on", "off"));
    // What was done, in order, and every turn at the same level.
    assert_eq!(
        result.observations["next"],
        json!([["tv on", "tv: now on"], ["lights off", "lights: now off"], ["finish", null]])
    );
    assert_eq!(result.steps("next")[1], ("lights off".to_owned(), json!("lights: now off")));
    let turns: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| {
            (
                entry.depth,
                entry.iteration,
                entry.parent,
                entry.tool_name.as_deref(),
                entry.value("next"),
            )
        })
        .collect();
    assert_eq!(
        turns,
        [
            (0, 0, None, None, json!("tv on")),
            (0, 1, Some(0), Some("tv on"), json!("lights off")),
            (0, 2, Some(1), Some("lights off"), json!("finish"))
        ]
    );
    // Each turn's entry holds what that turn's tool returned, and the built-in finish records nothing.
    assert_eq!(json!(result.trajectory[0].observations), json!({"next": "tv: now on"}));
    assert_eq!(json!(result.trajectory[2].observations), json!({}));
    // The trajectory holds what each request showed, though the tools changed the same house since.
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| entry.inputs["devices"]["tv"].clone())
            .collect::<Vec<_>>(),
        ["off", "on", "on"]
    );
    // The call's own answer is its first point.
    assert_eq!((result.chosen("next"), result.termination_reason), (Some("tv on"), Reason::Final));
}

#[test]
fn the_observation_shows_what_was_refused_so_the_next_point_can_put_it_right() {
    let house = house();
    let model = scripted(["lock door", "close door", "lock door", "finish"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&house)).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Lock up.", "devices" => &house }).unwrap();
    assert_eq!(shown(&model)[1]["observation"], "lock: refused, the door is open");
    assert_eq!(
        json!(house.get()),
        json!({"tv": "off", "lights": "on", "door": "closed", "lock": "locked"})
    );
    assert_eq!(pointed(&result), ["lock door", "close door", "lock door", "finish"]);
}

/// Something that offers tools of its own, which change while a run goes on.
struct Shelf {
    names: Shared<Vec<String>>,
}

impl Source for Shelf {
    fn offers(&self) -> Vec<Tool> {
        let offer = |name: &String| {
            let (names, first) = (self.names.clone(), name == "first");
            Tool::new(name.clone(), format!("The {name}."), move |_| {
                if first {
                    names.set(vec!["second".to_owned()]);
                    return Ok("put up the second");
                }
                Ok("did second")
            })
        };
        self.names.lock().iter().map(offer).collect()
    }
}

#[test]
fn something_that_offers_tools_is_given_whole_and_its_tools_are_on_the_map_as_they_are_then() {
    let names = Shared::new(vec!["first".to_owned()]);
    let model = scripted(["first", "second", "finish"]);
    let rdm = Rdm::new(operate())
        .tools_from(lines(&house()))
        .source(Shelf { names: names.clone() })
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => "Go on.", "devices" => json!({}) }).unwrap();
    // Each request's map holds what the shelf offers then: a name it puts up is offered from the next
    // request, and one it takes down is gone. Its tools are not copied onto the module's own map.
    let beside: Vec<Vec<String>> = model
        .requests()
        .iter()
        .map(|request| offered(request, "next")[4..].to_vec())
        .collect();
    assert_eq!(beside, [["first", "finish"], ["second", "finish"], ["second", "finish"]]);
    assert!(!rdm.tools().contains("first") && rdm.tools().len() == 4);
    assert_eq!(
        result.observations["next"],
        json!([["first", "put up the second"], ["second", "did second"], ["finish", null]])
    );
    // A name it offers that the map already has is a mistake of the program.
    names.set(vec!["tv on".to_owned()]);
    let error = rdm.call(inputs! { "request" => "Go on.", "devices" => json!({}) }).unwrap_err();
    assert!(error.is_usage());
    assert_eq!(
        error.to_string(),
        "['tv on'] are offered by `Shelf` and are already names on the map."
    );
}

#[test]
fn a_tool_added_while_the_module_runs_is_offered_from_the_next_request() {
    let house = house();
    // Do again what was done, as one tool.
    let movie = Tool::new("movie time", "Do again what was done for 'Movie time.'.", |call| {
        call.run("tv on")?;
        call.run("lights off")?;
        Ok("tv: now on, lights: now off")
    });
    // Keep what was just done as one tool: for the rest of this run, and for the runs after it.
    let keep = Tool::new("keep", "", move |call| {
        call.tools().insert(movie.clone());
        call.rdm().tools().insert(movie.clone());
        Ok("kept as movie time")
    });
    let model = scripted(["tv on", "lights off", "keep", "finish", "movie time", "finish"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&house)).tool(keep).model(model.clone());
    rdm.call(inputs! { "request" => "Movie time.", "devices" => &house }).unwrap();
    let requests = model.requests();
    assert!(!offered(&requests[2], "next").contains(&"movie time".to_owned()));
    assert!(offered(&requests[3], "next").contains(&"movie time".to_owned()));
    // The next run is offered it, and one point does what two did.
    house.update(|devices| {
        devices.insert("tv".into(), "off".into());
        devices.insert("lights".into(), "on".into());
    });
    let result = rdm
        .call(inputs! { "request" => "Let's watch a film.", "devices" => &house })
        .unwrap();
    assert_eq!(
        result.observations["next"],
        json!([["movie time", "tv: now on, lights: now off"], ["finish", null]])
    );
    assert_eq!((house.get()["tv"].as_str(), house.get()["lights"].as_str()), ("on", "off"));
    // A name taken off the module's map is not offered to the run after.
    rdm.tools().remove("movie time");
    assert!(rdm.tools().get("movie time").is_none() && rdm.tools().names().last().unwrap() == "keep");
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn each_run_points_at_its_own_copy_of_the_tools() {
    // A tool put on the run's map is this run's: another run, here at the same time on another
    // thread, is not offered it, and the module's own tools are as they were.
    let remember = Tool::new("remember", "", |call| {
        let request: String = call.arg("request")?;
        call.tools()
            .insert(Tool::new(format!("redo {request}"), format!("Do '{request}' again."), |_| {
                Ok("redone")
            }));
        Ok("remembered")
    });
    let signature = Signature::new("Remember requests.")
        .input("request")
        .input("observation")
        .tool("next", "What next?");
    let meeting = Meeting::of(2);
    let model = client(move |inputs, _, _| {
        if inputs["observation"] == "" {
            return "remember";
        }
        // Both runs have remembered before either is asked again.
        assert!(meeting.meet(), "the two runs were not running at once");
        "finish"
    });
    let rdm = Rdm::new(signature).tool(remember).model(model.clone());
    let results = rdm.batch([inputs! { "request" => "a" }, inputs! { "request" => "b" }]);
    assert!(
        results
            .iter()
            .all(|result| result.as_ref().is_ok_and(|result| result.trajectory.len() == 2))
    );
    assert_eq!(results[0].as_ref().unwrap().trajectory[0].inputs["request"], "a");
    let mut later: Vec<Vec<String>> = model
        .requests()
        .iter()
        .filter(|request| request.state["inputs"]["observation"] != "")
        .map(|request| offered(request, "next"))
        .collect();
    later.sort();
    assert_eq!(later, [["remember", "redo a", "finish"], ["remember", "redo b", "finish"]]);
    assert_eq!(rdm.tools().names(), ["remember"]);
}

fn count_up(count: &Shared<Json>) -> Tool {
    let count = count.clone();
    Tool::new("step", "", move |_| {
        Ok(count.update(|count| {
            count["n"] = json!(count["n"].as_i64().unwrap_or(0) + 1);
            count["n"].clone()
        }))
    })
}

#[test]
fn a_long_loop_is_a_loop_and_its_end_is_recorded() {
    let looped = || Signature::new("Count.").input_typed("count", "dict").tool("next", "What next?");
    // Five hundred turns at one level: no deeper than one.
    let count = Shared::new(json!({"n": 0}));
    let rdm = Rdm::new(looped())
        .tool(count_up(&count))
        .max_iters(500)
        .max_predictor_calls(1000)
        .model(client(|_, _, _| "step"));
    let result = rdm.call(inputs! { "count" => &count }).unwrap();
    assert_eq!((result.steps("next").len(), result.termination_reason), (500, Reason::MaxIters));
    assert!(result.trajectory.iter().all(|entry| entry.depth == 0) && result.trajectory.len() == 501);
    // Pointing at `finish` ends a loop, and says so where the tools are.
    let count = Shared::new(json!({"n": 0}));
    rdm.tools().insert(count_up(&count));
    rdm.set_model(client(|inputs, _, _| {
        if inputs["count"]["n"].as_i64() >= Some(2) {
            "finish"
        } else {
            "step"
        }
    }));
    let result = rdm.call(inputs! { "count" => &count }).unwrap();
    assert_eq!(result.observations["next"], json!([["step", 1], ["step", 2], ["finish", null]]));
    assert_eq!(result.termination_reason, "final");
    // Two outputs that point are refused: each loop would put the other's question again.
    let two = Signature::new("Count.").input_typed("count", "dict").tool("a", "?").tool("b", "?");
    let error = Rdm::new(two)
        .tool(count_up(&count))
        .model(client(|_, _, _| "step"))
        .call(inputs! { "count" => &count })
        .unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("one output that points at a tool"));
}

#[test]
fn a_loop_of_thousands_of_turns_needs_no_more_stack_than_one() {
    // The turns are a loop and the way here is taken apart one request at a time: a thread with
    // a small stack runs five thousand turns, and drops what it made.
    let run = || {
        let count = Shared::new(json!({"n": 0}));
        let signature = Signature::new("Count.").input_typed("count", "dict").tool("next", "What next?");
        let rdm = Rdm::new(signature)
            .tool(count_up(&count))
            .max_iters(5000)
            .max_predictor_calls(10_000)
            .model(client(|_, _, _| "step"));
        let result = rdm.call(inputs! { "count" => &count }).unwrap();
        (result.steps("next").len(), result.termination_reason)
    };
    assert_eq!(common::on_a_stack(192 * 1024, run), (5000, Reason::MaxIters));
}

#[test]
fn a_tool_observed_is_shown_as_json_and_a_tool_put_on_the_map_later_is_a_tool() {
    let look = Signature::new("Look.").input("observation").tool("next", "What next?");
    let model = client(|inputs, _, _| if inputs["observation"] == "" { "rooms" } else { "finish" });
    let rdm = Rdm::new(look).model(model.clone());
    // Put on the map after the module was built: offered and run under its name.
    rdm.tools()
        .insert(Tool::new("rooms", "", |_| Ok(json!({"ok": true, "rooms": ["hall", null]}))));
    rdm.call(inputs! {}).unwrap();
    assert_eq!(shown(&model)[1]["observation"], r#"{"ok": true, "rooms": ["hall", null]}"#);
    // A text is shown as it is, nothing as an empty text, and anything else as JSON, as inputs are shown.
    assert_eq!(rdm::as_observation(&json!("as it is")), "as it is");
    assert_eq!(rdm::as_observation(&json!(null)), "");
    assert_eq!(
        (
            rdm::as_observation(&json!(3)),
            rdm::as_observation(&json!(true)),
            rdm::as_observation(&json!([1, null, "é"]))
        ),
        ("3".into(), "true".into(), "[1, null, \"é\"]".into())
    );
}

#[test]
fn a_turn_that_changes_nothing_ends_the_loop_as_a_repeat() {
    let devices = house();
    let model = scripted(["tv on", "tv on", "tv on"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&devices)).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Put the TV on.", "devices" => &devices }).unwrap();
    // The second turn shows the same house and the same observation as the third would: it is not asked again.
    assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::Repeat));
    assert_eq!(pointed(&result), ["tv on", "tv on"]);
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.repeat).collect::<Vec<_>>(),
        [false, false, true]
    );
    // A plain value is a copy: the tool's change is not shown, and the loop comes round at once.
    let apart = house();
    let model = scripted(["tv on", "tv on"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&apart)).model(model.clone());
    let result = rdm
        .call(inputs! { "request" => "Put the TV on.", "devices" => apart.get() })
        .unwrap();
    assert_eq!(
        (model.requests().len(), result.termination_reason, pointed(&result).len()),
        (2, Reason::Repeat, 2)
    );
}

#[test]
fn a_tool_named_finish_takes_the_place_of_the_built_in_one() {
    let house = house();
    let said = Shared::new(Vec::new());
    let says = said.clone();
    let finish = Tool::new("finish", "Stop: everything the request asks for is so.", move |_| {
        says.lock().push("done");
        Ok("done")
    });
    let model = scripted(["tv on", "finish"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&house)).tool(finish).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Put the TV on.", "devices" => &house }).unwrap();
    let first = &model.requests()[0];
    assert_eq!(
        first.questions["next"].criteria.as_ref().unwrap()["finish"],
        "Stop: everything the request asks for is so."
    );
    assert_eq!(offered(first, "next").iter().filter(|name| *name == "finish").count(), 1);
    assert_eq!(said.get(), ["done"]);
    assert_eq!(result.observations["next"], json!([["tv on", "tv: now on"], ["finish", "done"]]));
    assert_eq!(model.requests().len(), 2);
}

#[test]
fn a_tool_that_fails_is_shown_as_the_observation_and_the_loop_goes_on() {
    let house = house();
    let jam = Tool::new("jam", "Jam the lock.", |_| Err::<(), _>("the lock is jammed".into()));
    let model = scripted(["jam", "finish"]);
    let rdm = Rdm::new(operate()).tools_from(lines(&house)).tool(jam).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Jam it.", "devices" => &house }).unwrap();
    assert_eq!(shown(&model)[1]["observation"], "Execution error in jam: the lock is jammed");
    assert_eq!((pointed(&result)[0].as_str(), result.termination_reason), ("jam", Reason::Final));
}

#[test]
fn a_pointer_can_be_given_no_finish() {
    let play = || Signature::new("Play.").input_typed("board", "dict").tool("next", "Which move?");
    let moved = |name: &'static str, board: &Shared<Json>, by: i64| {
        let board = board.clone();
        Tool::new(name, "", move |_| {
            board.update(|board| board["x"] = json!(board["x"].as_i64().unwrap_or(0) + by));
            Ok(())
        })
    };
    // Without `finish` there is nothing to stop at but a limit or a repeat: a game is played to its end.
    let board = Shared::new(json!({"x": 0}));
    let model = client(|_, _, _| "right");
    let rdm = Rdm::new(play())
        .tool(moved("left", &board, -1))
        .tool(moved("right", &board, 1))
        .finish(false)
        .max_iters(5)
        .model(model.clone());
    let result = rdm.call(inputs! { "board" => &board }).unwrap();
    assert!(model.requests().iter().all(|request| offered(request, "next") == ["left", "right"]));
    assert_eq!((board.get()["x"].clone(), result.termination_reason), (json!(5), Reason::MaxIters));
    // A tool named `finish` still ends the loop: it is a tool the caller chose to offer.
    let board = Shared::new(json!({"x": 0}));
    let over = Tool::new("finish", "The game is over.", |_| Ok("over"));
    let model = client(|inputs, _, _| {
        if inputs["board"]["x"].as_i64() >= Some(2) {
            "finish"
        } else {
            "right"
        }
    });
    let rdm = Rdm::new(play())
        .tool(moved("right", &board, 1))
        .tool(over)
        .finish(false)
        .model(model);
    assert_eq!(
        rdm.call(inputs! { "board" => &board }).unwrap().steps("next").last().unwrap(),
        &("finish".to_owned(), json!("over"))
    );
    // With one tool and no `finish` there is nothing to decide: it runs without a request, turn after turn.
    let board = Shared::new(json!({"x": 0}));
    let model = client(|_, _, _| "right");
    let rdm = Rdm::new(play())
        .tool(moved("right", &board, 1))
        .finish(false)
        .max_iters(3)
        .model(model.clone());
    let result = rdm.call(inputs! { "board" => &board }).unwrap();
    assert_eq!(
        (result.termination_reason, board.get()["x"].clone(), model.requests().len()),
        (Reason::MaxIters, json!(3), 0)
    );
    assert_eq!(json!(result.trajectory[0])["unasked"], json!(["next"]));
    // With no tool at all and no `finish`, there is nothing to point at.
    let error = Rdm::new(play())
        .finish(false)
        .model(client(|_, _, _| "right"))
        .call(inputs! { "board" => &board })
        .unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("none to point at"));
    // With no tool and `finish`, the one option is `finish`: nothing is asked, and the loop ends there.
    let model = client(|_, _, _| "finish");
    let result = Rdm::new(play()).model(model.clone()).call(inputs! { "board" => &board }).unwrap();
    assert_eq!(
        (
            result.observations["next"].clone(),
            model.requests().len(),
            result.termination_reason
        ),
        (json!([["finish", null]]), 0, Reason::Final)
    );
}

#[test]
fn the_names_a_module_always_offers_are_marked_on_the_pointer() {
    // What a fit reads: which output points, and which options it may not cut.
    let seen = Shared::new(Vec::new());
    let noted = seen.clone();
    let fit = move |signature: Signature, inputs: rdm::Inputs, _: &rdm::Call| {
        let output = &signature.outputs()["next"];
        let stays: Vec<String> = output
            .options()
            .unwrap()
            .iter()
            .filter(|option| output.stays(option))
            .map(|option| option.label.clone())
            .collect();
        noted
            .lock()
            .push((signature.pointer().map(str::to_owned), output.points, output.always.clone(), stays));
        // A fit may cut the options, and the answer is decided among those that are left.
        Ok((
            signature.with_options("next", [("lights off", "Set the lights to off."), ("finish", "Stop.")]),
            inputs,
        ))
    };
    let house = house();
    let model = client(|_, _, question| question.labels()[0].clone());
    let rdm = Rdm::new(operate())
        .tools_from(lines(&house))
        .always(["lock door"])
        .fit_fn(fit)
        .max_iters(1)
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => "Night.", "devices" => &house }).unwrap();
    let names = |names: &[&str]| names.iter().map(|name| name.to_string()).collect::<Vec<_>>();
    assert_eq!(
        seen.get(),
        [(
            Some("next".to_owned()),
            true,
            names(&["finish", "lock door"]),
            names(&["lock door", "finish"])
        )]
    );
    assert_eq!(offered(&model.requests()[0], "next"), ["lights off", "finish"]);
    assert_eq!(pointed(&result), ["lights off"]);
    // The signature as the module was given it is untouched: the pointer is filled in for each request.
    assert!(rdm.signature().output("next").unwrap().options().is_none() && rdm.signature().output("next").unwrap().is_pointer());
}
