//! The leaf, the limits, and modules inside modules.

mod common;

use common::{Say, asked, at_depth, client, client_using, skipped};
use rdm::{Ask, Json, Prediction, Rdm, Reason, Shared, Signature, Tool, inputs};
use serde_json::json;

fn count() -> Signature {
    Signature::new("Count upwards.")
        .input_typed("n", "int")
        .noul("again", "Should the counting go on?")
}

fn triage() -> Signature {
    Signature::new("Triage a support ticket.")
        .input("ticket")
        .noul("refund", "Is the customer asking for a refund?")
        .noul("escalate", "Does the ticket need an engineer?")
        .score("frustration", "How frustrated is the customer?", ["calm", "frustrated", "angry"])
}

fn ns(entries: &[&rdm::Entry]) -> Vec<i64> {
    let mut ns: Vec<i64> = entries.iter().map(|entry| entry.inputs["n"].as_i64().unwrap()).collect();
    ns.sort();
    ns
}

#[test]
fn the_module_calls_itself_until_max_depth_where_a_call_is_a_plain_predict() {
    let ran = Shared::new(Vec::new());
    let tree = |max_depth: usize| {
        ran.set(Vec::new());
        let noted = ran.clone();
        // Put the same question about the next two numbers, each in a sub-call.
        let again = Tool::new("again", "", move |call| {
            let n: i64 = call.arg("n")?;
            noted.lock().push(n);
            Ok([
                call.sub(inputs! { "n" => 2 * n })?.termination_reason,
                call.sub(inputs! { "n" => 2 * n + 1 })?.termination_reason,
            ])
        });
        let model = client(|_, _, _| true);
        let result = Rdm::new(count())
            .tool(again)
            .max_depth(max_depth)
            .model(model.clone())
            .call(inputs! { "n" => 1 })
            .unwrap();
        (result, model)
    };
    let sorted = |mut ran: Vec<i64>| {
        ran.sort();
        ran
    };

    // rdm > rdm > predict. The same module and the same question at every level; at depth 2 the
    // question is still answered, as by a plain predict, and the tool it selected is not run.
    let (result, model) = tree(2);
    let mut depths: Vec<usize> = result.trajectory.iter().map(|entry| entry.depth).collect();
    depths.sort();
    assert_eq!((depths, model.requests().len()), (vec![0, 1, 1, 2, 2, 2, 2], 7));
    assert_eq!(ns(&at_depth(&result, 2)), [4, 5, 6, 7]);
    assert!(
        at_depth(&result, 2)
            .iter()
            .all(|entry| entry.value("again") == true && entry.observations.as_ref().is_some_and(|seen| seen.is_empty()))
    );
    assert_eq!(sorted(ran.get()), [1, 2, 3]);
    assert_eq!(json!(result.observations), json!({"again": ["max_depth", "max_depth"]}));
    // What the depth limit stopped two levels down is reported at every level above it.
    assert_eq!(result.termination_reason, "max_depth");
    assert_eq!(
        json!(result.trajectory[1].observations),
        json!({"again": ["max_depth", "max_depth"]})
    );
    // With a depth of 1 the root uses its tools and every sub-call is a leaf.
    let (result, model) = tree(1);
    assert_eq!((model.requests().len(), ran.get()), (3, vec![1]));
    assert_eq!(json!(result.observations), json!({"again": ["max_depth", "max_depth"]}));
    // With a depth of 0 the module itself is a plain predict.
    let (result, model) = tree(0);
    assert_eq!(
        (
            model.requests().len(),
            ran.get().len(),
            result.yes("again"),
            result.termination_reason
        ),
        (1, 0, true, Reason::MaxDepth)
    );
}

#[test]
fn a_choice_with_one_option_is_answered_without_being_asked() {
    fn ways(left: &[String], lit: bool) -> Signature {
        let ways = Signature::new("Leave the room.")
            .input_with("left", "list", "Doors not yet tried.")
            .choice("go", "Which door in `inputs.left` should be tried next?", left);
        if lit { ways.noul("lit", "Is the room lit?") } else { ways }
    }
    let doors = |doors: &[&str]| doors.iter().map(|door| door.to_string()).collect::<Vec<_>>();

    let tried = Shared::new(Vec::new());
    let noted = tried.clone();
    let go = Tool::new("go", "", move |call| {
        let (go, left): (String, Vec<String>) = (call.arg("go")?, call.arg("left")?);
        noted.lock().push(go.clone());
        let rest: Vec<String> = left.into_iter().filter(|other| *other != go).collect();
        if rest.is_empty() {
            return Ok(json!("out"));
        }
        let below = call.sub_with(Ask::new(inputs! { "left" => &rest }).signature(ways(&rest, false)))?;
        Ok(below.observations["go"].clone())
    });
    let model = client(|_, name, question| {
        if name == "lit" {
            Say::Yes(true)
        } else {
            Say::Pick(question.labels().last().unwrap().clone())
        }
    });
    let rdm = Rdm::new(ways(&doors(&["a", "b", "c"]), false))
        .tool(go)
        .max_predictor_calls(2)
        .model(model.clone());
    let result = rdm.call(inputs! { "left" => ["a", "b", "c"] }).unwrap();
    // Three choices, and two requests: with one door left there is nothing to decide. The answer is
    // that door, as certain, its tool runs like any other, and no limit is charged for it.
    assert_eq!(
        (tried.get(), model.requests().len(), result.termination_reason),
        (doors(&["c", "b", "a"]), 2, Reason::Final)
    );
    let last = result.trajectory.last().unwrap();
    assert_eq!(
        (last.unasked.clone(), last.tokens, last.model.clone()),
        (Some(doors(&["go"])), Some(0), None)
    );
    assert_eq!(
        json!(last.outputs.as_ref().unwrap()["go"]),
        json!({"confidence": 1.0, "value": "a", "probabilities": {"a": 1.0}})
    );
    assert_eq!(
        (result.observations["go"].clone(), result.trajectory[0].unasked.is_none()),
        (json!("out"), true)
    );
    // Beside other questions, only the others are asked.
    let model = client(|_, name, question| {
        if name == "lit" {
            Say::Yes(true)
        } else {
            Say::Pick(question.labels()[0].clone())
        }
    });
    let rdm = Rdm::new(ways(&doors(&["a"]), true))
        .tool(Tool::new("go", "", |_| Ok("out")))
        .model(model.clone());
    let result = rdm.call(inputs! { "left" => ["a"] }).unwrap();
    assert_eq!(asked(&model), [["lit"]]);
    assert_eq!((result.chosen("go"), result.yes("lit")), (Some("a"), true));
    assert_eq!(
        result.trajectory[0].outputs.as_ref().unwrap().keys().collect::<Vec<_>>(),
        ["go", "lit"]
    );
    assert_eq!(json!(result.observations), json!({"go": "out"}));
    let keys: Vec<String> = json!(result.trajectory[0]).as_object().unwrap().keys().cloned().collect();
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
            "unasked",
            "observations"
        ]
    );
}

#[test]
fn a_call_can_be_asked_for_its_answers_only() {
    let ran = Shared::new(Vec::new());
    let noted = ran.clone();
    let again = Tool::new("again", "", move |call| {
        let n: i64 = call.arg("n")?;
        noted.lock().push(n);
        // The same question about another number, for its answer only: it says yes, and that runs nothing.
        let leaf = call.sub_with(Ask::new(inputs! { "n" => n + 1 }).max_depth(0))?;
        // And about another, which may run its tool, but nothing below it may.
        let deeper = call.sub_with(Ask::new(inputs! { "n" => n + 10 }).max_depth(1))?;
        assert_eq!(
            (leaf.yes("again"), leaf.termination_reason, deeper.yes("again")),
            (true, Reason::Final, true)
        );
        call.sub_with(Ask::new(inputs! { "n" => n + 100 }).answers_only())
    });
    let rdm = Rdm::new(count()).tool(again).max_depth(5).model(client(|_, _, _| true));
    let result = rdm.call(inputs! { "n" => 1 }).unwrap();
    // Every request said yes. The tool ran only where the call was allowed to go further.
    assert_eq!(ran.get(), [1, 11]);
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| entry.inputs["n"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        [1, 2, 11, 12, 21, 111, 101]
    );
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>(),
        [0, 1, 1, 2, 2, 2, 1]
    );
    // Asking for answers only keeps nothing from being done, so it is not a limit that was reached.
    // The limit put on the deeper call did stop a tool, and that is reported all the way up.
    assert_eq!(
        (&result.observations["again"]["termination_reason"], result.termination_reason),
        (&json!("final"), Reason::MaxDepth)
    );
    // The call that starts a run can be asked the same way: it is then a predict.
    ran.set(Vec::new());
    let result = rdm.call_with(Ask::new(inputs! { "n" => 1 }).max_depth(0)).unwrap();
    assert_eq!(
        (
            ran.get().len(),
            result.trajectory.len(),
            result.yes("again"),
            result.termination_reason
        ),
        (0, 1, true, Reason::Final)
    );
}

fn tick() -> Signature {
    Signature::new("Count by pointing.")
        .input_typed("counter", "dict")
        .tool("next", "What next?")
}

fn begin() -> Signature {
    Signature::new("Decide whether to count.")
        .input_typed("counter", "dict")
        .noul("count", "Should the counting start?")
}

fn ticking(counter: &Shared<Json>) -> Tool {
    let counter = counter.clone();
    Tool::new("tick", "", move |_| {
        Ok(counter.update(|counter| {
            counter["n"] = json!(counter["n"].as_i64().unwrap_or(0) + 1);
            counter["n"].clone()
        }))
    })
}

fn points_and_says_yes() -> std::sync::Arc<rdm::Scripted> {
    client(|_, name, _| if name == "next" { Say::Pick("tick".into()) } else { Say::Yes(true) })
}

#[test]
fn another_rdm_called_from_a_tool_joins_the_run_and_keeps_to_its_own_limits_too() {
    let start = || {
        Signature::new("Decide whether to count.")
            .input_typed("n", "int")
            .noul("count", "Should the counting start?")
    };
    let ran = Shared::new(Vec::new());
    let run = |outer_depth: usize, inner_depth: usize| {
        ran.set(Vec::new());
        let noted = ran.clone();
        let again = Tool::new("again", "", move |call| {
            let n: i64 = call.arg("n")?;
            noted.lock().push(n);
            Ok(call.sub(inputs! { "n" => n + 1 })?.termination_reason)
        });
        let model = client(|_, _, _| true);
        let inner = Rdm::new(count()).tool(again).max_depth(inner_depth).model(model.clone());
        let outer = Rdm::new(start())
            .tool(Tool::module("count", "", &inner))
            .max_depth(outer_depth)
            .model(model);
        outer.call(inputs! { "n" => 1 }).unwrap()
    };
    // The other module is a sub-call of the run it was called in: one trajectory.
    let result = run(9, 2);
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| (entry.depth, entry.inputs["n"].as_i64().unwrap()))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 1), (2, 2), (3, 3)]
    );
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.tool_name.as_deref()).collect::<Vec<_>>(),
        [None, Some("count"), Some("again"), Some("again")]
    );
    // Its own max_depth counts from where it was called: its tool ran at its first two levels and not below.
    assert_eq!((ran.get(), result.termination_reason), (vec![1, 2], Reason::MaxDepth));
    // It goes as deep as it would alone, whatever the depth of the module that called it: a module
    // behaves the same wherever it is used. What bounds them both is the run's requests and tokens.
    let result = run(1, 2);
    assert_eq!(
        (ran.get(), result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>()),
        (vec![1, 2], vec![0, 1, 2, 3])
    );

    // The turns its loop may take are its own, too.
    let counter = Shared::new(json!({"n": 0}));
    let inner = Rdm::new(tick()).tool(ticking(&counter)).max_iters(2).model(points_and_says_yes());
    let outer = Rdm::new(begin())
        .tool(Tool::module("count", "", &inner))
        .max_iters(20)
        .model(points_and_says_yes());
    let result = outer.call(inputs! { "counter" => &counter }).unwrap();
    let turns: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.iteration, entry.skipped))
        .collect();
    assert_eq!(turns, [(0, 0, None), (1, 0, None), (1, 1, None), (1, 2, Some(Reason::MaxIters))]);
    // What the module came to is the observation of the tool it was: its prediction, as plain data.
    assert_eq!(
        result.observations["count"]["observations"]["next"],
        json!([["tick", 1], ["tick", 2]])
    );
    assert_eq!(result.observations["count"]["termination_reason"], "max_iters");
}

#[test]
fn a_module_called_inside_a_tool_keeps_its_own_limits_and_the_runs() {
    let run = |inner_calls: usize, outer_calls: usize, inner_tokens: Option<u64>, outer_tokens: Option<u64>, tokens: u64| {
        let counter = Shared::new(json!({"n": 0}));
        let model = client_using(
            tokens,
            |_, name, _| if name == "next" { Say::Pick("tick".into()) } else { Say::Yes(true) },
        );
        let inner = Rdm::new(tick())
            .tool(ticking(&counter))
            .max_predictor_calls(inner_calls)
            .max_tokens(inner_tokens)
            .model(model.clone());
        let outer = Rdm::new(begin())
            .tool(Tool::module("count", "", &inner))
            .max_predictor_calls(outer_calls)
            .max_tokens(outer_tokens)
            .model(model);
        let result = outer.call(inputs! { "counter" => &counter }).unwrap();
        (result, counter.get()["n"].as_i64().unwrap())
    };
    // Alone, the inner module stops at its three requests. Called inside a tool it joins the run,
    // and its own limit still holds for what it does there.
    let (result, ticks) = run(3, 100, None, None, 0);
    assert_eq!((ticks, result.termination_reason), (3, Reason::MaxPredictorCalls));
    let stopped: Vec<Option<Reason>> = result.trajectory.iter().map(|entry| entry.skipped).collect();
    assert_eq!(stopped, [None, None, None, None, Some(Reason::MaxPredictorCalls)]);
    // The run's limit holds for the inner module too: here it is the smaller.
    let (result, ticks) = run(100, 4, None, None, 0);
    assert_eq!((ticks, result.termination_reason), (3, Reason::MaxPredictorCalls));
    // The same for tokens: each module's limit is on what is used under it.
    let (result, ticks) = run(100, 100, Some(250), None, 100);
    assert_eq!((ticks, result.termination_reason), (3, Reason::MaxTokens));
    let (result, ticks) = run(100, 100, None, Some(250), 100);
    assert_eq!((ticks, result.termination_reason), (2, Reason::MaxTokens));
}

#[test]
fn requests_turns_and_tokens_are_bounded_and_what_was_cut_is_recorded() {
    let again = || {
        Tool::new("again", "", |call| {
            let n: i64 = call.arg("n")?;
            Ok([call.sub(inputs! { "n" => 2 * n })?, call.sub(inputs! { "n" => 2 * n + 1 })?])
        })
    };
    // The number of requests answered is bounded across the whole run. A call that was not
    // answered comes back to the tool with no answers and the limit that stopped it.
    let model = client(|_, _, _| true);
    let result = Rdm::new(count())
        .tool(again())
        .max_depth(10)
        .max_predictor_calls(5)
        .model(model.clone())
        .call(inputs! { "n" => 1 })
        .unwrap();
    let cut = skipped(&result, "max_predictor_calls");
    assert_eq!(model.requests().len(), 5);
    assert!(!cut.is_empty() && cut.iter().all(|entry| entry.outputs.is_none() && !entry.asked()));
    let below = result
        .trajectory
        .iter()
        .filter_map(|entry| entry.observations.as_ref()?.get("again")?.as_array())
        .flatten();
    let refused: Vec<&Json> = below.filter(|sub| sub["again"].is_null()).collect();
    assert!(
        !refused.is_empty()
            && refused
                .iter()
                .all(|sub| sub["termination_reason"] == "max_predictor_calls" && sub.as_object().unwrap().contains_key("again"))
    );
    assert_eq!(result.termination_reason, "max_predictor_calls");

    // So are the tokens the requests use, as the model reports them.
    let model = client_using(100, |_, _, _| true);
    let result = Rdm::new(count())
        .tool(again())
        .max_depth(10)
        .max_tokens(350)
        .model(model.clone())
        .call(inputs! { "n" => 1 })
        .unwrap();
    assert_eq!(
        (
            model.requests().len(),
            result.trajectory.iter().filter_map(|entry| entry.tokens).sum::<u64>()
        ),
        (4, 400)
    );
    assert!(!skipped(&result, "max_tokens").is_empty());
    assert_eq!(result.termination_reason, Reason::MaxTokens);

    // Turns are the times a pointer's question is put in a row.
    let steps = Signature::new("Take steps.")
        .input_typed("n", "int")
        .input("observation")
        .tool("next", "What next?");
    let taken = Shared::new(0);
    let step = Tool::new("step", "", move |_| Ok(taken.update(|taken| std::mem::replace(taken, *taken + 1))));
    let model = client(|_, _, _| "step");
    let result = Rdm::new(steps)
        .tool(step)
        .max_iters(3)
        .model(model.clone())
        .call(inputs! { "n" => 0 })
        .unwrap();
    let turns: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.iteration, entry.skipped))
        .collect();
    assert_eq!(turns, [(0, 0, None), (0, 1, None), (0, 2, None), (0, 3, Some(Reason::MaxIters))]);
    assert_eq!((model.requests().len(), result.termination_reason), (3, Reason::MaxIters));
    // The turn a limit stopped is recorded with what it would have shown, and the tool that led to it.
    let stopped = json!(result.trajectory[3]);
    assert_eq!(
        stopped,
        json!({"depth": 0, "iteration": 3, "parent": 2, "tool_name": "step", "inputs": {"n": 0, "observation": "2"}, "skipped": "max_iters"})
    );
}

#[test]
fn the_reason_is_what_bounds_the_most() {
    assert_eq!(Reason::of([Reason::Repeat, Reason::MaxDepth, Reason::Final]), Reason::MaxDepth);
    assert_eq!(Reason::of([Reason::MaxDepth, Reason::MaxIters]), Reason::MaxIters);
    assert_eq!(Reason::of([Reason::MaxIters, Reason::MaxPredictorCalls]), Reason::MaxPredictorCalls);
    assert_eq!(
        Reason::of([Reason::MaxPredictorCalls, Reason::MaxTokens, Reason::Repeat]),
        Reason::MaxTokens
    );
    assert_eq!(Reason::of([]), Reason::Final);
    let names: Vec<String> = [
        Reason::MaxTokens,
        Reason::MaxPredictorCalls,
        Reason::MaxIters,
        Reason::MaxDepth,
        Reason::Repeat,
        Reason::Final,
    ]
    .iter()
    .map(|reason| reason.to_string())
    .collect();
    assert_eq!(
        json!(names),
        json!(["max_tokens", "max_predictor_calls", "max_iters", "max_depth", "repeat", "final"])
    );
    assert_eq!(json!(Reason::MaxPredictorCalls), json!("max_predictor_calls"));
}

#[test]
fn a_tool_that_fails_is_an_observation_and_not_the_end_of_the_run() {
    let refund = Tool::new("refund", "", |_| Err::<(), _>(rdm::Error::tool("the ledger did not answer")));
    let escalate = Tool::new("escalate", "", |_| Ok("paged"));
    // Any error that is not the module's own is the tool's: one from a file, say, boxed or not.
    let boxed = Tool::new("frustration", "", |_| {
        let failed: Box<dyn std::error::Error + Send + Sync> = Box::new(std::io::Error::other("no such ledger"));
        Err::<(), _>(failed.into())
    });
    let model = client(|_, _, question| {
        if question.kind == rdm::systemone::Kind::Noul {
            Say::Yes(true)
        } else {
            Say::Level(0)
        }
    });
    let result = Rdm::new(triage())
        .tool(refund)
        .tool(escalate)
        .tool(boxed)
        .model(model)
        .call(inputs! { "ticket" => "t" })
        .unwrap();
    assert_eq!(
        result.observations["refund"],
        "Execution error in refund: the ledger did not answer"
    );
    assert_eq!(
        (result.observations["escalate"].clone(), result.observations.len()),
        (json!("paged"), 2)
    );
    // A tool that failed did not stop anything: the reason is still that the answers were followed.
    assert_eq!(result.termination_reason, Reason::Final);
    // One of the module's own errors that was boxed on the way is still the module's.
    let usage: Box<dyn std::error::Error + Send + Sync> = Box::new(rdm::Error::usage("wrong"));
    assert!(rdm::Error::from(usage).is_usage());
}

#[test]
fn a_tool_can_be_another_rdm() {
    let route = Signature::new("Route a request to a smart home assistant.")
        .input("request")
        .noul("chat", "Is the request a question to answer in conversation?")
        .noul("devices", "Is the request a command to a device?");
    let device = Signature::new("Carry out a command to a device.")
        .input("request")
        .noul("lock", "Should the door be locked?");
    let locked = Shared::new(Vec::new());
    let noted = locked.clone();
    let lock = Tool::new("lock", "", move |call| {
        noted.lock().push(call.arg::<String>("request")?);
        Ok(())
    });
    let child = Rdm::new(device).tool(lock).model(client(|_, _, _| true));
    let chat = Tool::new("chat", "", |_| Ok(json!({"reply": "It is 21 degrees."})));
    let rdm = Rdm::new(route).tool(chat).tool(Tool::module("devices", "", &child));
    // A tool that writes returns what it wrote.
    rdm.set_model(client(|_, name, _| name == "chat"));
    assert_eq!(
        rdm.call(inputs! { "request" => "How warm is it?" }).unwrap().observations["chat"]["reply"],
        "It is 21 degrees."
    );
    // Another module called from a tool is a sub-call like any other: it joins the run, one level deeper.
    rdm.set_model(client(|_, name, _| name == "devices"));
    let result = rdm.call(inputs! { "request" => "Lock up." }).unwrap();
    assert_eq!(locked.get(), ["Lock up."]);
    assert_eq!(result.observations["devices"]["lock"]["value"], true);
    let outputs: Vec<(usize, Vec<&String>)> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.outputs.as_ref().unwrap().keys().collect()))
        .collect();
    assert_eq!(
        outputs.iter().map(|(depth, names)| (*depth, names.len())).collect::<Vec<_>>(),
        [(0, 2), (1, 1)]
    );
    assert_eq!(outputs[1].1, ["lock"]);
    // Setting the module's model leaves the other module's alone.
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.model.as_deref()).collect::<Vec<_>>(),
        [Some("jev-test"), Some("jev-test")]
    );
    // A module whose inputs the request cannot give is a mistake of the program.
    let other = Rdm::new(Signature::new("Other.").input("door").noul("lock", "Lock?")).model(client(|_, _, _| true));
    let misfit = rdm.copy();
    misfit.tools().insert(Tool::module("devices", "", &other));
    let error = misfit.call(inputs! { "request" => "Lock up." }).unwrap_err();
    assert!(
        error.is_usage()
            && error
                .to_string()
                .starts_with("Tool 'devices' takes ['door'], which name no answer or input of the request.")
    );
    // The copy has its own map: the module it was copied from still has the tool it had.
    assert!(rdm.call(inputs! { "request" => "Lock up." }).is_ok());
}

#[test]
fn a_module_is_a_handle_and_its_model_can_be_set_later() {
    let rdm = Rdm::new(count());
    let same = rdm.clone();
    // A clone is the same module: what is set through one is set for the other.
    same.set_model(client(|_, _, _| true));
    assert!(rdm.decision_model().is_some_and(|model| model.name() == "jev-test"));
    assert!(rdm.call(inputs! { "n" => 1 }).unwrap().yes("again"));
    // Another model from now on, and each request says which one answered it.
    rdm.set_model(rdm::Scripted::new(|_| [("again", rdm::systemone::Answer::noul(0.2))]).named("other"));
    let result = same.call(inputs! { "n" => 1 }).unwrap();
    assert_eq!((result.yes("again"), result.trajectory[0].model.as_deref()), (false, Some("other")));
    // A clone moved into a tool is the module itself, so calling it there starts a run of its own:
    // the call the tool is given is how it stays in the run it is in.
    let apart = rdm.copy().model(client(|_, _, _| true)).max_depth(1);
    let captured = apart.clone();
    apart.tools().insert(Tool::new("again", "", move |call| {
        let alone = captured.call_with(Ask::new(inputs! { "n" => 2 }).answers_only())?;
        Ok((alone.trajectory.len(), call.sub(inputs! { "n" => 3 })?.trajectory.len()))
    }));
    let result = apart.call(inputs! { "n" => 1 }).unwrap();
    assert_eq!((result.observations["again"].clone(), result.trajectory.len()), (json!([1, 0]), 2));
}

#[test]
fn a_prediction_reads_as_the_python_one_does() {
    let model = client(|_, name, _| {
        if name == "frustration" {
            Say::Level(2)
        } else {
            Say::Yes(name == "refund")
        }
    });
    let rdm = Rdm::new(triage())
        .tool(Tool::new("refund", "", |_| Ok("refund started")))
        .model(model);
    let result: Prediction = rdm.call(inputs! { "ticket" => "t" }).unwrap();
    let plain = json!(result);
    let keys: Vec<&str> = plain.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "refund",
            "escalate",
            "frustration",
            "observations",
            "termination_reason",
            "trajectory"
        ]
    );
    assert_eq!(plain["refund"]["value"], true);
    assert_eq!(
        (plain["termination_reason"].clone(), plain["trajectory"].as_array().unwrap().len()),
        (json!("final"), 1)
    );
    assert_eq!(
        (result.value("refund"), result.value("frustration"), result.value("nothing")),
        (json!(true), json!(2.0), json!(null))
    );
    assert_eq!(result.get("frustration").unwrap().confidence(), 1.0);
    assert_eq!(result.observation("refund"), Some(&json!("refund started")));
    assert!(result.answered() && result.steps("refund").is_empty());
    // An entry read back from JSON is the entry.
    let read: Vec<rdm::Entry> = serde_json::from_value(plain["trajectory"].clone()).unwrap();
    assert_eq!(read, result.trajectory);
}

#[test]
fn the_pointer_at_a_leaf_is_answered_and_nothing_is_run() {
    let counter = Shared::new(json!({"n": 0}));
    // The tool pointed at is kept from running: that is the depth limit, and the call says so.
    let rdm = Rdm::new(tick())
        .tool(ticking(&counter))
        .max_depth(0)
        .model(client(|_, _, _| "tick"));
    let result = rdm.call(inputs! { "counter" => &counter }).unwrap();
    assert_eq!(
        (
            result.chosen("next"),
            result.observations["next"].clone(),
            result.termination_reason
        ),
        (Some("tick"), json!([]), Reason::MaxDepth)
    );
    assert_eq!(counter.get()["n"], 0);
    // Pointing at `finish` there keeps nothing from being done.
    rdm.set_model(client(|_, _, _| "finish"));
    let result = rdm.call(inputs! { "counter" => &counter }).unwrap();
    assert_eq!(
        (
            result.chosen("next"),
            result.observations["next"].clone(),
            result.termination_reason
        ),
        (Some("finish"), json!([]), Reason::Final)
    );
}

#[test]
fn a_request_is_counted_before_what_its_fit_asks() {
    // The request being fitted has its place before the fit's own questions are asked, so they
    // cannot use it up: with one request allowed, it is the fit's question that is refused.
    let length = || {
        Signature::new("Judge a length.")
            .input_typed("length", "int")
            .noul("long", "Is `inputs.length` more than forty?")
    };
    let fit = move |signature: Signature, inputs: rdm::Inputs, call: &rdm::Call| {
        if signature.outputs().contains_key("long") {
            return Ok((signature, inputs));
        }
        let asked = call.sub_with(Ask::new(inputs! { "length" => 48 }).signature(length()).answers_only())?;
        assert_eq!(
            (asked.answered(), asked.termination_reason, call.requests_left()),
            (false, Reason::MaxPredictorCalls, 0)
        );
        Ok((signature, inputs))
    };
    let model = client(|_, name, _| if name == "frustration" { Say::Level(0) } else { Say::Yes(false) });
    let result = Rdm::new(triage())
        .fit_fn(fit)
        .max_predictor_calls(1)
        .model(model.clone())
        .call(inputs! { "ticket" => "t" })
        .unwrap();
    assert_eq!(asked(&model), [["refund", "escalate", "frustration"]]);
    let stopped: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.parent, entry.skipped))
        .collect();
    assert_eq!(stopped, [(0, None, None), (1, Some(0), Some(Reason::MaxPredictorCalls))]);
    assert_eq!(result.termination_reason, Reason::MaxPredictorCalls);
}

#[test]
fn tokens_are_looked_at_again_just_before_a_request_is_asked() {
    // Two sub-calls are admitted while there are tokens left. The first uses them, so the second
    // is not asked, and the place it had among the requests is given back.
    let left = Shared::new(0);
    let noted = left.clone();
    let again = Tool::new("again", "", move |call| {
        let n: i64 = call.arg("n")?;
        let below = call.batch([n + 1, n + 2].map(|n| Ask::new(inputs! { "n" => n }).answers_only()))?;
        noted.set(call.requests_left());
        Ok(below.iter().map(|below| below.termination_reason).collect::<Vec<_>>())
    });
    let model = client_using(100, |_, _, _| true);
    // One at a time, so that which of the two is asked does not depend on timing.
    let rdm = Rdm::new(count()).tool(again).max_tokens(150).threads(1).model(model.clone());
    let result = rdm.call(inputs! { "n" => 0 }).unwrap();
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.skipped).collect::<Vec<_>>(),
        [None, None, Some(Reason::MaxTokens)]
    );
    assert_eq!(json!(result.observations), json!({"again": ["final", "max_tokens"]}));
    assert_eq!(
        (model.requests().len(), left.get(), result.termination_reason),
        (2, 254, Reason::MaxTokens)
    );
}

#[test]
fn a_choice_a_fit_leaves_one_option_is_answered_without_asking_and_takes_no_place() {
    // A fit that cuts the teams down to the one a ticket about a charge can go to.
    let cut = |signature: Signature, inputs: rdm::Inputs, _: &rdm::Call| {
        let cut = signature.output("team").is_some();
        Ok((
            if cut {
                signature.with_options("team", ["billing"])
            } else {
                signature
            },
            inputs,
        ))
    };
    let route = || {
        Signature::new("Route a ticket.")
            .input("ticket")
            .choice("team", "Which team should take the ticket?", ["billing", "support"])
    };
    let model = client(|_, name, _| match name {
        "team" => Say::Pick("support".to_owned()),
        _ => Say::Yes(true),
    });

    // Beside another question, only the other is asked, and the one option is the answer, as certain.
    let both = route().noul("refund", "Is the customer asking for a refund?");
    let result = Rdm::new(both)
        .fit_fn(cut)
        .model(model.clone())
        .call(inputs! { "ticket" => "I was charged twice." })
        .unwrap();
    assert!(model.requests()[0].questions.keys().eq(["refund"]));
    assert_eq!(
        json!(result.choice("team")),
        json!({"confidence": 1.0, "value": "billing", "probabilities": {"billing": 1.0}})
    );
    assert!(result.yes("refund"));
    let entry = &result.trajectory[0];
    assert_eq!(
        (entry.unasked.clone(), entry.model.as_deref()),
        (Some(vec!["team".to_owned()]), Some("jev-test"))
    );
    assert_eq!(entry.outputs.as_ref().unwrap().keys().collect::<Vec<_>>(), ["team", "refund"]);

    // With nothing else to ask, nothing is sent, and no limit is charged: three such requests
    // under a request that was asked, in a run that may ask two.
    let model = client(|_, _, _| Say::Yes(true));
    let went = Shared::new(Vec::new());
    let noted = went.clone();
    let each = Tool::new("each", "Route each ticket.", move |call| {
        for ticket in ["one", "two", "three"] {
            let below = call.sub_with(Ask::new(inputs! { "ticket" => ticket }).signature(route()))?;
            noted
                .lock()
                .push((below.choice("team").map(|team| team.value.clone()), below.termination_reason));
        }
        Ok(())
    });
    let all = Signature::new("Route the tickets.")
        .input("ticket")
        .noul("each", "Should each ticket be routed?");
    let rdm = Rdm::new(all).tool(each).fit_fn(cut).max_predictor_calls(2).model(model.clone());
    let result = rdm.call(inputs! { "ticket" => "all of them" }).unwrap();
    assert_eq!(went.get(), vec![(Some("billing".to_owned()), Reason::Final); 3]);
    assert_eq!((model.requests().len(), result.termination_reason), (1, Reason::Final));
    for entry in &result.trajectory[1..] {
        assert_eq!(
            (entry.unasked.clone(), entry.tokens, entry.model.clone(), entry.skipped),
            (Some(vec!["team".to_owned()]), Some(0), None, None)
        );
    }
}
