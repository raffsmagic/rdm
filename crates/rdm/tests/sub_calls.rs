//! Sub-calls: a tool calls the module, and the answers come back to the tool.

mod common;

use common::{Say, asked, at_depth, client, client_using, shown};
use indexmap::IndexMap;
use rdm::systemone::{Answer, Params, Question, Scripted};
use rdm::{Ask, Call, Error, Inputs, Json, Rdm, Reason, Shared, Signature, Tool, inputs};
use serde_json::json;

fn find() -> Signature {
    Signature::new("Find out whether an archive is worth searching for a question.")
        .input_typed("archive", "dict")
        .input("question")
        .noul("look", "Could an archive of this kind answer `inputs.question`?")
}

fn judge() -> Signature {
    Signature::new("Judge one record of an archive against a question.")
        .input("record")
        .input("question")
        .noul("answers", "Does `inputs.record` answer `inputs.question`?")
}

fn triage() -> Signature {
    Signature::new("Triage a support ticket.")
        .input("ticket")
        .noul("refund", "Is the customer asking for a refund?")
        .noul("escalate", "Does the ticket need an engineer?")
        .score("frustration", "How frustrated is the customer?", ["calm", "frustrated", "angry"])
}

fn home() -> Signature {
    Signature::new("Carry out a request to a smart home.")
        .input("request")
        .noul("split", "Does `inputs.request` ask for more than one distinct action?")
        .noul("lock", "Does `inputs.request` ask only for a door to be locked?")
        .noul("chat", "Is `inputs.request` only a question to answer in conversation?")
}

/// Decides as the smart home's decision model would, from the words of the request.
fn at_home(inputs: &Json, name: &str, _: &Question) -> bool {
    let request = inputs["request"].as_str().unwrap();
    let several = request.contains(" and ");
    match name {
        "split" => several,
        "lock" => request.to_lowercase().contains("lock") && !several,
        _ => request.ends_with('?') && !several,
    }
}

fn lock() -> Tool {
    Tool::new("lock", "", |_| Ok("locked"))
}

fn links(result: &rdm::Prediction) -> Vec<(usize, Option<usize>, Option<&str>)> {
    result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.parent, entry.tool_name.as_deref()))
        .collect()
}

#[test]
fn a_tool_calls_the_module_on_a_piece_and_the_answers_come_back_to_the_tool() {
    let archive = json!({"a": "The hangar opens at dawn.", "b": "The hangar door sticks in frost.", "c": "Tides turn at noon."});
    // Judge each record that mentions the hangar, in a sub-call of its own, and keep those that answer.
    let look = Tool::new("look", "", |call| {
        let (archive, question): (IndexMap<String, String>, String) = (call.arg("archive")?, call.arg("question")?);
        let mut kept = Vec::new();
        for record in archive.values().filter(|text| text.contains("hangar")) {
            let verdict = call.sub_with(Ask::new(inputs! { "record" => record, "question" => &question }).signature(judge()))?;
            if verdict.yes("answers") {
                kept.push(record.clone());
            }
        }
        Ok(kept)
    });
    let model = client(|inputs, name, _| name == "look" || inputs["record"].as_str().unwrap().contains("dawn"));
    let rdm = Rdm::new(find()).tool(look).model(model.clone());
    let result = rdm
        .call(inputs! { "archive" => archive, "question" => "When does the hangar open?" })
        .unwrap();
    // The tool called the module twice. Each is a sub-call: one level deeper, with its own small
    // state and its own question, asked once, and recorded under the tool that made it.
    assert_eq!(
        links(&result),
        [(0, None, None), (1, Some(0), Some("look")), (1, Some(0), Some("look"))]
    );
    assert_eq!(
        shown(&model)[1..],
        [
            json!({"record": "The hangar opens at dawn.", "question": "When does the hangar open?"}),
            json!({"record": "The hangar door sticks in frost.", "question": "When does the hangar open?"}),
        ]
    );
    assert_eq!(asked(&model), [["look"], ["answers"], ["answers"]]);
    // Their answers came back to the tool, which decided in code what to make of them.
    assert_eq!(json!(result.observations), json!({"look": ["The hangar opens at dawn."]}));
    assert_eq!(
        at_depth(&result, 1).iter().map(|entry| entry.value("answers")).collect::<Vec<_>>(),
        [true, false]
    );
    // A sub-call returns what any call returns, without the run's trajectory.
    let alone = Rdm::new(judge())
        .model(model)
        .call(inputs! { "record" => "x", "question" => "q" })
        .unwrap();
    assert_eq!(
        (alone.termination_reason, alone.observations.len(), alone.trajectory.len()),
        (Reason::Final, 0, 1)
    );
}

#[test]
fn sub_calls_made_through_batch_join_the_run_too() {
    let archive: IndexMap<String, String> = (0..6).map(|n| (n.to_string(), format!("Record {n} about the hangar."))).collect();
    let look = Tool::new("look", "", |call| {
        let (archive, question): (IndexMap<String, String>, String) = (call.arg("archive")?, call.arg("question")?);
        let asks = archive
            .values()
            .map(|record| Ask::new(inputs! { "record" => record, "question" => &question }).signature(judge()));
        let verdicts = call.batch(asks)?;
        // One a limit refused has no answers, and says which limit.
        assert!(
            verdicts
                .iter()
                .all(|verdict| verdict.answered() || verdict.termination_reason == "max_predictor_calls")
        );
        assert!(verdicts.iter().all(|verdict| verdict.trajectory.is_empty()));
        Ok(verdicts.iter().filter(|verdict| verdict.yes("answers")).count())
    });
    let rdm = Rdm::new(find()).tool(look).max_predictor_calls(5).model(client(|_, _, _| true));
    let result = rdm.call(inputs! { "archive" => &archive, "question" => "q" }).unwrap();
    // Six sub-calls at once, on the run's own count of requests: one for the root, four answered, two refused.
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>(),
        [0, 1, 1, 1, 1, 1, 1]
    );
    // They were admitted in the order given, so it is the last two a limit stopped, whatever ran first.
    let stopped: Vec<bool> = result
        .trajectory
        .iter()
        .map(|entry| entry.skipped == Some(Reason::MaxPredictorCalls))
        .collect();
    assert_eq!(stopped, [false, false, false, false, false, true, true]);
    assert_eq!(result.trajectory[5].inputs["record"], "Record 4 about the hangar.");
    assert_eq!(json!(result.observations), json!({"look": 4}));
    assert_eq!(result.termination_reason, "max_predictor_calls");
}

#[test]
fn a_call_returns_its_own_answers_and_what_its_tools_came_to() {
    let claim = Signature::new("Decide whether a claim holds.")
        .input("claim")
        .noul("holds", "Is `inputs.claim` true?")
        .noul("ledger", "Would the ledger settle `inputs.claim`?");
    let checked = Signature::new("Decide whether a claim holds, given an entry.")
        .input("claim")
        .input("entry")
        .noul("holds", "Does `inputs.entry` show that `inputs.claim` is true?");
    let ledger = Tool::new("ledger", "", move |call| {
        let claim: String = call.arg("claim")?;
        call.sub_with(Ask::new(inputs! { "claim" => claim, "entry" => "Paid in full on 3 May." }).signature(checked.clone()))
    });
    let model = client(|inputs, name, _| name == "ledger" || inputs["entry"].as_str().is_some_and(|entry| entry.contains("Paid")));
    let result = Rdm::new(claim)
        .tool(ledger)
        .model(model)
        .call(inputs! { "claim" => "The invoice was paid." })
        .unwrap();
    // The first request could not tell, and the ledger's sub-call could. Each answer stays with the
    // request that gave it: a judgment made on less is not overwritten by one made on more.
    assert_eq!((result.yes("holds"), result.yes("ledger")), (false, true));
    // What the tool returned is the sub-call's prediction, as plain data.
    let below = &result.observations["ledger"];
    assert_eq!(below["holds"]["value"], true);
    assert_eq!(
        (&below["termination_reason"], &below["observations"], below.get("trajectory")),
        (&json!("final"), &json!({}), None)
    );
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| (entry.depth, entry.value("holds")))
            .collect::<Vec<_>>(),
        [(0, json!(false)), (1, json!(true))]
    );
}

#[test]
fn a_tool_can_put_each_piece_of_a_request_to_the_module_one_level_down() {
    let done = Shared::new(Vec::new());
    let locked = done.clone();
    // What a generative model would write: the request's pieces. Each is a request of its own.
    let split = Tool::new("split", "", |call| {
        let request: String = call.arg("request")?;
        request
            .split(" and ")
            .map(|piece| call.sub(inputs! { "request" => piece }))
            .collect::<rdm::Result<Vec<_>>>()
    });
    let lock = Tool::new("lock", "", move |call| {
        locked.lock().push(call.arg::<String>("request")?);
        Ok("locked")
    });
    let chat = Tool::new("chat", "", |_| Ok("It will rain."));
    let model = client(at_home);
    let rdm = Rdm::new(home()).tool(split).tool(lock).tool(chat).model(model.clone());
    let result = rdm
        .call(inputs! { "request" => "Lock the front door and will it rain? and lock the shed" })
        .unwrap();
    // The decision model decided once that the request has several parts, the tool made three
    // states, and each was decided by the same questions one level down. The tool chose nothing.
    let split = Some("split");
    assert_eq!(
        links(&result),
        [(0, None, None), (1, Some(0), split), (1, Some(0), split), (1, Some(0), split)]
    );
    assert!(asked(&model).iter().all(|names| names == &["split", "lock", "chat"]) && model.requests().len() == 4);
    let pieces: Vec<&str> = result.trajectory[1..]
        .iter()
        .map(|entry| entry.inputs["request"].as_str().unwrap())
        .collect();
    assert_eq!(pieces, ["Lock the front door", "will it rain?", "lock the shed"]);
    // The tool comes to what its pieces came to, in order, and each of those is what any call returns.
    let made: Vec<&Json> = result.observations["split"]
        .as_array()
        .unwrap()
        .iter()
        .map(|child| &child["observations"])
        .collect();
    assert_eq!(
        made,
        [
            &json!({"lock": "locked"}),
            &json!({"chat": "It will rain."}),
            &json!({"lock": "locked"})
        ]
    );
    assert_eq!(done.get(), ["Lock the front door", "lock the shed"]);
    assert_eq!(result.termination_reason, Reason::Final);
}

#[test]
fn what_a_tool_uses_of_a_model_of_its_own_is_counted_with_the_run() {
    // A generative model writes the pieces, and reports what it used.
    let split = Tool::new("split", "", |call| {
        call.spend(70);
        call.spend(30);
        let request: String = call.arg("request")?;
        request
            .split(" and ")
            .map(|piece| call.sub(inputs! { "request" => piece }))
            .collect::<rdm::Result<Vec<_>>>()
    });
    let rdm = Rdm::new(home()).tool(split).tool(lock()).model(client_using(10, at_home));
    let result = rdm.call(inputs! { "request" => "lock the gate and lock the shed" }).unwrap();
    // The request's own tokens and the tool's are recorded apart, on the request whose answer called the tool.
    assert_eq!(
        (result.trajectory[0].tokens, json!(result.trajectory[0].tool_tokens)),
        (Some(10), json!({"split": 100}))
    );
    assert!(result.trajectory.iter().all(|entry| entry.model.as_deref() == Some("jev-test")));
    assert!(
        result.trajectory[1..]
            .iter()
            .all(|entry| entry.tool_tokens.is_none() && entry.tokens == Some(10))
    );

    // Both count against max_tokens: here the tool's use leaves nothing for the next turn.
    let steps = Signature::new("Take steps.")
        .input("task")
        .input("observation")
        .tool("next", "What next?");
    let write = Tool::new("write", "", |call| {
        call.spend(100);
        Ok("written")
    });
    let limited = Rdm::new(steps)
        .tool(write)
        .max_tokens(100)
        .model(client_using(10, |_, _, _| "write"));
    let result = limited.call(inputs! { "task" => "t" }).unwrap();
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.skipped).collect::<Vec<_>>(),
        [None, Some(Reason::MaxTokens)]
    );
    assert_eq!(result.termination_reason, "max_tokens");
}

#[test]
fn what_a_request_uses_is_counted_once_however_deep_the_tool_that_asked_it() {
    let count = Signature::new("Count.").input_typed("n", "int").noul("again", "Again?");
    // Three sub-calls, each of which makes one more, and a model of the tool's own at the top.
    let again = Tool::new("again", "", |call| {
        if call.depth() == 0 {
            call.spend(7);
        }
        let n: i64 = call.arg("n")?;
        let mut below = Vec::new();
        for i in 0..(if n == 0 { 3 } else { 1 }) {
            below.push(call.sub(inputs! { "n" => 10 * n + i + 1 })?.yes("again"));
        }
        Ok(below)
    });
    let rdm = Rdm::new(count)
        .tool(again)
        .max_depth(2)
        .max_tokens(707)
        .model(client_using(100, |_, _, _| true));
    let result = rdm.call(inputs! { "n" => 0 }).unwrap();
    // Seven requests of 100 tokens. A tool's tokens are what its own model used, not what the requests it asked used.
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.tokens).collect::<Vec<_>>(),
        [Some(100); 7]
    );
    assert_eq!(result.termination_reason, "max_depth");
    let own: Vec<Json> = result.trajectory.iter().map(|entry| json!(entry.tool_tokens)).collect();
    assert_eq!(own[0], json!({"again": 7}));
    assert!(own[1..].iter().all(Json::is_null));
    // Each use is counted once in the run's total as well.
    assert_eq!((result.tokens(), result.asked(), result.trajectory[0].spent()), (707, 7, 107));
}

#[test]
fn a_request_already_asked_on_the_way_is_not_asked_again() {
    let wait = || {
        Signature::new("Wait for a door to open.")
            .input("door")
            .noul("wait", "Is `inputs.door` still shut?")
    };
    // Changes nothing, and asks the same question again.
    let same = Tool::new("wait", "", |call| call.sub(inputs! { "door" => call.arg::<String>("door")? }));
    // Asked again it would get the same answer and select the same tool, so it is not asked: the
    // run has come round, and says so.
    let model = client(|_, _, _| true);
    let result = Rdm::new(wait())
        .tool(same)
        .model(model.clone())
        .call(inputs! { "door" => "shut" })
        .unwrap();
    assert_eq!(model.requests().len(), 1);
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| (entry.depth, entry.repeat))
            .collect::<Vec<_>>(),
        [(0, false), (1, true)]
    );
    assert_eq!(result.termination_reason, "repeat");
    // The repeat returns the earlier answers, and they are recorded on it, with nothing asked and nothing run.
    assert_eq!(result.observations["wait"]["wait"]["value"], true);
    assert_eq!(result.observations["wait"]["termination_reason"], "repeat");
    let repeat = &result.trajectory[1];
    assert_eq!(
        (repeat.value("wait"), repeat.tokens, &repeat.model, &repeat.observations),
        (json!(true), None, &None, &None)
    );

    // The same holds further down: a state that comes back as its own descendant is a cycle.
    let split = Tool::new("split", "", |call| {
        Ok([
            call.sub(inputs! { "request" => "lock the gate and lock the shed" })?,
            call.sub(inputs! { "request" => "lock the shed" })?,
        ])
    });
    let model = client(at_home);
    let rdm = Rdm::new(home()).tool(split).tool(lock()).max_depth(9).model(model.clone());
    let result = rdm.call(inputs! { "request" => "lock the gate and lock the shed" }).unwrap();
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| (entry.depth, entry.repeat))
            .collect::<Vec<_>>(),
        [(0, false), (1, true), (1, false)]
    );
    assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::Repeat));

    // When a limit stopped something as well, the limit is the reason: it bounds more than a repeat does.
    let stay_and_knock = Tool::new("wait", "", |call| {
        call.sub(inputs! { "door" => "another door" })?;
        call.sub(inputs! { "door" => call.arg::<String>("door")? })
    });
    let rdm = Rdm::new(wait())
        .tool(stay_and_knock)
        .max_predictor_calls(1)
        .model(client(|_, _, _| true));
    let result = rdm.call(inputs! { "door" => "shut" }).unwrap();
    let cut: Vec<(Option<Reason>, bool)> = result.trajectory.iter().map(|entry| (entry.skipped, entry.repeat)).collect();
    assert_eq!(cut, [(None, false), (Some(Reason::MaxPredictorCalls), false), (None, true)]);
    assert_eq!(result.termination_reason, "max_predictor_calls");

    // Two states that are alike but not on each other's way are both asked.
    let twice = Tool::new("split", "", |call| {
        Ok([
            call.sub(inputs! { "request" => "lock up" })?,
            call.sub(inputs! { "request" => "lock up" })?,
        ])
    });
    let model = client(|inputs, name, _| name == "split" && inputs["request"] == "twice");
    let result = Rdm::new(home())
        .tool(twice)
        .model(model)
        .call(inputs! { "request" => "twice" })
        .unwrap();
    assert_eq!(
        result.trajectory.iter().map(|entry| entry.repeat).collect::<Vec<_>>(),
        [false, false, false]
    );
    assert_eq!(result.termination_reason, "final");
}

#[test]
fn what_makes_a_request_the_same_is_the_module_the_questions_and_the_state() {
    let doors = || {
        Signature::new("Wait.")
            .input_typed("door", "dict")
            .noul("wait", "Is any door shut?")
    };
    // The entries of a dict are part of what a request shows, in the order given.
    let turned = Tool::new("wait", "", |call| {
        call.sub(inputs! { "door" => json!({"b": "back", "1": "front"}) })
    });
    let model = client(|_, _, _| true);
    let result = Rdm::new(doors())
        .tool(turned)
        .max_depth(1)
        .model(model.clone())
        .call(inputs! { "door" => json!({"1": "front", "b": "back"}) })
        .unwrap();
    assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::MaxDepth));

    // The same inputs under other options are another request: the options are part of the question.
    let ways = |options: &[&str]| {
        Signature::new("Leave.")
            .input("room")
            .choice("go", "Which way?", options.iter().copied())
    };
    let other = ways(&["north", "east"]);
    let go = Tool::new("go", "", move |call| {
        call.sub_with(Ask::new(inputs! { "room" => "hall" }).signature(other.clone()).max_depth(0))
    });
    let model = client(|_, _, _| "north");
    let result = Rdm::new(ways(&["north", "south"]))
        .tool(go)
        .model(model.clone())
        .call(inputs! { "room" => "hall" })
        .unwrap();
    assert_eq!((model.requests().len(), result.termination_reason), (2, Reason::Final));

    // Another module asking the same questions about the same state is not a repeat of this one's request.
    let inner = Rdm::new(doors()).model(client(|_, _, _| true));
    let outer = Rdm::new(doors())
        .tool(Tool::module("wait", "", &inner))
        .model(client(|_, _, _| true));
    let result = outer.call(inputs! { "door" => json!({}) }).unwrap();
    assert_eq!(
        (result.trajectory.len(), result.trajectory[1].repeat, result.termination_reason),
        (2, false, Reason::Final)
    );
    // A copy of a module is a module of its own, and a clone is the same one.
    assert!(outer.copy().id() != outer.id() && outer.clone().id() == outer.id());
}

#[test]
fn fit_is_given_each_request_before_it_is_asked() {
    let length = || {
        Signature::new("Judge the length of a ticket.")
            .input_typed("ticket_length", "int")
            .noul("long", "Is `inputs.ticket_length` more than forty?")
    };
    let seen = Shared::new(Vec::new());
    let noted = seen.clone();
    // Shows the ticket's first line only, after asking a question of its own about the whole of it.
    let fit = move |signature: Signature, inputs: Inputs, call: &Call| {
        noted
            .lock()
            .push((signature.outputs().keys().cloned().collect::<Vec<_>>(), call.depth()));
        assert!(call.prediction().is_none() && call.tool().is_none());
        let ticket = inputs.value("ticket").and_then(|ticket| ticket.as_str().map(str::to_owned));
        let Some(ticket) = ticket.filter(|ticket| ticket.contains('\n')) else {
            return Ok((signature, inputs));
        };
        let long = call
            .sub_with(
                Ask::new(inputs! { "ticket_length" => ticket.len() })
                    .signature(length())
                    .max_depth(0),
            )?
            .yes("long");
        let first = ticket.lines().next().unwrap_or_default();
        Ok((
            signature,
            inputs.with("ticket", format!("{first}{}", if long { " [...]" } else { "" })),
        ))
    };
    let called = Shared::new(Vec::new());
    let given = called.clone();
    let refund = Tool::new("refund", "", move |call| {
        given.lock().extend([
            call.arg::<String>("ticket")?,
            call.inputs().value("ticket").unwrap().as_str().unwrap().to_owned(),
        ]);
        Ok(())
    });
    let model = client(|_, name, _| match name {
        "refund" | "long" => Say::Yes(true),
        "escalate" => Say::Yes(false),
        _ => Say::Level(0),
    });
    let whole = "I was charged twice.\nOrder 98423, last Thursday.";
    let rdm = Rdm::new(triage()).tool(refund).fit_fn(fit).model(model.clone());
    let result = rdm.call(inputs! { "ticket" => whole }).unwrap();
    // What fit returned is what the request showed. What it asked first is a sub-call of that request.
    assert_eq!(
        shown(&model),
        [json!({"ticket_length": 48}), json!({"ticket": "I was charged twice. [...]"})]
    );
    assert_eq!(result.trajectory[0].inputs, json!({"ticket": "I was charged twice. [...]"}));
    let recorded: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| {
            (
                entry.depth,
                entry.parent,
                entry.tool_name.clone(),
                entry.outputs.as_ref().unwrap().keys().cloned().collect::<Vec<_>>(),
            )
        })
        .collect();
    let names = |names: &[&str]| names.iter().map(|name| name.to_string()).collect::<Vec<_>>();
    assert_eq!(
        recorded,
        [
            (0, None, None, names(&["refund", "escalate", "frustration"])),
            (1, Some(0), None, names(&["long"]))
        ]
    );
    // It is given every request, its own included. A tool is given the call's inputs whole: fit
    // changes what the model is shown, not what the code acts on.
    assert_eq!(
        seen.get(),
        [(names(&["refund", "escalate", "frustration"]), 0), (names(&["long"]), 1)]
    );
    assert_eq!(called.get(), [whole, whole]);

    // What a fit uses on a model of its own is counted with the run, on the request it was fitting.
    let rewrite = |signature: Signature, inputs: Inputs, call: &Call| {
        call.spend(70);
        Ok((signature, inputs))
    };
    let quiet = client_using(
        100,
        |_, name, _| if name == "frustration" { Say::Level(0) } else { Say::Yes(false) },
    );
    let result = Rdm::new(triage())
        .fit_fn(rewrite)
        .max_tokens(170)
        .model(quiet)
        .call(inputs! { "ticket" => "Thanks!" })
        .unwrap();
    assert_eq!(
        (result.trajectory[0].tokens, result.trajectory[0].fit_tokens),
        (Some(100), Some(70))
    );
    let keys: Vec<String> = json!(result.trajectory[0]).as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        keys,
        [
            "depth",
            "iteration",
            "parent",
            "tool_name",
            "inputs",
            "fit_tokens",
            "tokens",
            "model",
            "outputs",
            "observations"
        ]
    );

    // Several are applied in the order given, and one that loses an output is a mistake of the program.
    let first = |signature: Signature, inputs: Inputs, _: &Call| Ok((signature, inputs.with("ticket", "first")));
    let second = |signature: Signature, inputs: Inputs, _: &Call| {
        let ticket = inputs.value("ticket").unwrap();
        Ok((
            signature,
            inputs.with("ticket", format!("{} then second", ticket.as_str().unwrap())),
        ))
    };
    let model = client(|_, name, _| if name == "frustration" { Say::Level(0) } else { Say::Yes(false) });
    Rdm::new(triage())
        .fit_fn(first)
        .fit_fn(second)
        .model(model.clone())
        .call(inputs! { "ticket" => "t" })
        .unwrap();
    assert_eq!(shown(&model), [json!({"ticket": "first then second"})]);
    let loses = |signature: Signature, inputs: Inputs, _: &Call| Ok((signature.without("escalate"), inputs));
    let error = Rdm::new(triage())
        .fit_fn(loses)
        .model(model)
        .call(inputs! { "ticket" => "t" })
        .unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("without the output 'escalate'"));
}

fn taxonomy() -> Json {
    json!({
        "Animals & Pet Supplies": {"Cat Supplies": {"Cat Beds": {}, "Cat Window Perches": {}}, "Bird Supplies": {"Bird Perches": {}}},
        "Furniture": {"Beds": {"Bunk Beds": {}}, "Shelving": {"Wall Shelves": {}, "Bookcases": {}}},
    })
}

fn children(path: &[String]) -> Vec<String> {
    let whole = taxonomy();
    let node = path.iter().fold(&whole, |node, label| &node[label.as_str()]);
    node.as_object()
        .map(|children| children.keys().cloned().collect())
        .unwrap_or_default()
}

/// The question at one node of the taxonomy: its options are that node's children, known only once it is reached.
fn ask(path: &[String]) -> Signature {
    Signature::new("Classify a product listing.")
        .input("listing")
        .input_with("path", "list", "The categories chosen so far.")
        .choice("descend", "Which direct child category best matches the listing?", children(path))
}

/// Go into the chosen category and ask about its own children, one level down. Comes to the path to a leaf.
fn descend() -> Tool {
    Tool::new("descend", "", |call| {
        let (mut path, listing): (Vec<String>, String) = (call.arg("path")?, call.arg("listing")?);
        path.push(call.arg("descend")?);
        if children(&path).is_empty() {
            return Ok(json!(path));
        }
        let below = call.sub_with(Ask::new(inputs! { "listing" => listing, "path" => &path }).signature(ask(&path)))?;
        Ok(below.observations["descend"].clone())
    })
}

/// Chooses the option that shares the most words with the listing.
fn pick(inputs: &Json, _: &str, question: &Question) -> String {
    let listing = inputs["listing"].as_str().unwrap().to_lowercase();
    let words: Vec<&str> = listing.split_whitespace().collect();
    let shared = |label: &String| {
        label
            .to_lowercase()
            .replace('&', "")
            .split_whitespace()
            .filter(|word| words.contains(word))
            .count()
    };
    let labels = question.labels();
    let most = labels.iter().map(shared).max().unwrap();
    labels.iter().find(|label| shared(label) == most).unwrap().clone()
}

const LISTING: &str = "A padded cat window perch with suction cups, in the pet supplies aisle.";

#[test]
fn a_decision_can_be_about_which_decision_comes_next() {
    let model = client(pick);
    let rdm = Rdm::new(ask(&[])).tool(descend()).model(model.clone());
    let result = rdm.call(inputs! { "listing" => LISTING, "path" => json!([]) }).unwrap();
    // Three requests, each with an answer space that did not exist until the one before was answered.
    let offered: Vec<Vec<String>> = model
        .requests()
        .iter()
        .map(|request| request.questions["descend"].labels())
        .collect();
    assert_eq!(
        offered,
        [
            ["Animals & Pet Supplies", "Furniture"],
            ["Cat Supplies", "Bird Supplies"],
            ["Cat Beds", "Cat Window Perches"]
        ]
    );
    // Each is a sub-call of the one before, and carries the path so far: state the tool built.
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| (entry.depth, entry.parent))
            .collect::<Vec<_>>(),
        [(0, None), (1, Some(0)), (2, Some(1))]
    );
    let paths: Vec<Json> = shown(&model).iter().map(|inputs| inputs["path"].clone()).collect();
    assert_eq!(
        paths,
        [
            json!([]),
            json!(["Animals & Pet Supplies"]),
            json!(["Animals & Pet Supplies", "Cat Supplies"])
        ]
    );
    // The call's own answer is its own choice. What the tool came to is where the descent ended.
    assert_eq!(result.chosen("descend"), Some("Animals & Pet Supplies"));
    assert_eq!(
        result.observations["descend"],
        json!(["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"])
    );
}

#[test]
fn parameters_fitted_to_one_answer_space_are_not_applied_to_another() {
    // `descend` is asked at every node of the taxonomy, each time over other options. Weights
    // set for the root's options apply to the root, and the requests below are answered without them.
    let rdm = Rdm::new(ask(&[])).tool(descend()).model(client(pick));
    rdm.set_params("descend", Params::weights([("Animals & Pet Supplies", 1.0), ("Furniture", 0.5)]));
    let result = rdm.call(inputs! { "listing" => LISTING, "path" => json!([]) }).unwrap();
    assert_eq!(
        result.observations["descend"],
        json!(["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"])
    );
    // And at the root they decide: with nothing on the first option, the second is chosen.
    rdm.set_params("descend", Params::weights([("Animals & Pet Supplies", 0.0), ("Furniture", 1.0)]));
    rdm.set_model(Scripted::new(|_| {
        [("descend", Answer::choice([("Animals & Pet Supplies", 0.6), ("Furniture", 0.4)]))]
    }));
    let result = rdm
        .max_depth(0)
        .call(inputs! { "listing" => LISTING, "path" => json!([]) })
        .unwrap();
    assert_eq!(
        (
            result.chosen("descend"),
            result.choice("descend").unwrap().probabilities["Furniture"]
        ),
        (Some("Furniture"), 0.4)
    );
}

#[test]
fn a_threshold_belongs_to_an_output_name_whichever_request_asks_it() {
    let start = Signature::new("Decide whether to look.")
        .input("question")
        .noul("look", "Is it worth looking?");
    let check = Signature::new("Check a note.")
        .input("note")
        .noul("look", "Is it worth looking further?")
        .noul("useful", "Is the note useful?");
    let look = Tool::new("look", "", move |call| {
        call.sub_with(Ask::new(inputs! { "note" => format!("note {}", call.requests()) }).signature(check.clone()))
    });
    let rdm = Rdm::new(start).tool(look).max_depth(4).model(client(|_, _, _| 0.6));
    assert_eq!(rdm.call(inputs! { "question" => "q" }).unwrap().trajectory.len(), 5);
    // The threshold is set once, on the module, and applies wherever `look` is asked: in the
    // first request and in requests whose questions were built at run time.
    rdm.set_params("look", Params::threshold(0.7));
    assert_eq!(rdm.call(inputs! { "question" => "q" }).unwrap().trajectory.len(), 1);
    // One can be set for an output that only a later request asks, and it applies there.
    rdm.unset_params("look");
    rdm.set_params("useful", Params::threshold(0.7));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(
        (result.trajectory[1].value("useful"), result.trajectory[1].value("look")),
        (json!(false), json!(true))
    );
    assert_eq!(rdm.all_params().keys().collect::<Vec<_>>(), ["useful"]);
}

#[test]
fn parameters_for_the_questions_of_a_sub_call_are_the_modules_too() {
    let look = Tool::new("look", "", |call| {
        let (archive, question): (IndexMap<String, String>, String) = (call.arg("archive")?, call.arg("question")?);
        let verdicts = call.batch(
            archive
                .values()
                .map(|record| Ask::new(inputs! { "record" => record, "question" => &question }).signature(judge())),
        )?;
        Ok(verdicts.iter().map(|verdict| verdict.yes("answers")).collect::<Vec<_>>())
    });
    // `answers` is asked only in sub-calls. Its threshold is set where every other is.
    let rdm = Rdm::new(find())
        .tool(look)
        .model(client(|_, name, _| if name == "look" { Say::Yes(true) } else { Say::P(0.6) }));
    let archive = json!({"a": "The hangar opens at dawn.", "b": "The hangar door sticks in frost."});
    assert_eq!(
        json!(rdm.call(inputs! { "archive" => &archive, "question" => "q" }).unwrap().observations),
        json!({"look": [true, true]})
    );
    rdm.set_params("answers", Params::threshold(0.7));
    assert_eq!(
        json!(rdm.call(inputs! { "archive" => &archive, "question" => "q" }).unwrap().observations),
        json!({"look": [false, false]})
    );
}

#[test]
fn a_tool_must_be_able_to_take_its_arguments() {
    // A name that is no answer and no input is refused: a misspelt name, or an argument written
    // for another module, never quietly receives something else.
    let model = || {
        client(|_, _, question| {
            if question.kind == rdm::systemone::Kind::Noul {
                Say::Yes(true)
            } else {
                Say::Level(0)
            }
        })
    };
    let careless = Rdm::new(triage())
        .tool(Tool::new("refund", "", |call| call.arg::<String>("tickt")))
        .model(model());
    let error = careless.call(inputs! { "ticket" => "t" }).unwrap_err();
    assert!(error.is_usage());
    assert!(
        error
            .to_string()
            .starts_with("Tool 'refund' takes ['tickt'], which name no answer or input of the request.")
    );
    assert!(error.to_string().contains("['escalate', 'frustration', 'refund', 'ticket']"));
    // A value read as a type it is not is the same kind of mistake.
    let mistyped = Rdm::new(triage())
        .tool(Tool::new("refund", "", |call| call.arg::<u64>("ticket")))
        .model(model());
    assert!(mistyped.call(inputs! { "ticket" => "t" }).unwrap_err().is_usage());
    // An answer is read as its plain value: a bool, a number.
    let plain = Tool::new("refund", "", |call| {
        Ok((
            call.arg::<bool>("refund")?,
            call.arg::<f64>("frustration")?,
            call.arg::<Option<String>>("ticket")?,
        ))
    });
    assert_eq!(
        Rdm::new(triage())
            .tool(plain)
            .model(model())
            .call(inputs! { "ticket" => "t" })
            .unwrap()
            .observations["refund"],
        json!([true, 0.0, "t"])
    );

    // A module used wrongly inside a tool is an error wherever it happens: at the root, or in a
    // sub-call two levels down, where any other failure would be an observation.
    let wrong = Tool::new("lock", "", |call| call.arg::<String>("door"));
    let split = Tool::new("split", "", |call| call.sub(inputs! { "request" => "lock the shed" }));
    let rdm = Rdm::new(home()).tool(split).tool(wrong).model(client(at_home));
    let error = rdm.call(inputs! { "request" => "lock the gate and lock the shed" }).unwrap_err();
    assert!(error.is_usage() && error.to_string().starts_with("Tool 'lock' takes ['door']"));
    // So is a sub-call with inputs its signature does not take.
    let misfit = Tool::new("split", "", |call| call.sub(inputs! { "query" => "lock the shed" }));
    let error = Rdm::new(home())
        .tool(misfit)
        .model(client(at_home))
        .call(inputs! { "request" => "a and b" })
        .unwrap_err();
    assert!(
        error.is_usage()
            && error
                .to_string()
                .contains("The call has inputs ['query']; its signature takes ['request']")
    );
}

#[test]
fn a_model_that_fails_is_an_error_wherever_it_is_asked() {
    let failing = Scripted::fallible(|request| {
        if request.state["inputs"]["request"] == "lock the shed" {
            return Err(rdm::systemone::Error::Transport("the line dropped".into()));
        }
        Ok(request
            .questions
            .keys()
            .map(|name| (name.clone(), Answer::noul(if name == "split" { 0.9 } else { 0.1 })))
            .collect())
    });
    let split = Tool::new("split", "", |call| call.sub(inputs! { "request" => "lock the shed" }));
    let error = Rdm::new(home())
        .tool(split)
        .model(failing)
        .call(inputs! { "request" => "a and b" })
        .unwrap_err();
    // Not the observation of the tool that asked: the run cannot go on without its model.
    assert_eq!(error, Error::Model(rdm::systemone::Error::Transport("the line dropped".into())));
    // An answer that is not evidence for the question is the model's failure too.
    let wrong = Scripted::new(|_| {
        [
            ("split", Answer::noul(1.5)),
            ("lock", Answer::noul(0.5)),
            ("chat", Answer::noul(0.5)),
        ]
    });
    assert!(matches!(
        Rdm::new(home()).model(wrong).call(inputs! { "request" => "r" }),
        Err(Error::Model(rdm::systemone::Error::Evidence(_)))
    ));
}

#[test]
fn a_tool_that_takes_the_call_reads_the_run_state_and_the_module() {
    let seen = Shared::new(Vec::new());
    let noted = seen.clone();
    let look = Tool::new("look", "", move |call| {
        // The module is on the call, so a tool needs no reference of its own to make a sub-call.
        let hits: Vec<String> = serde_json::from_value(call.state()["index"][call.state()["run_id"].as_str().unwrap()].clone())?;
        // The call says where the tool is: the depth, the turn, what the request showed, the history so far, and the requests left.
        noted.lock().push(json!([
            call.state(),
            call.depth(),
            call.iteration(),
            call.shown(),
            call.trajectory().len(),
            call.requests_left(),
            call.tool(),
            call.index()
        ]));
        assert_eq!(
            call.rdm().signature().instructions(),
            "Find out whether an archive is worth searching for a question."
        );
        let question: String = call.arg("question")?;
        let mut verdicts = Vec::new();
        for record in hits {
            verdicts.push(
                call.sub_with(Ask::new(inputs! { "record" => record, "question" => &question }).signature(judge()))?
                    .yes("answers"),
            );
        }
        assert_eq!(call.requests_left(), 254);
        Ok(verdicts)
    });
    let model = client(|_, _, _| true);
    let rdm = Rdm::new(find()).tool(look).model(model.clone());
    let state = json!({"index": {"run 7": ["The hangar opens at dawn."]}, "run_id": "run 7", "question": "not an input"});
    let result = rdm
        .call_with(Ask::new(inputs! { "archive" => json!({"a": "x"}), "question" => "q" }).state(state.clone()))
        .unwrap();
    // What was passed as state reached the tool and no request: it is state the model never sees.
    // Its names are its own, so one may be the name of an input.
    assert_eq!(
        seen.get(),
        [json!([state, 0, 0, {"archive": {"a": "x"}, "question": "q"}, 1, 255, "look", 0])]
    );
    assert_eq!(json!(result.observations), json!({"look": [true]}));
    let sent = json!(model.requests()).to_string();
    assert!(!sent.contains("run 7") && !sent.contains("index") && !sent.contains("not an input"));
    assert_eq!(result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>(), [0, 1]);
    // With none given, the state is an empty object.
    let empty = Tool::new("look", "", |call| Ok(call.state().clone()));
    assert_eq!(
        Rdm::new(find())
            .tool(empty)
            .model(model)
            .call(inputs! { "archive" => json!({}), "question" => "q" })
            .unwrap()
            .observations["look"],
        json!({})
    );
}

#[test]
fn state_given_to_a_module_inside_a_tool_is_its_own_and_the_callers_is_as_it_was() {
    // A module that is given state works the same when it is called inside another module's tool:
    // the state is its own and that of the calls under it, and the caller's is as it was.
    let (inner_seen, outer_seen) = (Shared::new(Vec::new()), Shared::new(Vec::new()));
    let (inner_noted, outer_noted) = (inner_seen.clone(), outer_seen.clone());
    let lookup = Tool::new("look", "", move |call| {
        inner_noted.lock().push(call.state().clone());
        let record = call.state()["shelf"].clone();
        let verdict = call.sub_with(
            Ask::new(inputs! { "record" => record, "question" => call.input("question")? })
                .signature(judge())
                .max_depth(0),
        )?;
        Ok(verdict.yes("answers"))
    });
    let model = client(|_, _, _| true);
    let inner = Rdm::new(find()).tool(lookup).model(model.clone());
    let within = inner.clone();
    let outer_look = Tool::new("look", "", move |call| {
        outer_noted.lock().push(call.state().clone());
        let ask = Ask::new(inputs! { "archive" => call.input("archive")?, "question" => call.input("question")? })
            .state(json!({"shelf": "The hangar opens at dawn."}));
        let found = call.on(&within).sub_with(ask)?.observations["look"].clone();
        outer_noted.lock().push(call.state().clone());
        Ok(found)
    });
    let outer = Rdm::new(find()).tool(outer_look).model(model);
    let inputs = || inputs! { "archive" => json!({}), "question" => "q" };
    assert_eq!(
        json!(
            inner
                .call_with(Ask::new(inputs()).state(json!({"shelf": "alone"})))
                .unwrap()
                .observations
        ),
        json!({"look": true})
    );
    let result = outer.call_with(Ask::new(inputs()).state(json!({"who": "outer"}))).unwrap();
    assert_eq!(json!(result.observations), json!({"look": true}));
    assert_eq!(
        inner_seen.get(),
        [json!({"shelf": "alone"}), json!({"shelf": "The hangar opens at dawn."})]
    );
    assert_eq!(outer_seen.get(), [json!({"who": "outer"}), json!({"who": "outer"})]);
    // The other module's requests are in the same trajectory, one level down each.
    assert_eq!(result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>(), [0, 1, 2]);
}

#[test]
fn a_tool_records_what_it_changed_on_the_request_it_ran_for() {
    let keep = Tool::new("refund", "", |call| {
        call.changed(json!({"by": "test", "did": "add", "name": "x"}));
        call.changed(json!({"by": "test", "did": "remove", "name": "x"}));
        Ok(())
    });
    let model = client(|_, name, _| {
        if name == "frustration" {
            Say::Level(0)
        } else {
            Say::Yes(name == "refund")
        }
    });
    let result = Rdm::new(triage())
        .tool(keep)
        .model(model)
        .call(inputs! { "ticket" => "t" })
        .unwrap();
    assert_eq!(
        json!(result.trajectory[0])["changes"],
        json!([{"by": "test", "did": "add", "name": "x"}, {"by": "test", "did": "remove", "name": "x"}])
    );
    assert_eq!(
        json!(result.trajectory[0]).as_object().unwrap().keys().next_back().unwrap(),
        "changes"
    );
}

#[test]
fn a_live_input_passed_on_to_a_sub_call_is_still_live() {
    let act = Signature::new("Operate a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul("split", "Does `inputs.task` ask for more than one thing?")
        .noul("switch", "Does `inputs.task` ask for the TV to be put on?");
    let house = Shared::new(json!({"tv": "off"}));
    let changed = house.clone();
    let switch = Tool::new("switch", "", move |call| {
        // The argument is what the request showed; the house itself is what the tool keeps.
        let shown: Json = call.arg("house")?;
        changed.update(|house| house["tv"] = json!("on"));
        Ok(shown["tv"].clone())
    });
    // Each part is a request of its own, about the same house: the input is passed on as it was given.
    let split = Tool::new("split", "", |call| {
        let task: String = call.arg("task")?;
        let parts = task
            .split(" and ")
            .map(|part| call.sub(inputs! { "task" => part, "house" => call.input("house")? }));
        Ok(parts
            .collect::<rdm::Result<Vec<_>>>()?
            .iter()
            .map(|part| part.observations["switch"].clone())
            .collect::<Vec<_>>())
    });
    let model = client(|inputs, name, _| (name == "split") == inputs["task"].as_str().unwrap().contains(" and "));
    let rdm = Rdm::new(act).tool(split).tool(switch).model(model.clone());
    let result = rdm.call(inputs! { "task" => "tv on and tv on again", "house" => &house }).unwrap();
    // The second part was shown the house as the first part's tool left it, and the trajectory keeps what each showed.
    assert_eq!(
        shown(&model).iter().map(|inputs| inputs["house"]["tv"].clone()).collect::<Vec<_>>(),
        ["off", "off", "on"]
    );
    assert_eq!(
        result
            .trajectory
            .iter()
            .map(|entry| entry.inputs["house"]["tv"].clone())
            .collect::<Vec<_>>(),
        ["off", "off", "on"]
    );
    assert_eq!(result.observations["split"], json!(["off", "on"]));
    assert!(result.trajectory.iter().all(|entry| !entry.repeat));
}

#[test]
fn a_tool_recurses_into_the_module_that_is_running_not_the_one_it_was_written_for() {
    // The model is never sure a layer is solid: it says 0.6 where it should say no.
    let dig = Signature::new("Dig until the layer is solid.")
        .input_typed("layer", "int")
        .input_typed("solid_at", "int")
        .noul("dig", "Is the layer still loose?");
    let deeper = Tool::new("dig", "", |call| {
        let (layer, solid_at): (i64, i64) = (call.arg("layer")?, call.arg("solid_at")?);
        call.sub(inputs! { "layer" => layer + 1, "solid_at" => solid_at })
    });
    let model = client(|inputs, _, _| {
        if inputs["layer"].as_i64() < inputs["solid_at"].as_i64() {
            0.9
        } else {
            0.6
        }
    });
    let rdm = Rdm::new(dig).tool(deeper).max_depth(8).model(model);
    // A copy with a threshold of its own: the same tool, and its sub-calls are the copy's, so
    // the copy's threshold applies at every level below. The module it was copied from is as it was.
    let tuned = rdm.copy().params("dig", Params::threshold(0.75));
    let dug = |module: &Rdm| module.call(inputs! { "layer" => 0, "solid_at" => 3 }).unwrap().trajectory.len();
    assert_eq!((dug(&rdm), dug(&tuned)), (9, 4));
    assert_eq!(
        (rdm.params_of("dig"), tuned.params_of("dig")),
        (None, Some(Params::threshold(0.75)))
    );
}
