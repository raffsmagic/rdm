//! A run written down as it happens, and the same run again from what was written.

mod common;

use std::sync::{Arc, Mutex};

use common::{Say, client, scripted};
use rdm::record::{Event, replay};
use rdm::systemone::{self, Answer, DecisionModel, Params, Scripted};
use rdm::{Ask, Error, Json, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;

/// A record that keeps every event, in the order it was handed them.
#[derive(Clone, Default)]
struct Kept(Arc<Mutex<Vec<Event>>>);

impl Kept {
    fn record(&self) -> impl Fn(&Event) -> rdm::Result<()> + Send + Sync + 'static {
        let kept = self.0.clone();
        move |event| {
            kept.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn events(&self) -> Vec<Event> {
        self.0.lock().unwrap().clone()
    }

    fn names(&self) -> Vec<&'static str> {
        self.events().iter().map(Event::name).collect()
    }

    /// The events as a host writes them: a line of JSON each, and read back.
    fn as_written(&self) -> Vec<Event> {
        let lines: Vec<String> = self.events().iter().map(|event| serde_json::to_string(event).unwrap()).collect();
        lines.iter().map(|line| serde_json::from_str(line).unwrap()).collect()
    }
}

const ZEN: &str =
    "Beautiful is better than ugly.\nSimple is better than complex.\nFlat is better than nested.\nSparse is better than dense.";

/// The README's search: a yes about a text asks the same question of each half, one level down.
fn search(model: Arc<Scripted>) -> Rdm {
    let signature = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");
    let look = Tool::new("look", "Look in each half of the text.", |call| {
        let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() < 2 {
            return Ok(vec![text]);
        }
        let mut found = Vec::new();
        for half in lines.chunks(lines.len().div_ceil(2)) {
            let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
        }
        Ok(found)
    });
    Rdm::new(signature).tool(look).max_depth(6).model(model)
}

fn nested() -> Arc<Scripted> {
    client(|inputs, _, _| inputs["text"].as_str().unwrap_or_default().contains("nested"))
}

#[test]
fn a_recursive_run_is_handed_over_event_by_event_each_at_its_place() {
    let kept = Kept::default();
    let rdm = search(nested()).record(kept.record());
    let result = rdm
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();
    assert_eq!(result.trajectory.len(), 5);

    let events = kept.events();
    let run = events[0].run();
    assert!(events.iter().all(|event| event.run() == run));
    assert_eq!(events.first().unwrap().name(), "began");
    assert_eq!(
        *events.last().unwrap(),
        Event::Ended {
            run,
            reason: Some(rdm::Reason::Final),
            error: None
        }
    );

    // Every request is opened once and asked once, and is opened just before it is asked.
    let opened: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            Event::Opened { index, inputs, .. } => {
                // An `asked` follows each, and shows what it showed: the inputs are not written twice.
                assert!(inputs.is_none());
                Some(*index)
            }
            _ => None,
        })
        .collect();
    assert_eq!(opened, [0, 1, 2, 3, 4]);
    for (at, event) in events.iter().enumerate() {
        if let Event::Opened { index, .. } = event {
            assert!(matches!(&events[at + 1], Event::Asked { index: asked, .. } if asked == index));
        }
    }

    // A tool's intent comes before the sub-calls it makes, and its outcome after them.
    let place = |wanted: &dyn Fn(&Event) -> bool| events.iter().position(wanted).unwrap();
    let started = place(&|event| matches!(event, Event::Started { index: 0, .. }));
    let returned = place(&|event| matches!(event, Event::Returned { index: 0, .. }));
    let below = place(&|event| {
        matches!(
            event,
            Event::Opened {
                index: 1,
                parent: Some(0),
                depth: 1,
                ..
            }
        )
    });
    assert!(started < below && below < returned);
    let Event::Returned {
        observation, output, tool, ..
    } = &events[returned]
    else {
        unreachable!()
    };
    assert_eq!((output.as_str(), tool.as_str()), ("look", "look"));
    assert_eq!(*observation, json!(["Flat is better than nested."]));
}

#[test]
fn deleting_everything_but_the_events_and_replaying_them_gives_the_same_run() {
    let kept = Kept::default();
    let live = search(nested()).record(kept.record());
    let first = live
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();

    // Another module, built anew as a process that started again would build it, answered only
    // from the events as they were written: the key of a request is its content, not the module.
    let events = kept.as_written();
    let replayed = Arc::new(replay("jev-test", &events));
    assert_eq!(replayed.len(), 5);
    let rebuilt = search(Arc::new(Scripted::new(|_| -> [(&str, Answer); 0] {
        panic!("nothing is asked again")
    })));
    rebuilt.set_model(replayed);
    let again = rebuilt
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();
    assert_eq!(again.trajectory, first.trajectory);
    assert_eq!(again.observations, first.observations);
}

#[test]
fn a_decision_is_made_again_from_the_same_evidence_when_a_threshold_changes() {
    let signature = Signature::new("Triage a ticket.")
        .input("ticket")
        .noul("refund", "Is `inputs.ticket` asking for a refund?");
    let refunded = Shared::new(0);
    let refund = {
        let refunded = refunded.clone();
        Tool::new("refund", "Refund the order.", move |_| {
            refunded.update(|count| *count += 1);
            Ok("refunded")
        })
    };
    let kept = Kept::default();
    let rdm = Rdm::new(signature).tool(refund).model(client(|_, _, _| 0.6)).record(kept.record());
    assert!(rdm.call(inputs! { "ticket" => "Charged twice?" }).unwrap().yes("refund"));

    // A gate raised: the same answer kept, decided against the new threshold, and nothing asked.
    rdm.set_model(Arc::new(replay("jev-test", &kept.events())));
    rdm.set_params("refund", Params::threshold(0.7));
    let again = rdm.call(inputs! { "ticket" => "Charged twice?" }).unwrap();
    assert!(!again.yes("refund"));
    assert_eq!(again.probability("refund"), Some(0.6));
    assert_eq!(refunded.get(), 1);
    // And the numbers it was decided with are in the event, so the decision follows from the log.
    let params: Vec<_> = kept
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::Asked { params, .. } => Some(params),
            _ => None,
        })
        .collect();
    assert!(params[0].is_empty());
    assert_eq!(params.last().unwrap()["refund"], Params::threshold(0.7));
}

#[test]
fn a_replay_that_would_have_to_ask_something_new_says_so_and_asks_nothing() {
    let kept = Kept::default();
    search(nested())
        .record(kept.record())
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();
    let rdm = search(nested());
    rdm.set_model(Arc::new(replay("jev-test", &kept.events())));
    let error = rdm
        .call(inputs! { "text" => ZEN, "question" => "Is sparse better than dense?" })
        .unwrap_err();
    assert!(matches!(error, Error::Model(systemone::Error::Unrecorded(_))), "{error:?}");
}

#[test]
fn a_record_that_refuses_an_intent_keeps_the_tool_from_running() {
    let signature = Signature::new("Triage a ticket.")
        .input("ticket")
        .noul("refund", "Is `inputs.ticket` asking for a refund?");
    let refunded = Shared::new(false);
    let refund = {
        let refunded = refunded.clone();
        Tool::new("refund", "Refund the order.", move |_| {
            refunded.set(true);
            Ok("refunded")
        })
    };
    let names = Arc::new(Mutex::new(Vec::new()));
    let seen = names.clone();
    let rdm = Rdm::new(signature)
        .tool(refund)
        .model(client(|_, _, _| true))
        .record(move |event: &Event| {
            seen.lock().unwrap().push(event.name());
            match event {
                Event::Started { .. } => Err(Error::usage("the trace could not be written")),
                _ => Ok(()),
            }
        });
    let error = rdm.call(inputs! { "ticket" => "Charged twice." }).unwrap_err();
    assert_eq!(error, Error::usage("the trace could not be written"));
    assert!(!refunded.get());
    // A record that refused is handed nothing after it.
    assert_eq!(*names.lock().unwrap(), ["began", "opened", "asked", "started"]);
}

#[test]
fn a_model_that_failed_is_an_event_and_a_replay_fails_the_same_way() {
    let signature = Signature::new("Triage a ticket.")
        .input("ticket")
        .noul("refund", "Is `inputs.ticket` asking for a refund?");
    let down = Scripted::fallible(|_| Err(systemone::Error::Transport("connection refused".into()))).named("jev");
    let kept = Kept::default();
    let rdm = Rdm::new(signature).model(down).record(kept.record());
    let error = rdm.call(inputs! { "ticket" => "Charged twice." }).unwrap_err();
    assert_eq!(error, Error::Model(systemone::Error::Transport("connection refused".into())));

    let events = kept.as_written();
    assert!(matches!(
        &events[2],
        Event::Asked {
            response: None,
            error: Some(systemone::Error::Transport(_)),
            ..
        }
    ));
    assert!(matches!(&events[3], Event::Ended { reason: None, error: Some(said), .. } if said == "connection refused"));
    // A failure kept is given again, and is not a request never asked.
    rdm.set_model(Arc::new(replay("jev", &events)));
    assert_eq!(rdm.call(inputs! { "ticket" => "Charged twice." }).unwrap_err(), error);
}

#[test]
fn a_run_is_known_by_the_tag_its_call_was_given_and_says_what_module_began_it() {
    let kept = Kept::default();
    let rdm = search(nested()).record(kept.record());
    let tag = json!({"utterance": 41, "channel": "text"});
    rdm.call_with(Ask::new(inputs! { "text" => "One line.", "question" => "Is it?" }).tag(tag.clone()))
        .unwrap();
    let Event::Began { tag: began, module, .. } = &kept.events()[0] else {
        panic!("a run begins with `began`")
    };
    assert_eq!(*began, tag);
    assert_eq!(module.instructions, "Search a text for the answer to a question.");
    assert_eq!(module.tools, [("look".to_owned(), "Look in each half of the text.".to_owned())]);
    assert_eq!((module.max_depth, module.finish), (6, true));

    // Runs at once are told apart by their number.
    let asks = ["a", "b", "c"].map(|text| Ask::new(inputs! { "text" => text, "question" => "?" }));
    kept.0.lock().unwrap().clear();
    rdm.batch(asks).into_iter().for_each(|result| drop(result.unwrap()));
    let mut runs: Vec<u64> = kept.events().iter().map(Event::run).collect();
    runs.dedup();
    runs.sort();
    runs.dedup();
    assert_eq!(runs.len(), 3);
}

#[test]
fn a_pointer_loop_and_the_map_it_changes_are_events_too() {
    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");
    let keep = Tool::new("keep", "Keep a routine.", |call| {
        call.changed(json!({"added": "movie time"}));
        Ok("kept")
    });
    let kept = Kept::default();
    let rdm = Rdm::new(operate)
        .tool(keep)
        .model(scripted(["keep", "finish"]))
        .record(kept.record());
    rdm.call(inputs! { "task" => "Keep movie time." }).unwrap();
    assert_eq!(
        kept.names(),
        [
            "began", "opened", "asked", "started", "changed", "returned", "opened", "asked", "ended"
        ]
    );
    let events = kept.events();
    assert!(matches!(&events[4], Event::Changed { index: 0, change, .. } if *change == json!({"added": "movie time"})));
    assert!(matches!(&events[6], Event::Opened { index: 1, parent: Some(0), iteration: 1, tool: Some(tool), .. } if tool == "keep"));
}

#[test]
fn a_request_not_asked_shows_its_inputs_in_its_own_event() {
    // The second turn is the first again, and is answered from it: `repeat`, and nothing asked.
    let operate = Signature::new("Wait for a door.").input("door").tool("next", "What next?");
    let wait = Tool::new("wait", "Wait.", |_| Ok(Json::Null));
    let kept = Kept::default();
    let rdm = Rdm::new(operate)
        .tool(wait)
        .model(client(|_, _, _| Say::Pick("wait".into())))
        .record(kept.record());
    let result = rdm.call(inputs! { "door" => "front" }).unwrap();
    assert_eq!(result.termination_reason, "repeat");
    let events = kept.events();
    let repeat = events
        .iter()
        .find(|event| matches!(event, Event::Opened { repeat: true, .. }))
        .unwrap();
    assert!(matches!(repeat, Event::Opened { inputs: Some(inputs), .. } if *inputs == json!({"door": "front"})));
    assert_eq!(kept.names().iter().filter(|name| **name == "asked").count(), 1);
}

#[test]
fn what_a_fit_asks_is_written_down_as_requests_of_the_run() {
    let short = Signature::new("Say whether a text is long.")
        .input("text")
        .noul("long", "Is `inputs.text` long?");
    // A fit is given every request of the module, its own sub-calls too: it asks only about the search.
    let fit = move |signature: Signature, inputs: rdm::Inputs, call: &rdm::Call| {
        if signature.output("look").is_some() {
            let text: String = call.arg("text")?;
            call.sub_with(Ask::new(inputs! { "text" => text }).signature(short.clone()).answers_only())?;
        }
        Ok((signature, inputs))
    };
    let kept = Kept::default();
    let rdm = search(nested()).fit_fn(fit).record(kept.record());
    rdm.call(inputs! { "text" => "One line.", "question" => "Is it?" }).unwrap();
    let asked: Vec<(usize, Vec<String>)> = kept
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::Asked { index, request, .. } => Some((index, request.unwrap().questions.keys().cloned().collect())),
            _ => None,
        })
        .collect();
    assert_eq!(asked, [(1, vec!["long".to_owned()]), (0, vec!["look".to_owned()])]);
}

#[test]
fn a_module_with_no_record_writes_nothing_and_a_key_is_the_requests_content() {
    let model = nested();
    let first = search(model.clone());
    let second = search(model.clone());
    first.call(inputs! { "text" => "One line.", "question" => "Is it?" }).unwrap();
    second.call(inputs! { "text" => "One line.", "question" => "Is it?" }).unwrap();
    let requests = model.requests();
    assert_eq!(
        systemone::key(&requests[0], &model.name()),
        systemone::key(&requests[1], &model.name())
    );
}

#[test]
fn what_a_language_model_wrote_for_a_tool_is_kept_and_replayed_a_retry_included() {
    use rdm::lm::{Generate, Lm, ReplayLm, ScriptedLm};

    // The first reply cannot be used, and the model is asked again with what was wrong.
    let tries = Arc::new(Mutex::new(0));
    let writes = {
        let tries = tries.clone();
        move |_: &str, _: &str| {
            let mut tries = tries.lock().unwrap();
            *tries += 1;
            if *tries == 1 {
                "not JSON".to_owned()
            } else {
                r#"{"answer": "An owl.", "sure": true}"#.to_owned()
            }
        }
    };
    let run = |lm: Lm, kept: &Kept| {
        // Two outputs, so that a reply that is not JSON cannot be taken as the answer.
        let answer = Generate::parse("question -> answer, sure: bool").unwrap().lm(lm);
        let reply = Tool::new("reply", "Answer.", move |call| {
            answer
                .call_in(call, inputs! { "question" => call.arg::<String>("question")? })?
                .read::<String>("answer")
        });
        Rdm::new(
            Signature::new("Chat.")
                .input("question")
                .noul("reply", "Is `inputs.question` a question?"),
        )
        .tool(reply)
        .model(client(|_, _, _| true))
        .record(kept.record())
        .call(inputs! { "question" => "What eats mice at night?" })
        .unwrap()
    };
    let kept = Kept::default();
    let live = run(Lm::of(ScriptedLm::new(writes), "scripted"), &kept);
    assert_eq!(live.observations["reply"], "An owl.");
    assert_eq!(
        kept.names(),
        ["began", "opened", "asked", "started", "wrote", "wrote", "returned", "ended"]
    );
    assert!(matches!(&kept.events()[4], Event::Wrote { tool: Some(tool), index: 0, .. } if tool == "reply"));

    // Both replies come back from what was written, in the order they were asked for.
    let again = Kept::default();
    let replayed = run(Lm::of(ReplayLm::new(&kept.as_written()), "scripted"), &again);
    assert_eq!(replayed.observations, live.observations);
    assert_eq!(*tries.lock().unwrap(), 2);
    assert_eq!(again.names(), kept.names());
}

#[test]
fn a_run_kept_without_what_it_was_shown_still_reads_back_and_replays() {
    let kept = Kept::default();
    let live = search(nested()).record(kept.record());
    let first = live
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();

    // The host keeps the evidence and withholds what the requests showed and what the tools returned.
    let withheld: Vec<Event> = kept
        .events()
        .iter()
        .map(|event| {
            let mut written = serde_json::to_value(event).unwrap();
            for field in ["request", "inputs", "observation"] {
                written.as_object_mut().unwrap().remove(field);
            }
            written["withheld"] = json!(["request", "inputs", "observation"]);
            serde_json::from_value(written).unwrap()
        })
        .collect();
    assert!(withheld.iter().all(|event| !matches!(event, Event::Asked { request: Some(_), .. })));

    let rebuilt = search(nested());
    rebuilt.set_model(Arc::new(replay("jev-test", &withheld)));
    let again = rebuilt
        .call(inputs! { "text" => ZEN, "question" => "Is flat better than nested?" })
        .unwrap();
    assert_eq!(again.trajectory, first.trajectory);
}
