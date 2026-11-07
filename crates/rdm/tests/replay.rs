//! Code a REPL runs for a run, written down and replayed: what the world answered it, what it
//! drew at random, what a language model wrote for it, and how each piece of it ended.
#![cfg(feature = "rlm")]

mod common;

use std::sync::{Arc, Mutex};

use common::{client, scripted};
use rdm::lm::{Lm, ReplayLm, ScriptedLm, Type, input_of};
use rdm::record::{Ending, Event, replay};
use rdm::repl::Repl;
use rdm::rlm::Rlm;
use rdm::systemone;
use rdm::{Error, Json, Rdm, Shared, Signature, inputs};
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

    /// The events as a host writes them, a line of JSON each, and reads them back.
    fn events(&self) -> Vec<Event> {
        let events = self.0.lock().unwrap();
        let lines: Vec<String> = events.iter().map(|event| serde_json::to_string(event).unwrap()).collect();
        lines.iter().map(|line| serde_json::from_str(line).unwrap()).collect()
    }

    fn names(&self) -> Vec<&'static str> {
        self.0.lock().unwrap().iter().map(Event::name).collect()
    }
}

/// Events with their run's number taken out: two runs are set side by side by what happened.
fn unnumbered(events: &[Event]) -> Vec<Json> {
    let each = events.iter().map(|event| {
        let mut written = serde_json::to_value(event).unwrap();
        written.as_object_mut().unwrap().remove("run");
        written
    });
    each.collect()
}

/// A heater, and a REPL whose code can read it, cool it and draw at random. A function that
/// is replayed is never called: it is answered from the events.
fn heater(world: &Shared<i64>) -> Repl {
    let (cooled, read) = (world.clone(), world.clone());
    Repl::new()
        .function("cooler", "Make it one degree cooler.", move |_| {
            Ok(cooled.update(|heat| {
                *heat -= 1;
                format!("heat is {heat}")
            }))
        })
        .function("heat", "How warm it is, in degrees.", move |_| Ok(read.get()))
}

const COOL: &str =
    "def cool_to(degrees):\n    import random\n    while heat() > degrees:\n        cooler()\n    return [heat(), random.randint(1, 100)]";

/// A module whose map has the written function on it, and a model that points at it once.
fn operating(repl: &Repl, model: impl systemone::DecisionModel + 'static) -> Rdm {
    let operate = Signature::new("Operate a heater.")
        .input("task")
        .input_typed("degrees", "int")
        .tool("next", "What should happen next for `inputs.task`?");
    Rdm::new(operate).source(repl.clone()).model(model)
}

#[test]
fn a_written_function_is_replayed_from_what_the_world_answered_it_and_nothing_is_done_again() {
    let world = Shared::new(21);
    let repl = heater(&world);
    repl.define("cool_to", "Cool the heater to some degrees.", COOL).unwrap();
    let kept = Kept::default();
    let live = operating(&repl, scripted(["cool_to", "finish"]))
        .record(kept.record())
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap();
    assert_eq!(world.get(), 18);
    let names = kept.names();
    // heat() five times and cooler() three, the randomness `random` seeded itself with, and the code as it ran.
    assert_eq!(names.iter().filter(|name| **name == "answered").count(), 8);
    assert_eq!(names.iter().filter(|name| **name == "drew").count(), 1);
    let ran: Vec<Event> = kept.events().into_iter().filter(|event| event.name() == "ran").collect();
    assert!(
        matches!(&ran[..], [Event::Ran { repl, ending: Ending::Ran, .. }] if repl == "repl"),
        "{ran:?}"
    );

    // The heater is somewhere else now. The run is replayed from its events alone.
    world.set(30);
    let events = kept.events();
    let again_kept = Kept::default();
    let again = operating(&repl.replaying(&events), replay("jev-test", &events))
        .record(again_kept.record())
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap();
    assert_eq!(again.trajectory, live.trajectory);
    assert_eq!(again.observations, live.observations);
    assert_eq!(world.get(), 30, "nothing was cooled");
    // And it is written down the same, event for event.
    assert_eq!(unnumbered(&again_kept.events()), unnumbered(&events));
}

#[test]
fn a_replay_that_goes_another_way_is_ended_and_says_so() {
    let world = Shared::new(21);
    let repl = heater(&world);
    repl.define("cool_to", "Cool the heater to some degrees.", COOL).unwrap();
    let kept = Kept::default();
    operating(&repl, scripted(["cool_to", "finish"]))
        .record(kept.record())
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap();
    let events = kept.events();

    // Cooler by one more degree: heat() is asked a fifth time, which it never was.
    let error = operating(&repl.replaying(&events), replay("jev-test", &events))
        .call(inputs! { "task" => "Cool it.", "degrees" => 17 })
        .unwrap_err();
    assert!(matches!(&error, Error::Model(systemone::Error::Unrecorded(_))), "{error:?}");
    assert_eq!(world.get(), 18);

    // A replay without what was drawn: `random` asks, and is refused.
    let undrawn: Vec<Event> = events.iter().filter(|event| event.name() != "drew").cloned().collect();
    let error = operating(&repl.replaying(&undrawn), replay("jev-test", &undrawn))
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap_err();
    assert!(error.to_string().contains("bytes of randomness"), "{error}");
}

#[test]
fn a_record_that_refuses_what_a_function_answered_ends_the_run() {
    let world = Shared::new(21);
    let repl = heater(&world);
    repl.define("cool_to", "Cool the heater to some degrees.", COOL).unwrap();
    let refusing = |event: &Event| match event {
        Event::Answered { function, .. } if function == "cooler" => Err(Error::usage("the log is full")),
        _ => Ok(()),
    };
    let error = operating(&repl, scripted(["cool_to", "finish"]))
        .record(refusing)
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap_err();
    assert_eq!(error.to_string(), "the log is full");
    // The one cooler() that ran is the one whose answer could not be kept: the code went no further.
    assert_eq!(world.get(), 20);
}

/// A language model that writes these cells, one a step. Asked for anything that is no step,
/// a question the code put to it with `llm_query`, it says how long the question is.
fn writes<const N: usize>(cells: [&str; N]) -> Lm {
    let cells: Vec<String> = cells.iter().map(|cell| (*cell).to_owned()).collect();
    let scripted = ScriptedLm::new(move |_system, prompt| {
        let Some(step) = input_of(prompt, "iteration") else {
            return format!("{} characters", prompt.len());
        };
        let at: usize = step.split('/').next().and_then(|at| at.parse().ok()).unwrap_or(1);
        let cell = cells.get(at - 1).or(cells.last()).cloned().unwrap_or_default();
        json!({"reasoning": format!("step {at}"), "code": cell}).to_string()
    });
    Lm::of(scripted, "scripted")
}

#[test]
fn an_rlm_that_is_a_tool_is_replayed_steps_queries_answers_and_draws_alike() {
    let world = Shared::new(21);
    let cells = [
        "import random\nprint(heat(), random.random())",
        "said = llm_query('How warm is ' + str(heat()) + '?')\nmany = llm_query_batched(['a', 'bb'])\nprint(said, many)\ncooler()",
        "SUBMIT(heat())",
    ];
    let rlm = |lm: Lm, env: &Repl| {
        let check = Rlm::new("Read the heater.")
            .input("task")
            .output_typed("heat", Type::Int)
            .lm(lm)
            .repl(env);
        let route = Signature::new("Route a request.")
            .input("task")
            .noul("check", "Does `inputs.task` need the heater read?");
        Rdm::new(route).tool(check.tool("check", "Read the heater."))
    };
    let kept = Kept::default();
    let live = rlm(writes(cells), &heater(&world))
        .model(client(|_, _, _| true))
        .record(kept.record())
        .call(inputs! { "task" => "How warm is it?" })
        .unwrap();
    assert_eq!(live.observations["check"], json!({"heat": 20}));
    let names = kept.names();
    // Three steps and three queries are what the language models wrote; three cells ran.
    assert_eq!(names.iter().filter(|name| **name == "wrote").count(), 6);
    assert_eq!(names.iter().filter(|name| **name == "ran").count(), 3);
    assert_eq!(names.iter().filter(|name| **name == "drew").count(), 1);

    // Everything but the events is gone: the language model, the heater, the decision model.
    world.set(40);
    let events = kept.events();
    let again_kept = Kept::default();
    let replayed = Lm::of(ReplayLm::new(&events), "scripted");
    let again = rlm(replayed, &heater(&world).replaying(&events))
        .model(replay("jev-test", &events))
        .record(again_kept.record())
        .call(inputs! { "task" => "How warm is it?" })
        .unwrap();
    assert_eq!(again.observations, live.observations);
    assert_eq!(again.trajectory, live.trajectory);
    assert_eq!(world.get(), 40);
    let (replayed, recorded) = (unnumbered(&again_kept.events()), unnumbered(&events));
    for (at, (replayed, recorded)) in replayed.iter().zip(&recorded).enumerate() {
        assert_eq!(replayed, recorded, "event {at}");
    }
    assert_eq!(replayed.len(), recorded.len());

    // A replay whose language model has not seen a query the code asks ends there.
    let unasked: Vec<Event> = events
        .iter()
        .filter(|event| !matches!(event, Event::Wrote { request: Some(request), .. } if request.to_string().contains("bb")))
        .cloned()
        .collect();
    let error = rlm(Lm::of(ReplayLm::new(&unasked), "scripted"), &heater(&world).replaying(&events))
        .model(replay("jev-test", &events))
        .call(inputs! { "task" => "How warm is it?" })
        .unwrap_err();
    assert!(matches!(&error, Error::Model(systemone::Error::Unrecorded(_))), "{error:?}");
}

#[test]
fn a_host_that_kept_no_code_and_no_arguments_still_replays_and_one_that_kept_no_answers_cannot() {
    let world = Shared::new(21);
    let repl = heater(&world);
    repl.define("cool_to", "Cool the heater to some degrees.", COOL).unwrap();
    let kept = Kept::default();
    let live = operating(&repl, scripted(["cool_to", "finish"]))
        .record(kept.record())
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap();
    let withheld = |keep_values: bool| -> Vec<Event> {
        let each = kept.events().into_iter().map(|mut event| {
            match &mut event {
                Event::Ran { code, printed, .. } => (*code, *printed) = (None, None),
                Event::Answered { args, value, .. } => {
                    *args = None;
                    if !keep_values {
                        *value = None;
                    }
                }
                _ => {}
            }
            event
        });
        // As a host writes them and reads them back.
        let lines: Vec<String> = each.map(|event| serde_json::to_string(&event).unwrap()).collect();
        lines.iter().map(|line| serde_json::from_str(line).unwrap()).collect()
    };

    // Without the code and the arguments, each answer is given to its function's calls in order.
    let events = withheld(true);
    let again = operating(&repl.replaying(&events), replay("jev-test", &events))
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap();
    assert_eq!(again.trajectory, live.trajectory);
    // Without the answers, there is nothing to answer with, and the run says so.
    let events = withheld(false);
    let error = operating(&repl.replaying(&events), replay("jev-test", &events))
        .call(inputs! { "task" => "Cool it.", "degrees" => 18 })
        .unwrap_err();
    assert!(error.to_string().contains("was not kept"), "{error}");
}

#[test]
fn a_function_that_answered_nothing_reads_back_as_an_answer_and_not_as_one_withheld() {
    let answered = |value: Option<Json>| Event::Answered {
        run: 1,
        index: 0,
        repl: "repl".to_owned(),
        function: "lights_off".to_owned(),
        args: Some(json!({})),
        value,
        error: None,
    };
    let read = |event: &Event| -> Event { serde_json::from_str(&serde_json::to_string(event).unwrap()).unwrap() };
    assert_eq!(read(&answered(Some(Json::Null))), answered(Some(Json::Null)));
    assert_eq!(read(&answered(None)), answered(None));
}

#[test]
fn the_time_written_code_read_is_read_again_from_the_events() {
    // The system's clock, which will have moved by the time of the replay.
    let repl = heater(&Shared::new(21));
    let stamp = "def stamp(degrees):\n    import time, datetime\n    return [time.time(), str(datetime.datetime.now()), str(datetime.date.today())]";
    repl.define("stamp", "Say when it is.", stamp).unwrap();
    let kept = Kept::default();
    let live = operating(&repl, scripted(["stamp", "finish"]))
        .record(kept.record())
        .call(inputs! { "task" => "When is it?", "degrees" => 18 })
        .unwrap();
    // One reading for each clock the code read.
    assert_eq!(kept.names().iter().filter(|name| **name == "clocked").count(), 3);

    let events = kept.events();
    let again = operating(&repl.replaying(&events), replay("jev-test", &events))
        .call(inputs! { "task" => "When is it?", "degrees" => 18 })
        .unwrap();
    assert_eq!(again.trajectory, live.trajectory);

    // Without the readings, the clock is not read again: the run says so.
    let unread: Vec<Event> = events.iter().filter(|event| event.name() != "clocked").cloned().collect();
    let error = operating(&repl.replaying(&unread), replay("jev-test", &unread))
        .call(inputs! { "task" => "When is it?", "degrees" => 18 })
        .unwrap_err();
    assert!(
        matches!(&error, Error::Model(systemone::Error::Unrecorded(why)) if why.contains("clock")),
        "{error:?}"
    );
}
