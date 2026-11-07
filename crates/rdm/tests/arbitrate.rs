//! Arbitration: the tools a request selected are settled as they come back, and the call is
//! over once they are settled, not once every one of them has come back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rdm::arbitrate::{ARBITRATE, Arbiter, Ledger, Ruling};
use rdm::record::Event;
use rdm::{Arbitrate, Call, Rdm, Signature, Tool, inputs};
use serde_json::json;
use systemone::{Answer, Scripted};

/// How long a test waits for what it expects before it fails: never what makes it pass.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
const PATIENCE: Duration = Duration::from_secs(20);

fn ask() -> Signature {
    Signature::new("Answer a question.")
        .input("question")
        .noul("quick", "Answer it quickly?")
        .noul("slow", "Work it out slowly?")
        .noul("late", "Ask someone else?")
}

/// A decision model that says yes to the questions named, and no to every other yes or no it
/// is asked. `next` it answers as one holding those yeses would: wait where something running
/// is waited for, start a branch where its option is named, finish where a result is used, and
/// say what is not known otherwise.
fn says(yes: &'static [&'static str]) -> Arc<Scripted> {
    Arc::new(Scripted::new(move |request| {
        let is = |name: &str| yes.contains(&name);
        let asked: Vec<&String> = request
            .questions
            .keys()
            .filter(|name| name.as_str() != "next" && !name.starts_with("same_"))
            .collect();
        let mut answers: Vec<(String, Answer)> = asked
            .iter()
            .map(|name| ((*name).clone(), Answer::noul(if is(name) { 0.9 } else { 0.1 })))
            .collect();
        // `same_X` is answered with the earlier result named `same_X=Y`, or none.
        for (name, question) in request.questions.iter().filter(|(name, _)| name.starts_with("same_")) {
            let labels = question.labels();
            let named = yes.iter().find_map(|said| said.strip_prefix(&format!("{name}=")));
            let chosen = named.or_else(|| labels.last().map(String::as_str)).unwrap_or_default().to_owned();
            answers.push((name.clone(), Answer::chose(&chosen, labels.iter().map(String::as_str))));
        }
        if let Some(next) = request.questions.get("next") {
            let options = next.labels();
            let any = |prefix: &str| asked.iter().any(|name| name.starts_with(prefix) && is(name));
            let start = options.iter().find(|option| option.starts_with("start_") && is(option));
            let chosen = match (any("wait_") && options.iter().any(|option| option == "wait"), start) {
                (true, _) => "wait",
                (false, Some(start)) => start.as_str(),
                (false, None) if any("use_") => "finish",
                _ => "unknown",
            };
            answers.push(("next".to_owned(), Answer::chose(chosen, options.iter().map(String::as_str))));
        }
        answers
    }))
}

/// The events of a run, kept as they come.
type Events = Arc<Mutex<Vec<Event>>>;

/// Every event of a run, as it was handed to the record.
fn kept() -> (Events, impl Fn(&Event) -> rdm::Result<()> + Send + Sync + 'static) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let keep = events.clone();
    (events, move |event: &Event| {
        keep.lock().unwrap().push(event.clone());
        Ok(())
    })
}

/// A tool that waits to be let go, and takes no notice of being stopped.
fn blocked(name: &str, go: Receiver<()>) -> Tool {
    let go = Mutex::new(go);
    Tool::new(name, "Waits.", move |_| {
        go.lock().unwrap().recv().map_err(|_| "never let go")?;
        Ok("came back late")
    })
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_call_is_over_once_it_is_settled_while_a_branch_it_stopped_is_still_running() {
    let (go, wait) = channel::<()>();
    let wait = Mutex::new(wait);
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let (events, keep) = kept();
    let rdm = Rdm::new(ask())
        // Comes back only once `slow` has begun: stopped then, `slow` is stopped while it runs.
        .tool(Tool::new("quick", "Answers quickly.", move |_| {
            begins.lock().unwrap().recv().map_err(|_| "slow never began")?;
            Ok("the answer")
        }))
        .tool(Tool::new("slow", "Waits.", move |_| {
            begun.lock().unwrap().send(()).map_err(|_| "no one to tell")?;
            wait.lock().unwrap().recv().map_err(|_| "never let go")?;
            Ok("came back late")
        }))
        .arbiter(Arbitrate::new())
        .record(keep)
        .model(says(&["quick", "slow", "use_quick"]));
    // Called on a thread of its own: were it to wait for `slow`, it would never come back.
    let (sent, came) = channel();
    std::thread::spawn(move || sent.send(rdm.call(inputs! { "question" => "q" })).is_ok());
    let result = came
        .recv_timeout(PATIENCE)
        .expect("the call came back while `slow` was still blocked")
        .unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["quick"]);
    assert_eq!(arbitration.stopped, ["slow"]);
    assert_eq!(result.observations.get("slow"), None);
    assert_eq!(result.observations["quick"], "the answer");
    // Let go, `slow` comes back after the run has ended, and that is recorded too: a stop is
    // not a claim that what it did was not done.
    go.send(()).unwrap();
    let names = || events.lock().unwrap().iter().map(|event| event.name()).collect::<Vec<_>>();
    let deadline = std::time::Instant::now() + PATIENCE;
    while names().iter().filter(|name| **name == "returned").count() < 2 {
        assert!(std::time::Instant::now() < deadline, "`slow` never came back: {:?}", names());
        std::thread::yield_now();
    }
    let names = names();
    let at = |name: &str| names.iter().position(|kept| *kept == name).unwrap();
    assert!(at("stopped") < at("ended"));
    assert_eq!(names.last(), Some(&"returned"));
}

#[test]
fn a_quick_result_that_does_not_answer_is_not_used_and_the_slow_one_is_waited_for() {
    let (go, wait) = channel();
    let go = Mutex::new(go);
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Guesses.", move |_| {
            // `slow` comes back only once the guess is in.
            go.lock().unwrap().send(()).unwrap();
            Ok("a guess")
        }))
        .tool(blocked("slow", wait))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "slow", "use_slow", "wait_slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["slow"]);
    assert!(arbitration.answered);
    assert_eq!(result.observations["slow"], "came back late");
}

#[test]
fn a_branch_not_selected_is_started_by_a_ruling_and_reads_what_came_back_before_it() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Guesses.", |_| Ok("a guess")))
        .tool(Tool::new("late", "Asks someone else.", |call: &Call| {
            let siblings: Vec<_> = call
                .siblings()
                .into_iter()
                .filter_map(|branch| Some((branch.output, branch.result?)))
                .collect();
            Ok(json!({"started_by_arbitration": call.arbitrated(), "read": siblings}))
        }))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "start_late", "use_late"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.started, ["late"]);
    assert_eq!(arbitration.used, ["late"]);
    assert_eq!(arbitration.unused, ["quick"]);
    assert_eq!(
        result.observations["late"],
        json!({"started_by_arbitration": true, "read": [["quick", "a guess"]]})
    );
}

#[test]
fn a_first_ruling_holds_back_what_is_not_needed_yet_and_it_never_runs_if_it_is_not() {
    let ran = Arc::new(AtomicBool::new(false));
    let running = ran.clone();
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", |_| Ok("the answer")))
        .tool(Tool::new("slow", "Works it out at length, and costs.", move |_| {
            running.store(true, Ordering::Relaxed);
            Ok("worked out")
        }))
        .arbiter(Arbitrate::new().first("Should `{name}` start now, before anything has come back?"))
        .model(says(&["quick", "slow", "start_quick", "use_quick"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.held, ["slow"]);
    assert_eq!(arbitration.used, ["quick"]);
    assert_eq!(arbitration.rounds, 2);
    assert!(!ran.load(Ordering::Relaxed));
}

#[test]
fn independent_parts_are_each_answered_and_the_call_waits_for_both() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers one part.", |_| Ok("part one")))
        .tool(Tool::new("slow", "Answers the other part.", |_| Ok("part two")))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "slow", "use_quick", "use_slow", "wait_quick", "wait_slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["quick", "slow"]);
    assert!(arbitration.stopped.is_empty());
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_stopped_branch_begins_nothing_more_and_its_sub_call_fails_as_stopped() {
    let (told, heard) = channel();
    let told = Mutex::new(told);
    let model = says(&["quick", "slow", "use_quick"]);
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let rdm = Rdm::new(ask())
        // Comes back only once `slow` has begun, so `slow` is stopped while it runs.
        .tool(Tool::new("quick", "Answers quickly.", move |_| {
            begins.lock().unwrap().recv().map_err(|_| "slow never began")?;
            Ok("the answer")
        }))
        .tool(Tool::new("slow", "Works on.", move |call: &Call| {
            begun.lock().unwrap().send(()).map_err(|_| "no one to tell")?;
            while !call.stopped() {
                std::thread::yield_now();
            }
            let below = call.sub(inputs! { "question" => "more" });
            told.lock().unwrap().send(below.map(|_| ())).unwrap();
            Ok("not seen")
        }))
        .arbiter(Arbitrate::new())
        .model(model.clone());
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.arbitration.unwrap().stopped, ["slow"]);
    let below = heard.recv_timeout(PATIENCE).unwrap();
    assert!(below.unwrap_err().is_stopped());
    // The question and its one ruling: the stopped branch asked nothing.
    assert_eq!(model.requests().len(), 2);
}

#[test]
fn a_ruling_is_a_request_under_the_one_it_settles_answered_and_selecting_nothing() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", |_| Ok("the answer")))
        .tool(Tool::new("late", "Asks someone else.", |_| Ok("not asked")))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "use_quick"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.trajectory.len(), 2);
    let ruling = &result.trajectory[1];
    assert_eq!(ruling.parent, Some(0));
    assert_eq!(ruling.tool_name.as_deref(), Some(ARBITRATE));
    assert_eq!(ruling.inputs["branches"][0]["name"], "quick");
    assert_eq!(ruling.inputs["branches"][0]["result"], "the answer");
    // `late` was not selected, and could be started; `slow` has no tool, and is no branch.
    assert_eq!(ruling.inputs["branches"][1]["name"], "late");
    assert_eq!(ruling.inputs["branches"][1]["state"], "not started");
    assert_eq!(ruling.inputs["branches"].as_array().unwrap().len(), 2);
}

#[test]
fn an_arbiter_that_fails_leaves_the_request_to_wait_for_all_and_use_all_and_that_is_recorded() {
    struct Failing;
    impl Arbiter for Failing {
        fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
            Err("no ruling today".into())
        }
    }
    let (events, keep) = kept();
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers one part.", |_| Ok("part one")))
        .tool(Tool::new("slow", "Answers the other part.", |_| Ok("part two")))
        .arbiter(Failing)
        .record(keep)
        .model(says(&["quick", "slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.arbitration.unwrap().used, ["quick", "slow"]);
    let events = events.lock().unwrap();
    let errors: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::Arbitrated { error, .. } => error.clone(),
            _ => None,
        })
        .collect();
    assert!(!errors.is_empty() && errors.iter().all(|error| error == "no ruling today"));
}

#[test]
fn a_module_with_no_arbiter_waits_for_every_tool_and_says_nothing_of_arbitration() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", |_| Ok("the answer")))
        .tool(Tool::new("slow", "Works it out.", |_| Ok("worked out")))
        .model(says(&["quick", "slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.arbitration, None);
    assert_eq!(result.observations.len(), 2);
    assert_eq!(result.trajectory.len(), 1);
}

#[test]
fn a_tool_the_arbiter_leaves_alone_runs_unasked_and_is_waited_for() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", |_| Ok("the answer")))
        .tool(Tool::new("slow", "Keeps a note.", |_| Ok("kept")))
        .tool(Tool::new("late", "Asks someone else.", |_| Ok("not asked")))
        .arbiter(Arbitrate::new().beside(["slow"]))
        .model(says(&["quick", "slow", "use_quick"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.observations["slow"], "kept");
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["quick"]);
    assert!(arbitration.stopped.is_empty());
    let ruling = &result.trajectory[1];
    let names: Vec<&str> = ruling.inputs["branches"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|branch| branch["name"].as_str())
        .collect();
    assert_eq!(names, ["quick", "late"]);
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn what_a_branch_has_so_far_can_answer_and_the_request_does_not_wait_for_the_rest() {
    let (told, heard) = channel();
    let told = Mutex::new(told);
    let rdm = Rdm::new(ask())
        .tool(Tool::new("slow", "Answers, then goes on at length.", move |call: &Call| {
            call.partial("the answer");
            while !call.stopped() {
                std::thread::yield_now();
            }
            told.lock().unwrap().send(()).unwrap();
            Ok("the answer, and a great deal more")
        }))
        .arbiter(Arbitrate::new())
        .model(says(&["slow", "use_slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["slow"]);
    assert_eq!(arbitration.partial, ["slow"]);
    assert_eq!(arbitration.versions, [("slow".to_owned(), 1)]);
    assert_eq!(arbitration.stopped, ["slow"]);
    assert_eq!(result.observations["slow"], "the answer");
    heard.recv_timeout(PATIENCE).expect("the branch was told it was stopped");
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_branch_let_finish_is_not_waited_for_nor_stopped_and_what_it_comes_to_is_recorded() {
    let (go, wait) = channel::<()>();
    let wait = Mutex::new(wait);
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let (told, heard) = channel();
    let told = Mutex::new(told);
    let (events, keep) = kept();
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", move |_| {
            begins.lock().unwrap().recv().map_err(|_| "slow never began")?;
            Ok("the answer")
        }))
        .tool(Tool::new("slow", "Checks the answer at length.", move |call: &Call| {
            begun.lock().unwrap().send(()).map_err(|_| "no one to tell")?;
            wait.lock().unwrap().recv().map_err(|_| "never let go")?;
            told.lock().unwrap().send((call.shadowed(), call.stopped())).unwrap();
            Ok("checked")
        }))
        .arbiter(Arbitrate::new().shadows("Let `{name}` finish unwaited, so what it comes to can be checked?"))
        .record(keep)
        .model(says(&["quick", "slow", "use_quick", "shadow_slow"]));
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.used, ["quick"]);
    assert_eq!(arbitration.shadowed, ["slow"]);
    assert!(arbitration.stopped.is_empty());
    go.send(()).unwrap();
    assert_eq!(heard.recv_timeout(PATIENCE).unwrap(), (true, false));
    let deadline = std::time::Instant::now() + PATIENCE;
    let late = || {
        events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, Event::Returned { tool, .. } if tool == "slow"))
    };
    while !late() {
        assert!(std::time::Instant::now() < deadline, "what the shadow came to was not recorded");
        std::thread::yield_now();
    }
}

#[test]
fn rulings_can_be_put_to_a_decision_model_of_their_own() {
    let routes = Arc::new(Scripted::new(|request| {
        request
            .questions
            .keys()
            .map(|name| (name.clone(), Answer::noul(0.9)))
            .collect::<Vec<_>>()
    }));
    let settles = says(&["use_quick"]);
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", |_| Ok("the answer")))
        .arbiter(Arbitrate::new().model(settles.clone()))
        .model(routes.clone());
    let result = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert_eq!(result.arbitration.unwrap().used, ["quick"]);
    assert_eq!(routes.requests().len(), 1);
    assert_eq!(settles.requests().len(), 1);
    assert!(settles.requests()[0].questions.contains_key("next"));
}

#[test]
fn a_ruling_that_nothing_answers_and_nothing_could_is_unresolved_and_says_so() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Guesses.", |_| Ok("a guess")))
        .arbiter(Arbitrate::new())
        .model(says(&["quick"]));
    let arbitration = rdm.call(inputs! { "question" => "q" }).unwrap().arbitration.unwrap();
    assert!(arbitration.used.is_empty());
    assert!(arbitration.unresolved);
    assert!(!arbitration.answered);
}

#[test]
fn a_ruling_can_say_which_results_say_the_same_thing() {
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Looks it up.", |_| Ok("Paris")))
        .tool(Tool::new("slow", "Works it out.", |_| Ok("It is Paris.")))
        .arbiter(Arbitrate::new().agrees("Which result before `{name}` says the same as it?"))
        .model(says(&[
            "quick",
            "slow",
            "use_quick",
            "use_slow",
            "wait_quick",
            "wait_slow",
            "same_slow=quick",
            "same_quick=slow",
        ]));
    let mut same = Vec::new();
    let (events, keep) = kept();
    let result = rdm.record(keep).call(inputs! { "question" => "q" }).unwrap();
    // The last ruling's sameness is the request's arbitration's too.
    let summary = result.arbitration.unwrap().same;
    assert_eq!(summary.len(), 1, "{summary:?}");
    for event in events.lock().unwrap().iter() {
        if let Event::Arbitrated { ruling, .. } = event {
            same.extend(ruling.same.iter().map(|(later, earlier, _)| (later.clone(), earlier.clone())));
        }
    }
    // Whichever came back first, the later one is held to say the same as it.
    assert!(same.contains(&("slow".to_owned(), "quick".to_owned())) || same.contains(&("quick".to_owned(), "slow".to_owned())));
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_branch_in_the_middle_of_doing_something_outside_is_waited_for_and_what_it_did_is_kept() {
    let (go, wait) = channel::<()>();
    let wait = Mutex::new(wait);
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let rdm = Rdm::new(ask())
        // Comes back once `slow` is in the middle of sending, then lets the send finish.
        .tool(Tool::new("quick", "Answers quickly.", move |_| {
            begins.lock().unwrap().recv().map_err(|_| "slow never began")?;
            Ok("the answer")
        }))
        .tool(Tool::new("slow", "Sends a message.", move |call: &Call| {
            call.acting(|| {
                begun.lock().unwrap().send(()).unwrap();
                wait.lock().unwrap().recv().unwrap();
            })?;
            Ok("sent")
        }))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "slow", "use_quick", "use_slow"]));
    let (sent, came) = channel();
    std::thread::spawn(move || sent.send(rdm.call(inputs! { "question" => "q" })).is_ok());
    // The ruling would stop `slow`, but it is sending: the call waits for it.
    assert!(
        came.recv_timeout(Duration::from_millis(300)).is_err(),
        "the call came back before the send ended"
    );
    go.send(()).unwrap();
    let result = came.recv_timeout(PATIENCE).unwrap().unwrap();
    assert_eq!(result.observations["slow"], "sent");
    assert!(result.arbitration.unwrap().stopped.is_empty());
}

// Needs branches running at once: no thread is started in a WebAssembly module without threads.
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_stopped_branch_begins_nothing_outside() {
    let (told, heard) = channel();
    let told = Mutex::new(told);
    let (begun, begins) = channel();
    let (begun, begins) = (Mutex::new(begun), Mutex::new(begins));
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers quickly.", move |_| {
            begins.lock().unwrap().recv().map_err(|_| "slow never began")?;
            Ok("the answer")
        }))
        .tool(Tool::new("slow", "Sends once it is ready.", move |call: &Call| {
            begun.lock().unwrap().send(()).unwrap();
            while !call.stopped() {
                std::thread::yield_now();
            }
            told.lock().unwrap().send(call.acting(|| ()).is_err()).unwrap();
            Ok("not sent")
        }))
        .arbiter(Arbitrate::new())
        .model(says(&["quick", "slow", "use_quick"]));
    let _ = rdm.call(inputs! { "question" => "q" }).unwrap();
    assert!(heard.recv_timeout(PATIENCE).unwrap());
}

#[test]
fn a_finish_with_nothing_used_or_with_results_in_conflict_is_not_taken_and_the_likeliest_other_step_is() {
    // `next` is answered finish, most likely, then start_late, then unknown.
    let model = |uses: f64, conflict: f64| {
        Arc::new(Scripted::new(move |request| {
            let mut answers: Vec<(String, Answer)> = Vec::new();
            for name in request.questions.keys() {
                let answer = match name.as_str() {
                    "next" => {
                        let labels = request.questions["next"].labels();
                        // Waiting first while a branch runs, so the ruling that counts sees both.
                        let p = |label: &str| match label {
                            "wait" => 0.6,
                            "finish" => 0.5,
                            "start_late" => 0.3,
                            "unknown" => 0.2,
                            _ => 0.0,
                        };
                        Answer::choice(labels.iter().map(|label| (label.as_str(), p(label))))
                    }
                    "conflict" => Answer::noul(conflict),
                    // Not selected by the answers: only a ruling starts it.
                    "late" => Answer::noul(0.1),
                    name if name.starts_with("use_") => Answer::noul(uses),
                    _ => Answer::noul(0.9),
                };
                answers.push((name.clone(), answer));
            }
            answers
        }))
    };
    let rdm = |model| {
        Rdm::new(ask())
            .tool(Tool::new("quick", "Guesses.", |_| Ok("a guess")))
            .tool(Tool::new("slow", "Guesses otherwise.", |_| Ok("another guess")))
            .tool(Tool::new("late", "Finds out.", |_| Ok("found out")))
            .arbiter(Arbitrate::new())
            .model(model)
    };
    // Nothing used: a finish is not taken, and the likeliest step after it starts `late`.
    let arbitration = rdm(model(0.1, 0.1))
        .call(inputs! { "question" => "q" })
        .unwrap()
        .arbitration
        .unwrap();
    assert_eq!(arbitration.started, ["late"]);
    // Results held to disagree: the same.
    let arbitration = rdm(model(0.9, 0.9))
        .call(inputs! { "question" => "q" })
        .unwrap()
        .arbitration
        .unwrap();
    assert_eq!(arbitration.started, ["late"]);
    // Used, and agreeing: it finishes.
    let arbitration = rdm(model(0.9, 0.1))
        .call(inputs! { "question" => "q" })
        .unwrap()
        .arbitration
        .unwrap();
    assert!(arbitration.started.is_empty() && arbitration.answered);
}

// A guess the right answer shows wrong does not hold the right answer back.
#[test]
fn a_result_used_finishes_though_a_guess_not_used_disagrees_with_it() {
    let model = Arc::new(Scripted::new(|request| {
        let mut answers: Vec<(String, Answer)> = Vec::new();
        for name in request.questions.keys() {
            let answer = match name.as_str() {
                "next" => {
                    let labels = request.questions["next"].labels();
                    // Waiting is likelier than starting more, while the slow one still runs.
                    let p = |label: &str| match label {
                        "finish" => 0.5,
                        "wait" => 0.3,
                        "start_late" => 0.15,
                        _ => 0.05,
                    };
                    Answer::choice(labels.iter().map(|label| (label.as_str(), p(label))))
                }
                "conflict" => Answer::noul(0.99),
                "use_quick" => Answer::noul(0.01),
                "use_slow" => Answer::noul(0.99),
                "late" => Answer::noul(0.1),
                _ => Answer::noul(0.9),
            };
            answers.push((name.clone(), answer));
        }
        answers
    }));
    let arbitration = Rdm::new(ask())
        .tool(Tool::new("quick", "Guesses.", |_| Ok("probably Lyon")))
        .tool(Tool::new("slow", "Looks it up.", |_| Ok("Paris, by the atlas")))
        .tool(Tool::new("late", "Asks someone else.", |_| Ok("not asked")))
        .arbiter(Arbitrate::new())
        .model(model)
        .call(inputs! { "question" => "q" })
        .unwrap()
        .arbitration
        .unwrap();
    assert_eq!(arbitration.used, ["slow"]);
    assert!(arbitration.answered && !arbitration.unresolved && !arbitration.conflict);
    assert!(arbitration.started.is_empty());
}

#[test]
fn an_arbiter_is_told_whether_each_ruling_was_acted_on_and_knows_the_request_by_one_key() {
    struct Telling {
        told: Mutex<Vec<(bool, (u64, usize))>>,
        inner: Arbitrate,
    }
    impl Arbiter for Telling {
        fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
            self.inner.arbitrate(ledger, call)
        }
        fn acted(&self, ledger: &Ledger, _: &Ruling, acted: bool) {
            self.told.lock().unwrap().push((acted, ledger.request));
        }
    }
    let telling = Arc::new(Telling {
        told: Mutex::new(Vec::new()),
        inner: Arbitrate::new(),
    });
    struct Shared(Arc<Telling>);
    impl Arbiter for Shared {
        fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
            self.0.arbitrate(ledger, call)
        }
        fn acted(&self, ledger: &Ledger, ruling: &Ruling, acted: bool) {
            self.0.acted(ledger, ruling, acted)
        }
    }
    let rdm = Rdm::new(ask())
        .tool(Tool::new("quick", "Answers one part.", |_| Ok("part one")))
        .tool(Tool::new("slow", "Answers the other part.", |_| Ok("part two")))
        .arbiter(Shared(telling.clone()))
        .model(says(&["quick", "slow", "use_quick", "use_slow", "wait_quick", "wait_slow"]));
    rdm.call(inputs! { "question" => "q" }).unwrap();
    let told = telling.told.lock().unwrap().clone();
    // Every ruling after branches came back is told of, and the last was acted on.
    assert!(!told.is_empty() && told.last().unwrap().0, "{told:?}");
    // One request, one key, the same for each ruling on it.
    assert!(told.iter().all(|(_, request)| *request == told[0].1));
}
