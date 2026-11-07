//! Pictures: an input a signature says is one is shown to the model as itself.
//!
//! - A request with none is the request it always was
//! - A picture follows the state, and its input says where it is
//! - What is no picture is refused before anything is recorded or sent
//! - Keys, records and replays
//! - Fit, Near, a language model, ReAnchor

use std::sync::{Arc, Mutex};

use rdm::record::{Event, replay};
use rdm::systemone::{self, Answer, DecisionModel, Request, Scripted};
use rdm::{Json, Rdm, Signature, image, inputs};
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

    fn keys(&self) -> Vec<String> {
        let asked = self.events().into_iter().filter_map(|event| match event {
            Event::Asked { key, .. } => Some(key),
            _ => None,
        });
        asked.collect()
    }
}

fn look() -> Signature {
    Signature::new("Look at a task.")
        .input("task")
        .image("window", "The window in front.")
        .noul("editor", "Is `inputs.window` a text editor?")
}

/// A stand-in that says yes when the first picture it is shown is a red one.
fn sees_red() -> Arc<Scripted> {
    let red = image("image/png", b"red");
    Arc::new(
        Scripted::new(move |request: &Request| [("editor", Answer::noul(if request.state.get(1) == Some(&red) { 0.9 } else { 0.1 }))])
            .named("jev"),
    )
}

fn red() -> Json {
    image("image/png", b"red")
}

fn green() -> Json {
    image("image/png", b"green")
}

// A request with none is the request it always was

#[test]
fn a_signature_with_no_picture_sends_and_keys_what_it_did_before_pictures() {
    // Pinned from the code before pictures: the same bytes on the wire, and the same key.
    let plain = Signature::new("Look at a task.")
        .input("task")
        .input_with("window", "str", "The window in front.")
        .noul("editor", "Is `inputs.task` about an editor?");
    let given = json!({"task": "Save the file.", "window": "Editor"});
    let request = plain.request(given.as_object().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_string(&request).unwrap(),
        r#"{"state":{"instructions":"Look at a task.","input_fields":"1. `task` (str): \n2. `window` (str): The window in front.","inputs":{"task":"Save the file.","window":"Editor"}},"questions":{"editor":{"instructions":"Is `inputs.task` about an editor?","type":"noul"}}}"#
    );
    const KEY: &str = "6a59f8996526a1afe26c786feb26e793cf8fc62a4e7fd43a9445bbafb558796e";
    assert_eq!(systemone::key(&request, "jev"), KEY);
    let kept = Kept::default();
    let rdm = Rdm::new(plain)
        .model(Scripted::new(|_| [("editor", Answer::noul(0.9))]).named("jev"))
        .record(kept.record());
    rdm.call(inputs! { "task" => "Save the file.", "window" => "Editor" }).unwrap();
    assert_eq!(kept.keys(), [KEY]);
}

#[test]
fn a_picture_input_that_is_null_has_no_picture_this_time_and_shows_null() {
    let model = sees_red();
    let result = Rdm::new(look())
        .model(model.clone())
        .call(inputs! { "task" => "Save.", "window" => Json::Null })
        .unwrap();
    assert!(!result.yes("editor"));
    let state = &model.requests()[0].state;
    assert!(state.is_object());
    assert_eq!(state["inputs"], json!({"task": "Save.", "window": null}));
    assert_eq!(
        state["input_fields"],
        "1. `task` (str): \n2. `window` (image): The window in front."
    );
}

// A picture follows the state, and its input says where it is

#[test]
fn pictures_follow_the_state_in_the_order_their_inputs_are_declared() {
    let two = Signature::new("Compare two windows.")
        .image("before", "")
        .input("task")
        .image("after", "")
        .noul("changed", "Do `inputs.before` and `inputs.after` differ?");
    // Given in another order than they are declared in.
    let given = json!({"after": green(), "task": "Did it save?", "before": red()});
    let request = two.request(given.as_object().unwrap()).unwrap();
    assert_eq!(
        request.state,
        json!([
            {
                "instructions": "Compare two windows.",
                "input_fields": "1. `before` (image): \n2. `task` (str): \n3. `after` (image):",
                "inputs": {"after": "<image: after>", "task": "Did it save?", "before": "<image: before>"},
            },
            red(),
            green(),
        ])
    );
    assert_eq!(request.images(), 2);
}

#[test]
fn json_of_the_callers_own_that_looks_like_a_picture_is_not_lifted_out() {
    // Only the signature says what is a picture: a str input shows its value as it is.
    let mine = Signature::new("Read a value.")
        .input_typed("value", "dict")
        .noul("ok", "Is `inputs.value` fine?");
    let request = mine.request(json!({"value": red()}).as_object().unwrap()).unwrap();
    assert_eq!(request.state["inputs"]["value"], red());
    assert_eq!(request.images(), 0);
}

// What is no picture is refused before anything is recorded or sent

#[test]
fn a_value_that_is_no_picture_is_a_usage_error_before_anything_is_recorded_or_sent() {
    for wrong in [
        json!("a screenshot"),
        json!({"type": "image_url", "image_url": {"url": "https://example.com/window.png"}}),
        image("image/gif", b"gif"),
        json!(42),
    ] {
        let (kept, model) = (Kept::default(), sees_red());
        let rdm = Rdm::new(look()).model(model.clone()).record(kept.record());
        let error = rdm.call(inputs! { "task" => "Save.", "window" => wrong.clone() }).unwrap_err();
        assert!(error.is_usage(), "{wrong}: {error}");
        assert!(error.to_string().contains("'window' is a picture"), "{error}");
        assert!(model.requests().is_empty());
        let names: Vec<&str> = kept.events().iter().map(Event::name).collect();
        assert_eq!(names, ["began", "ended"], "{wrong}");
    }
}

// Keys, records and replays

#[test]
fn two_pictures_make_two_keys() {
    let kept = Kept::default();
    let rdm = Rdm::new(look()).model(sees_red()).record(kept.record());
    let seen = rdm.call(inputs! { "task" => "Save.", "window" => red() }).unwrap();
    let unseen = rdm.call(inputs! { "task" => "Save.", "window" => green() }).unwrap();
    assert!(seen.yes("editor") && !unseen.yes("editor"));
    let keys = kept.keys();
    assert_eq!(keys.len(), 2);
    assert_ne!(keys[0], keys[1]);
}

#[test]
fn a_record_without_its_pictures_keeps_their_hashes_and_replays_by_key() {
    let kept = Kept::default();
    let rdm = Rdm::new(look()).model(sees_red()).record(kept.record());
    let first = rdm.call(inputs! { "task" => "Save.", "window" => red() }).unwrap();
    // The record a module hands over has the picture whole.
    let encoded = "cmVk"; // "red", in base64
    assert!(
        kept.events()
            .iter()
            .any(|event| serde_json::to_string(event).unwrap().contains(encoded))
    );

    let stripped: Vec<Event> = kept.events().iter().map(Event::without_images).collect();
    let written: Vec<String> = stripped.iter().map(|event| serde_json::to_string(event).unwrap()).collect();
    assert!(written.iter().all(|line| !line.contains(encoded)));
    let Some(Event::Asked {
        request: Some(request), ..
    }) = stripped.iter().find(|event| event.name() == "asked")
    else {
        panic!("the request was asked");
    };
    let red_hash = systemone::Image::of(&red()).unwrap().sha256();
    assert_eq!(
        request.state[1],
        json!({"type": "image_ref", "sha256": red_hash, "content_type": "image/png"})
    );
    // The inputs the request showed still say where the picture was.
    assert_eq!(request.state[0]["inputs"]["window"], "<image: window>");

    let read: Vec<Event> = written.iter().map(|line| serde_json::from_str(line).unwrap()).collect();
    let rebuilt = Rdm::new(look()).model(replay("jev", &read));
    let again = rebuilt.call(inputs! { "task" => "Save.", "window" => red() }).unwrap();
    assert_eq!(again.trajectory, first.trajectory);
    // Another picture was never asked about.
    let other = rebuilt.call(inputs! { "task" => "Save.", "window" => green() }).unwrap_err();
    assert!(matches!(other, rdm::Error::Model(systemone::Error::Unrecorded(_))));
}

#[test]
fn without_images_strips_a_picture_wherever_a_model_could_read_it() {
    let hash = systemone::Image::of(&red()).unwrap().reference();
    // A picture in an input that is not a picture input, beside one that is.
    let mine = Signature::new("Look.")
        .input_typed("note", "dict")
        .image("window", "")
        .noul("ok", "Is `inputs.window` fine?");
    let request = mine
        .request(json!({"note": {"seen": [red()]}, "window": green()}).as_object().unwrap())
        .unwrap();
    // A state of the caller's own, an object with a picture deep in it.
    let own = Request::new(json!({"inputs": {"pages": [{"picture": red()}]}})).question("ok", systemone::Question::noul("Fine?"));
    for (request, at) in [(request, "/0/inputs/note/seen/0"), (own, "/inputs/pages/0/picture")] {
        let asked = Event::Asked {
            run: 1,
            index: 0,
            key: "k".into(),
            model: "m".into(),
            request: Some(request),
            response: None,
            error: None,
            params: Default::default(),
        };
        let Event::Asked { request: Some(kept), .. } = asked.without_images() else {
            panic!("an asked event stays one");
        };
        assert_eq!(kept.state.pointer(at), Some(&hash), "{at}");
        assert!(!serde_json::to_string(&kept).unwrap().contains("data:image"));
    }
    // What a tool returned.
    let returned = Event::Returned {
        run: 1,
        index: 0,
        output: "look".into(),
        tool: "look".into(),
        observation: json!({"shot": red(), "said": "a window"}),
    };
    let Event::Returned { observation, .. } = returned.without_images() else {
        panic!("a returned event stays one");
    };
    assert_eq!(observation, json!({"shot": hash, "said": "a window"}));
}

#[test]
fn a_request_not_asked_keeps_its_picture_in_its_inputs_and_without_images_strips_it() {
    let opened = Event::Opened {
        run: 1,
        index: 0,
        parent: None,
        depth: 0,
        iteration: 0,
        tool: None,
        inputs: Some(json!({"task": "Save.", "window": red(), "note": "plain"})),
        repeat: false,
        skipped: None,
    };
    let Event::Opened { inputs: Some(inputs), .. } = opened.without_images() else {
        panic!("an opened event stays one");
    };
    assert_eq!(inputs["window"]["type"], "image_ref");
    assert_eq!((inputs["task"].as_str(), inputs["note"].as_str()), (Some("Save."), Some("plain")));
}

// Fit, Near, a language model, ReAnchor

#[cfg(feature = "fit")]
#[test]
fn fit_cuts_the_long_text_and_shows_the_picture_whole_to_every_part() {
    use rdm::fit::{Fit, NEEDED, size};

    let covered = Signature::new("Decide whether a handbook covers what the window asks.")
        .input("handbook")
        .image("window", "The window in front.")
        .noul("covered", "Does `inputs.handbook` answer what `inputs.window` asks?");
    let mut rules: Vec<String> = (0..300)
        .map(|n| format!("Rule {n}: staff badge {n} opens door {n} on weekdays.\n"))
        .collect();
    rules[120] = "Rule 120: the nightly hotel limit in London is 180 pounds.\n".to_owned();
    let handbook = rules.concat();
    // A picture far larger than a request may show of text: it is not counted.
    let picture = image("image/png", &vec![7u8; 30_000]);
    let shown = picture.clone();
    let model = Arc::new(Scripted::new(move |request: &Request| {
        assert_eq!(request.state.get(1), Some(&shown), "every request is shown the picture");
        let found = request.state[0]["inputs"]["handbook"]
            .as_str()
            .is_some_and(|text| text.contains("hotel limit"));
        let name = request.questions.keys().next().cloned().unwrap_or_default();
        [(name, Answer::noul(if found { 0.9 } else { 0.1 }))]
    }));
    let rdm = Rdm::new(covered).fit(Fit::new().max_chars(2000)).model(model.clone());
    let result = rdm.call(inputs! { "handbook" => &handbook, "window" => picture.clone() }).unwrap();

    assert!(result.yes("covered") && result.trajectory.len() > 10);
    let requests = model.requests();
    assert!(requests.iter().filter(|request| request.questions.contains_key(NEEDED)).count() > 10);
    assert!(
        requests
            .iter()
            .all(|request| size(&request.state[0]["inputs"]) <= 2000 && request.state[1] == picture)
    );
    // The text was cut, and the picture was not.
    let kept = result.trajectory[0].inputs["handbook"].as_str().unwrap();
    assert!(kept.contains(&rules[120]) && kept.len() < 2000);
    assert_eq!(result.trajectory[0].inputs["window"], picture);
}

#[cfg(feature = "near")]
#[test]
fn near_measures_text_and_a_picture_for_its_input_is_a_usage_error() {
    use rdm::Tool;
    use rdm::near::Near;

    let dimensions = Signature::new("Say what a request is about.")
        .image("request", "")
        .noul("heating", "Is it about warmth?");
    assert!(Near::new(dimensions).check().unwrap_err().is_usage());

    // The request's input of that name is a picture.
    let about = Signature::new("Say what a request is about.")
        .input("request")
        .noul("heating", "Is `inputs.request` about warmth?");
    let operate = Signature::new("Operate a house.").image("request", "").tool("next", "What next?");
    let steps = (0..12).map(|n| Tool::new(format!("step {n}"), "", |_| Ok("done")));
    let rdm = Rdm::new(operate)
        .tools_from(steps)
        .fit(Near::new(about.clone()).top(2))
        .model(sees_red());
    let error = rdm.call(inputs! { "request" => red() }).unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("is a picture"), "{error}");

    // A picture in another input changes nothing of what is near.
    let operate = Signature::new("Operate a house.")
        .input("request")
        .image("window", "")
        .tool("next", "What next?");
    // The pointer finishes; every dimension is a maybe.
    let model = Arc::new(Scripted::new(|request: &Request| {
        let answer = |name: &String| match request.questions[name].labels() {
            labels if labels.is_empty() => Answer::noul(0.5),
            labels => Answer::chose("finish", labels.iter().map(String::as_str)),
        };
        request
            .questions
            .keys()
            .map(|name| (name.clone(), answer(name)))
            .collect::<Vec<_>>()
    }));
    let steps = (0..12).map(|n| Tool::new(format!("step {n}"), "", |_| Ok("done")));
    let rdm = Rdm::new(operate)
        .tools_from(steps)
        .fit(Near::new(about).top(2))
        .model(model.clone());
    rdm.call(inputs! { "request" => "Warmer, please.", "window" => red() }).unwrap();
    let pointed = model
        .requests()
        .into_iter()
        .find(|request| request.questions.contains_key("next"))
        .unwrap();
    assert_eq!(pointed.questions["next"].labels().len(), 3, "two steps and finish");
    assert_eq!(pointed.state[1], red());
}

#[cfg(feature = "lm")]
#[test]
fn a_language_model_that_sees_is_shown_the_pictures_and_one_that_does_not_refuses_them_unless_told_to_leave_them_out() {
    use rdm::lm::lm15::Part;
    use rdm::lm::{Lm, LmDecisionModel, ScriptedLm};

    let request = look()
        .request(json!({"task": "Save.", "window": red()}).as_object().unwrap())
        .unwrap();
    let reply = || ScriptedLm::new(|_system, _prompt| r#"{"editor": {"noul": 0.8}}"#.to_owned());

    // Not shown pictures, it refuses, as a client of a server that does not read them does, and sends nothing.
    let blind = Arc::new(reply());
    let refused = LmDecisionModel::new(Lm::shared(blind.clone(), "blind"))
        .decide(&request)
        .unwrap_err();
    assert!(matches!(refused, systemone::Error::Invalid(_)), "{refused}");
    assert!(blind.requests().is_empty());

    // Told to, it answers without them, and says so.
    let answered = LmDecisionModel::new(Lm::shared(blind.clone(), "blind"))
        .leave_out_images(true)
        .decide(&request)
        .unwrap();
    assert_eq!(answered.answers["editor"].noul, Some(0.8));
    let asked = &blind.requests()[0].messages[0];
    assert!(asked.parts.iter().all(|part| matches!(part, Part::Text(_))));
    let text = asked.text().unwrap_or_default();
    assert!(text.starts_with("1 picture of the inputs left out"), "{text}");
    assert!(text.contains("[[ ## window ## ]]\n<image: window>"), "{text}");

    let seeing = Arc::new(reply());
    LmDecisionModel::new(Lm::shared(seeing.clone(), "seeing").images(true))
        .decide(&request)
        .unwrap();
    let asked = &seeing.requests()[0].messages[0];
    let images: Vec<(&str, Option<&str>)> = asked
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::Image(picture) => Some((picture.media_type.as_str(), picture.data.as_deref())),
            _ => None,
        })
        .collect();
    assert_eq!(images, [("image/png", Some("cmVk"))]);
    assert!(!asked.text().unwrap_or_default().contains("left out"));
}

#[cfg(feature = "lm")]
#[test]
fn a_generate_never_writes_a_picture_into_its_prompt_as_text() {
    use rdm::lm::lm15::Part;
    use rdm::lm::{Generate, Lm, ScriptedLm};

    // A tool that writes, taking `window` by name from a request that showed a picture.
    let run = |generate: Generate| {
        let describe = generate.tool("describe", "Describe the window.");
        let rdm = Rdm::new(look().with_output("describe", rdm::Output::noul("Should `inputs.window` be described?")))
            .tool(describe)
            .model(Scripted::new(|request: &Request| {
                request
                    .questions
                    .keys()
                    .map(|name| (name.clone(), Answer::noul(0.9)))
                    .collect::<Vec<_>>()
            }));
        rdm.call(inputs! { "task" => "Save.", "window" => red() })
    };
    let reply = || Arc::new(ScriptedLm::new(|_system, _prompt| r#"{"reply": "An editor."}"#.to_owned()));
    let generate = |lm: Lm| Generate::parse("window -> reply").unwrap().lm(lm);

    // Not shown pictures, and not told to leave them out: the module was used wrongly, and nothing was sent.
    let blind = reply();
    let error = run(generate(Lm::shared(blind.clone(), "blind"))).unwrap_err();
    assert!(error.is_usage() && error.to_string().contains("`window` is a picture"), "{error}");
    assert!(blind.requests().is_empty());

    // Told to leave them out: the input shows where the picture was, and the model is told.
    let blind = reply();
    let result = run(generate(Lm::shared(blind.clone(), "blind")).leave_out_images(true)).unwrap();
    assert_eq!(result.observations["describe"], json!({"reply": "An editor."}));
    let asked = blind.requests()[0].messages[0].text().unwrap_or_default();
    assert!(asked.starts_with("1 picture of the inputs left out"), "{asked}");
    assert!(
        asked.contains("[[ ## window ## ]]\n<image: window>") && !asked.contains("cmVk"),
        "{asked}"
    );

    // Shown pictures: the picture is an image part, and still no text.
    let seeing = reply();
    run(generate(Lm::shared(seeing.clone(), "seeing").images(true))).unwrap();
    let asked = &seeing.requests()[0].messages[0];
    assert!(
        asked
            .parts
            .iter()
            .any(|part| matches!(part, Part::Image(picture) if picture.data.as_deref() == Some("cmVk")))
    );
    assert!(!asked.text().unwrap_or_default().contains("cmVk"));
}

#[cfg(feature = "reanchor")]
#[test]
fn reanchor_fits_from_a_run_that_was_shown_pictures_kept_whole_or_not() {
    use rdm::reanchor::{ReAnchor, Want, gather};
    use rdm::{Ask, Params};

    // A shy model: its yes for a red window sits at about 0.35, its no at 0.2.
    let shy = Scripted::new(|request: &Request| {
        let red = request.state.get(1) == Some(&image("image/png", b"red"));
        let said = request.state[0]["inputs"]["task"].as_str().unwrap_or_default().len() as f64;
        [("editor", Answer::noul(if red { 0.33 + said / 1000.0 } else { 0.2 }))]
    })
    .named("edge");
    let kept = Kept::default();
    let rdm = Rdm::new(look()).model(shy).record(kept.record());
    let tasks = ["save", "save it", "save the file", "save all", "save now"];
    for (i, task) in tasks.iter().enumerate() {
        rdm.call_with(Ask::new(inputs! { "task" => *task, "window" => red() }).tag(2 * i))
            .unwrap();
        rdm.call_with(Ask::new(inputs! { "task" => *task, "window" => green() }).tag(2 * i + 1))
            .unwrap();
    }
    let label = |tag: &Json, output: &str| {
        let i = tag.as_u64()?;
        (output == "editor").then_some(Want::Noul(i % 2 == 0))
    };
    let whole = gather(kept.events().iter(), label);
    let stripped: Vec<Event> = kept.events().iter().map(Event::without_images).collect();
    let without = gather(stripped.iter(), label);
    assert_eq!(whole.evidence.len(), 10);
    assert_eq!(without.evidence.len(), 10);
    let (fitted, again) = (ReAnchor::new().fit(&whole.evidence), ReAnchor::new().fit(&without.evidence));
    assert_eq!(fitted.params, again.params);
    let threshold = fitted.params["editor"].threshold.unwrap();
    assert!(threshold > 0.2 && threshold < 0.33, "{threshold}");
    rdm.set_params("editor", Params::threshold(threshold));
    assert!(rdm.call(inputs! { "task" => "save", "window" => red() }).unwrap().yes("editor"));
}
