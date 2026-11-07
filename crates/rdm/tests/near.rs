#![cfg(feature = "near")]
//! Near: labels for the names on a map, and the names nearest a request.
//!
//! - Coordinates: a text's answers to named questions
//! - A map cut to what is nearest a request
//! - Dimensions written while the module runs
//! - Thresholds
//! - A program that is saved, loaded and copied
//! - With other fits
//! - The two notebook cells, beside what the Python module asked and printed for them

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use rdm::near::{Coordinates, Given, Labels, MEASURES, Measure, Near, Saved, coordinates, text};
use rdm::systemone::{Answer, Kind, Question, Request, Scripted};
use rdm::{Ask, Entry, Error, Json, Output, Params, Rdm, Reason, Shared, Signature, Tool, inputs};
use serde_json::json;

/// How likely each word makes each dimension. What a text says of a dimension is its likeliest word's.
const WORDS: [(&str, &[(&str, f64)]); 3] = [
    ("heating", &[("cooler", 0.9), ("warmer", 0.9), ("degrees", 0.8), ("stuffy", 0.6)]),
    ("lights", &[("light", 0.9), ("lamp", 0.8), ("dim", 0.5)]),
    ("doors", &[("door", 0.9), ("lock", 0.8)]),
];

fn likely(words: &[(&str, &[(&str, f64)])], name: &str, text: &str) -> f64 {
    let of = words
        .iter()
        .find(|(dimension, _)| *dimension == name)
        .map_or(&[][..], |(_, words)| *words);
    of.iter()
        .filter(|(word, _)| text.contains(word))
        .map(|(_, p)| *p)
        .fold(0.0, f64::max)
}

/// What the stand-in chooses for a question that is not yes or no, by the words of the text.
fn picked(name: &str, text: &str) -> String {
    if name == "urgency" {
        // a score: the index of its level
        let level = if text.contains("today") {
            "2"
        } else if text.contains("week") {
            "1"
        } else {
            "0"
        };
        return level.to_owned();
    }
    if name == "room" {
        let room = ["hall", "kitchen"].into_iter().find(|room| text.contains(room));
        return room.unwrap_or("None").to_owned();
    }
    // How a request uses a dimension: a filter where it says it must be that, a boost where it mentions it.
    let dimension = name.strip_suffix("_use").unwrap_or(name);
    let used = if text.contains(&format!("must be {dimension}")) {
        "filter"
    } else if likely(&WORDS, dimension, text) > 0.0 {
        "boost"
    } else {
        "ignore"
    };
    used.to_owned()
}

/// What the stand-in says to one question: a dimension is as likely as the words of the text make it, and a
/// pointer takes the names of a script in turn, or with none the first tool offered, once.
fn says(name: &str, question: &Question, inputs: &Json, script: &Mutex<VecDeque<String>>) -> Answer {
    let text = inputs["request"].as_str().unwrap_or_default().to_lowercase();
    let labels = question.labels();
    let among = || labels.iter().map(String::as_str);
    match question.kind {
        Kind::Noul => Answer::noul(likely(&WORDS, name, &text)),
        Kind::Score => Answer::score(labels.iter().map(|level| if *level == picked(name, &text) { 1.0 } else { 0.0 })),
        Kind::Choice if name == "room" || name.ends_with("_use") => Answer::chose(&picked(name, &text), among()),
        Kind::Choice => {
            let seen = inputs["observation"].as_str().is_some_and(|seen| !seen.is_empty());
            let next = script.lock().unwrap().pop_front();
            let chosen = next.unwrap_or_else(|| if seen { "finish".to_owned() } else { labels[0].clone() });
            Answer::chose(&chosen, among())
        }
    }
}

/// A stand-in System One model, under a name, with a script for the pointer.
fn client_as(name: &str, script: &[&str]) -> Arc<Scripted> {
    let script = Mutex::new(script.iter().map(|name| name.to_string()).collect::<VecDeque<_>>());
    let answers = move |request: &Request| {
        let inputs = &request.state["inputs"];
        let each = request.questions.iter();
        each.map(|(name, question)| (name.clone(), says(name, question, inputs, &script)))
            .collect::<Vec<_>>()
    };
    Arc::new(Scripted::new(answers).named(name))
}

fn client() -> Arc<Scripted> {
    client_as("jev-test", &[])
}

/// The requests that asked a question of that name.
fn asked(model: &Scripted, name: &str) -> Vec<Request> {
    let requests = model.requests().into_iter();
    requests.filter(|request| request.questions.contains_key(name)).collect()
}

/// The texts a question of that name was asked about.
fn texts(model: &Scripted, name: &str) -> Vec<Json> {
    let requests = asked(model, name);
    requests.iter().map(|request| request.state["inputs"]["request"].clone()).collect()
}

/// What each request that pointed was offered.
fn offered(model: &Scripted) -> Vec<Vec<String>> {
    let requests = asked(model, "next");
    requests.iter().map(|request| request.questions["next"].labels()).collect()
}

/// What the request that pointed so many before the last was offered: 1 is the last.
fn offered_back(model: &Scripted, back: usize) -> Vec<String> {
    let offered = offered(model);
    offered[offered.len() - back].clone()
}

/// The entries of a run that asked a question of that name.
fn entries<'a>(trajectory: &'a [Entry], name: &str) -> Vec<&'a Entry> {
    let has = |entry: &&Entry| entry.outputs.as_ref().is_some_and(|outputs| outputs.contains_key(name));
    trajectory.iter().filter(has).collect()
}

fn close(one: f64, other: f64) {
    assert!((one - other).abs() < 1e-9, "{one} is not {other}");
}

fn about() -> Signature {
    Signature::new("Say what a text is about.")
        .input("request")
        .noul("heating", "Is `inputs.request` about how warm it is?")
        .noul("lights", "Is `inputs.request` about a light?")
}

fn blank() -> Signature {
    Signature::new("Say what a text is about.").input("request")
}

fn operate() -> Signature {
    Signature::new("Operate a house, one step at a time.")
        .input("request")
        .input("observation")
        .tool("next", "What should happen next for `inputs.request`?")
}

/// A tool that says its own name.
fn step(name: &str, description: &str) -> Tool {
    let did = format!("did {name}");
    Tool::new(name, description, move |_| Ok(did.clone()))
}

/// A map of six lights and two heating steps.
fn steps() -> Vec<Tool> {
    let rooms = ["hall", "kitchen", "porch", "attic", "cellar", "study"];
    let lights = rooms.map(|room| step(&format!("{room} light"), &format!("Turn the {room} light on.")));
    let heating = [
        step("cooler", "Make it one degree cooler."),
        step("warmer", "Make it one degree warmer."),
    ];
    lights.into_iter().chain(heating).collect()
}

fn house(near: &Near, model: &Arc<Scripted>) -> Rdm {
    Rdm::new(operate()).tools_from(steps()).fit(near.clone()).model(model.clone())
}

const COOLER: &str = "Two degrees cooler, please.";

// Coordinates: a text's answers to named questions

#[test]
fn a_text_has_a_coordinate_for_each_dimension_and_is_asked_once() {
    let model = client();
    let near = Near::new(about()).model(model.clone());
    // Every number has a name: how likely a yes is, as the model answered.
    assert_eq!(
        near.embed(["Make it cooler.", "The lamp in the hall."]).unwrap(),
        [
            coordinates([("heating", 0.9), ("lights", 0.0)]),
            coordinates([("heating", 0.0), ("lights", 0.8)])
        ]
    );
    // A text is asked once, all its dimensions in one request, and what it answered is kept.
    assert_eq!(
        near.embed(["The lamp in the hall.", "Nothing of either."]).unwrap(),
        [
            coordinates([("heating", 0.0), ("lights", 0.8)]),
            coordinates([("heating", 0.0), ("lights", 0.0)])
        ]
    );
    let about = texts(&model, "heating");
    assert_eq!(
        (
            about.iter().filter(|text| **text == "The lamp in the hall.").count(),
            model.requests().len()
        ),
        (1, 3)
    );
    assert!(
        model
            .requests()
            .iter()
            .all(|request| request.questions.keys().eq(["heating", "lights"]))
    );
    assert_eq!(json!(near.answers()["Make it cooler."]), json!({"heating": 0.9, "lights": 0.0}));
    // The same text twice in one call is asked once.
    assert_eq!(near.embed(["Warmer.", "Warmer."]).unwrap().len(), 2);
    assert_eq!(model.requests().len(), 4);
    // Things are compared by what their coordinates share: a search by named dimensions, with no map.
    let embedded = near.embed(["Two degrees cooler.", "A lamp.", "A lamp, and warmer."]).unwrap();
    let (cooler, lamp, both) = (&embedded[0], &embedded[1], &embedded[2]);
    close(near.score(cooler, cooler), 0.81);
    close(near.score(cooler, lamp), 0.0);
    close(near.score(cooler, both), 0.81);
}

#[test]
fn the_dimensions_are_one_input_and_a_decision_for_each() {
    let usage = |error: Error| match error {
        Error::Usage(message) => message,
        other => panic!("{other:?} is not a usage error"),
    };
    let pointing = Near::new(about().tool("kind", "Which tool is `inputs.request` for?"));
    assert_eq!(
        usage(pointing.check().unwrap_err()),
        "The dimensions are a signature with one input and a `Noul`, `Score` or `Choice` output for each dimension; ['kind'] are not."
    );
    // It is said by everything that asks, and not only by `check`.
    assert!(pointing.embed(["Cooler."]).unwrap_err().is_usage());
    assert_eq!(
        usage(Near::new(about().input("more")).check().unwrap_err()),
        "The dimensions are a signature with one input and a `Noul`, `Score` or `Choice` output for each dimension."
    );
    assert_eq!(
        usage(Near::new(about()).top(0).check().unwrap_err()),
        "top must be at least 1: a request has to offer something."
    );
    // The dimensions' input is the request's input of the same name.
    let ask = Signature::new("Operate a house.")
        .input("question")
        .input("observation")
        .tool("next", "What next?");
    let rdm = Rdm::new(ask).tools_from(steps()).fit(Near::new(about()).top(2)).model(client());
    assert_eq!(
        usage(rdm.call(inputs! { "question" => "Cooler." }).unwrap_err()),
        "The dimensions take `request`, and the request has only ['observation', 'question']: name the dimensions' input after the \
         request's input that says what is wanted."
    );
    // Used alone, it asks a model of its own, and says so where it has none.
    let alone = Near::new(about());
    assert!(usage(alone.embed(["Cooler."]).unwrap_err()).starts_with("The dimensions have no decision model of their own"));
}

#[test]
fn a_dimension_can_be_a_score_or_a_choice_and_its_parameters_are_read_when_a_label_is() {
    let sorted = Signature::new("Say what a text is about.")
        .input("request")
        .noul("heating", "Is `inputs.request` about how warm it is?")
        .score(
            "urgency",
            "How soon does `inputs.request` need doing?",
            ["whenever", "this week", "today"],
        )
        .choice_or_none(
            "room",
            "Which room is `inputs.request` about?",
            ["hall", "kitchen"],
            "No room is named.",
        );
    let model = client();
    let near = Near::new(sorted).model(model.clone());
    let both = ["Cooler in the kitchen, today.", "The hall, some time this week."];
    // A score is how far up its rubric the text is, and a choice is a dimension for each option but "none of these".
    let embedded = near.embed(both).unwrap();
    assert_eq!(
        embedded[0],
        coordinates([("heating", 0.9), ("urgency", 1.0), ("room.hall", 0.0), ("room.kitchen", 1.0)])
    );
    assert_eq!(
        embedded[1],
        coordinates([("heating", 0.0), ("urgency", 0.5), ("room.hall", 1.0), ("room.kitchen", 0.0)])
    );
    assert_eq!(near.columns(), ["heating", "urgency", "room.hall", "room.kitchen"]);
    // What is kept is what the model gave. Cuts give a score its level and weights multiply a choice's options:
    // both are read when a label is, so nothing is asked again.
    assert_eq!(
        json!(near.answers()[both[1]]["urgency"]),
        json!({"probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": 1.0})
    );
    near.set_params("urgency", Params::cuts([1.5, 1.8]));
    near.set_params("room", Params::weights([("kitchen", 0.5)]));
    let embedded = near.embed(both).unwrap();
    let (today, week) = (&embedded[0], &embedded[1]);
    assert_eq!(
        (today["urgency"], week["urgency"], today["room.kitchen"], week["room.hall"]),
        (1.0, 0.0, 0.5, 1.0)
    );
    assert_eq!(model.requests().len(), 2);
}

#[test]
fn a_choice_with_one_option_is_a_dimension_that_is_not_asked() {
    // As in a run: one option leaves nothing to decide, so it is answered with that option, as certain.
    let kinds = blank().choice("kind", "Which kind of text is `inputs.request`?", ["request"]);
    let model = client();
    let near = Near::new(kinds.clone()).model(model.clone());
    assert_eq!(near.embed(["Cooler."]).unwrap(), [coordinates([("kind.request", 1.0)])]);
    assert!(model.requests().is_empty());
    let near = Near::new(kinds.noul("heating", "Is `inputs.request` about how warm it is?")).model(model.clone());
    assert_eq!(
        near.embed(["Cooler."]).unwrap(),
        [coordinates([("kind.request", 1.0), ("heating", 0.9)])]
    );
    assert!(model.requests()[0].questions.keys().eq(["heating"]));
}

#[test]
fn things_are_compared_by_the_measure_chosen() {
    let request = coordinates([("heating", 1.0), ("lights", 0.0)]);
    let one = coordinates([("heating", 0.8), ("lights", 0.0)]);
    let both = coordinates([("heating", 1.0), ("lights", 1.0)]);
    let scores = |measure: &str| {
        let near = Near::new(about()).measure(measure.parse().unwrap());
        [&one, &both].map(|thing| (near.score(&request, thing) * 1000.0).round() / 1000.0)
    };
    assert_eq!(MEASURES, ["cosine", "dot", "euclidean", "jaccard", "manhattan"]);
    assert_eq!(scores("dot"), [0.8, 1.0]); // what is shared, as much as both have of it: a dimension only the thing has costs it nothing
    assert_eq!(scores("cosine"), [1.0, 0.707]); // the direction shared: a dimension only the thing has counts against it
    assert_eq!(scores("euclidean"), [0.833, 0.5]); // one over one plus the distance
    assert_eq!(scores("manhattan"), [0.833, 0.5]);
    assert_eq!(scores("jaccard"), [0.8, 0.5]); // what is shared over what either has
    // A measure of your own is a function of the request's coordinates and a thing's.
    let lit = Near::new(about()).measure(Measure::custom(|_, thing| thing["lights"]));
    assert_eq!(lit.score(&request, &both), 1.0);
    assert_eq!(
        "nearest".parse::<Measure>().unwrap_err(),
        Error::Usage(
            "measure must be one of ['cosine', 'dot', 'euclidean', 'jaccard', 'manhattan'], or a function of two sets of coordinates; \
             got 'nearest'."
                .to_owned()
        )
    );
    // The measure orders what is near. What shares nothing with the request is not near under any
    // measure, though a distance gives it a number: the lights are not offered for a request about heating.
    for measure in MEASURES {
        let model = client();
        let near = Near::new(about()).top(3).measure(measure.parse().unwrap());
        house(&near, &model).call(inputs! { "request" => COOLER }).unwrap();
        assert_eq!(offered(&model)[0], ["cooler", "warmer", "finish"], "{measure}");
    }
    // Where the order differs, the map is cut differently. A step about heating and a light is as near as any by the
    // dot product, so it is kept as the first on the map; by the cosine its other dimension puts it further off.
    for (measure, first) in [(Measure::Dot, "both"), (Measure::Cosine, "cooler")] {
        let model = client();
        let tools = std::iter::once(step("both", "Make it warmer and turn on a light.")).chain(steps());
        let near = Near::new(about()).top(1).measure(measure);
        let rdm = Rdm::new(operate()).tools_from(tools).fit(near).model(model.clone());
        rdm.call(inputs! { "request" => COOLER }).unwrap();
        assert_eq!(offered(&model)[0], [first, "finish"]);
    }
}

// A map cut to what is nearest a request

#[test]
fn a_pointer_is_offered_the_tools_nearest_the_request() {
    let model = client();
    let near = Near::new(about()).top(3);
    let rdm = house(&near, &model);
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    // The two heating steps share a dimension with the request and the lights share none. `finish` is always offered.
    assert_eq!(offered(&model)[0], ["cooler", "warmer", "finish"]);
    assert_eq!(
        json!(result.observations["next"]),
        json!([["cooler", "did cooler"], ["finish", null]])
    );
    // Each name on the map was asked once, and the request once: the second turn asks nothing again.
    assert_eq!((asked(&model, "heating").len(), asked(&model, "next").len()), (8 + 1, 2));
    // What it asked are sub-calls of the run, for the answers only: in the trajectory, one level down, under
    // the request that was being fitted, the request's own text first and then the names in the order of the map.
    let labelled = entries(&result.trajectory, "heating");
    assert_eq!(labelled.len(), 9);
    for entry in &labelled {
        assert_eq!((entry.depth, entry.parent, entry.tool_name.as_deref()), (1, Some(0), None));
        assert_eq!(entry.observations, Some(IndexMap::new()));
    }
    let shown: Vec<&str> = labelled.iter().map(|entry| entry.inputs["request"].as_str().unwrap()).collect();
    assert_eq!(
        shown[..3],
        [
            COOLER,
            "hall light: Turn the hall light on.",
            "kitchen light: Turn the kitchen light on."
        ]
    );
    assert_eq!(shown[8], "warmer: Make it one degree warmer.");
    let pointed: Vec<Json> = entries(&result.trajectory, "next")
        .iter()
        .map(|entry| entry.value("next"))
        .collect();
    assert_eq!(pointed, ["cooler", "finish"]);
    assert_eq!(near.answers().len(), 9);
    // The next request asks about itself only, however long the map is.
    let later = rdm.call(inputs! { "request" => "It is dim: a lamp, please." }).unwrap();
    assert_eq!((asked(&model, "heating").len(), later.trajectory.len()), (8 + 2, 3));
    assert_eq!(offered(&model)[2], ["hall light", "kitchen light", "porch light", "finish"]);
}

#[test]
fn what_near_asks_counts_against_the_runs_limits() {
    let model = client();
    let near = Near::new(about()).top(3);
    let rdm = house(&near, &model).max_predictor_calls(4);
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    // The run's first request and three of the nine about dimensions were asked; a limit stopped the rest, and the run says so.
    assert_eq!((model.requests().len(), result.termination_reason), (4, Reason::MaxPredictorCalls));
    // They were admitted in the order given, and a text whose request was stopped is not kept.
    assert_eq!(
        near.answers().keys().collect::<Vec<_>>(),
        [
            COOLER,
            "hall light: Turn the hall light on.",
            "kitchen light: Turn the kitchen light on."
        ]
    );
    assert_eq!(result.trajectory.iter().filter(|entry| entry.skipped.is_some()).count(), 6 + 1);
    // Labelling a long map is done before the run, outside it: then a run asks about its request only.
    let near = Near::new(about()).top(3).model(model.clone());
    let rdm = house(&near, &model).max_predictor_calls(4);
    let names = rdm.tools().all();
    near.embed(names.iter().map(|tool| text(tool.name(), tool.description()))).unwrap();
    assert_eq!(near.answers().len(), 8);
    assert_eq!(rdm.call(inputs! { "request" => COOLER }).unwrap().termination_reason, Reason::Final);
}

#[test]
fn a_map_cut_to_one_name_leaves_nothing_to_ask() {
    // With no `finish` on offer and the one nearest name kept, the pointer has one option: it is
    // answered with it, as a choice with one option is, and the model is asked only what a text is about.
    let model = client();
    let near = Near::new(about()).top(1);
    let rdm = house(&near, &model).finish(false).max_iters(2);
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert!(asked(&model, "next").is_empty());
    assert_eq!(result.pointed("next"), ["cooler", "cooler"]);
    assert_eq!(result.termination_reason, Reason::MaxIters);
    let turns = entries(&result.trajectory, "next");
    assert_eq!(turns.len(), 2);
    for turn in turns {
        assert_eq!(
            (turn.unasked.clone(), turn.tokens, turn.model.clone()),
            (Some(vec!["next".to_owned()]), Some(0), None)
        );
        assert_eq!(
            json!(turn.outputs.as_ref().unwrap()["next"]),
            json!({"confidence": 1.0, "value": "cooler", "probabilities": {"cooler": 1.0}})
        );
    }
    // A turn that asked nothing took no place: the request and the eight names are the nine that were asked.
    assert_eq!(model.requests().len(), 9);
    let limited = house(&Near::new(about()).top(1), &model)
        .finish(false)
        .max_iters(2)
        .max_predictor_calls(10);
    assert_eq!(
        limited.call(inputs! { "request" => COOLER }).unwrap().termination_reason,
        Reason::MaxIters
    );
}

#[test]
fn a_map_is_left_whole_where_it_is_short_or_nothing_sets_its_names_apart() {
    let model = client();
    // Eight names and room for eight: nothing is asked.
    let rdm = house(&Near::new(about()).top(8), &model);
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert!(asked(&model, "heating").is_empty());
    assert_eq!(offered(&model)[0].len(), 9);
    // A request that shares nothing with any name has no direction, so every name is offered.
    let rdm = house(&Near::new(about()).top(3), &model);
    rdm.call(inputs! { "request" => "Do whatever is best." }).unwrap();
    assert_eq!(offered_back(&model, 2).len(), 9);
    // The names the module always offers are not measured: a tool that takes no side, such as one that writes.
    let rdm = house(&Near::new(about()).top(2), &model).always(["hall light"]);
    rdm.call(inputs! { "request" => "Warmer, by two degrees." }).unwrap();
    assert_eq!(offered_back(&model, 2), ["cooler", "warmer", "hall light", "finish"]);
}

#[test]
fn names_that_are_as_near_as_each_other_are_kept_in_the_order_they_had() {
    // Three steps that are as much about the three dimensions as each other, the amounts in another order. A sum
    // is added up as the Python module adds it, with what each addition rounds away carried along, so two that
    // are as near as each other come out so, and the first on the map is kept.
    const TABLE: [(&str, [f64; 3]); 4] = [
        ("falling", [0.3, 0.2, 0.1]),
        ("level", [0.2, 0.2, 0.2]),
        ("rising", [0.1, 0.2, 0.3]),
        ("Whatever", [1.0, 1.0, 1.0]),
    ];
    let table = || {
        Arc::new(Scripted::new(|request: &Request| {
            let text = request.state["inputs"]["request"].as_str().unwrap_or_default();
            let row = TABLE.iter().find(|(name, _)| text.starts_with(name)).map(|(_, row)| *row);
            let answer = |(position, (name, question)): (usize, (&String, &Question))| {
                let answer = match question.kind {
                    Kind::Noul => Answer::noul(row.unwrap()[position]),
                    _ => Answer::chose("finish", question.labels().iter().map(String::as_str)),
                };
                (name.clone(), answer)
            };
            request.questions.iter().enumerate().map(answer).collect::<Vec<_>>()
        }))
    };
    let dimensions = || {
        ["a", "b", "c"].into_iter().fold(blank(), |signature, name| {
            signature.noul(name, format!("Is `inputs.request` {name}?"))
        })
    };
    // What the Python module offered for this map, and how near it said each step is.
    let python = [
        ("dot", ["level"], ["falling", "level"], [0.6, 0.6000000000000001, 0.6]),
        (
            "cosine",
            ["level"],
            ["falling", "level"],
            [0.9258200997725513, 1.0, 0.9258200997725513],
        ),
        (
            "euclidean",
            ["level"],
            ["falling", "level"],
            [0.4179136465089488, 0.4191746152772845, 0.4179136465089488],
        ),
        (
            "manhattan",
            ["falling"],
            ["falling", "rising"],
            [0.29411764705882354, 0.2941176470588235, 0.29411764705882354],
        ),
        (
            "jaccard",
            ["level"],
            ["falling", "level"],
            [0.19999999999999998, 0.20000000000000004, 0.19999999999999998],
        ),
    ];
    for (measure, one, two, scores) in python {
        for (top, kept) in [(1, &one[..]), (2, &two[..])] {
            let model = table();
            let near = Near::new(dimensions()).top(top).measure(measure.parse().unwrap());
            let tools = ["falling", "level", "rising"].map(|name| step(name, "A step."));
            let rdm = Rdm::new(operate()).tools_from(tools).fit(near.clone()).model(model.clone());
            rdm.call(inputs! { "request" => "Whatever" }).unwrap();
            let offered = &offered(&model)[0];
            assert_eq!(offered[..offered.len() - 1], *kept, "{measure}, the {top} nearest");
            let texts = ["Whatever", "falling: A step.", "level: A step.", "rising: A step."];
            let embedded = near.clone().model(model.clone()).embed(texts).unwrap();
            let said: Vec<f64> = embedded[1..].iter().map(|thing| near.score(&embedded[0], thing)).collect();
            assert_eq!(said, scores, "{measure}");
        }
    }
    // The map that was found to be cut otherwise: three of its steps are as far from the request as each other.
    let request = coordinates([
        ("d1", 0.0),
        ("d2", 0.0),
        ("d3.a", 0.2 * 0.8),
        ("d3.b", 0.3 * 0.8),
        ("w1", 0.3 * 0.5),
    ]);
    let one = coordinates([("d1", 0.6), ("d2", 0.0), ("d3.a", 0.4), ("d3.b", 0.4), ("w1", 1.0)]);
    let other = coordinates([("d1", 0.0), ("d2", 1.0), ("d3.a", 0.4), ("d3.b", 0.4), ("w1", 0.6)]);
    assert_eq!(Measure::Manhattan.score(&request, &one), 0.3508771929824561);
    assert_eq!(Measure::Manhattan.score(&request, &other), 0.3508771929824561);
}

#[test]
fn a_request_can_say_how_it_uses_each_dimension() {
    let request = "A lamp would be nice, and cooler: it must be heating.";
    // As it is, the request is as much about lights as about heating, and the three nearest include a light.
    let model = client();
    house(&Near::new(about()).top(3), &model)
        .call(inputs! { "request" => request })
        .unwrap();
    assert_eq!(offered(&model)[0], ["hall light", "cooler", "warmer", "finish"]);
    // Asked how it uses each dimension, it says heating is a filter and lights a boost: only the heating steps will do.
    let model = client();
    let near = Near::new(about()).top(3).uses(true);
    let rdm = house(&near, &model);
    let result = rdm.call(inputs! { "request" => request }).unwrap();
    assert_eq!(offered(&model)[0], ["cooler", "warmer", "finish"]);
    // That is one request more for each request, about its own text, kept as the labels are and counted in the run.
    let uses = asked(&model, "heating_use");
    assert_eq!(uses.len(), 1);
    assert_eq!(
        serde_json::to_value(&uses[0]).unwrap(),
        json!({
            "state": {
                "instructions": "Say how a request uses each label of the things it could be given.",
                "input_fields": "1. `request` (str):",
                "inputs": {"request": request},
            },
            "questions": {
                "heating_use": {
                    "instructions": "How does `inputs.request` use the label `heating`, which asks of a thing: Is `inputs.request` about how warm it is?",
                    "type": "choice",
                    "criteria": {"filter": "Only what has this label will do.", "boost": "What has this label is better.", "ignore": "This label does not matter here."},
                },
                "lights_use": {
                    "instructions": "How does `inputs.request` use the label `lights`, which asks of a thing: Is `inputs.request` about a light?",
                    "type": "choice",
                    "criteria": {"filter": "Only what has this label will do.", "boost": "What has this label is better.", "ignore": "This label does not matter here."},
                },
            },
        })
    );
    let used = entries(&result.trajectory, "heating_use");
    assert_eq!((used.len(), used[0].depth, used[0].parent), (1, 1, Some(0)));
    assert_eq!(
        json!(near.labels().uses),
        json!({request: {
            "heating_use": {"probabilities": {"filter": 1.0, "boost": 0.0, "ignore": 0.0}, "confidence": 1.0},
            "lights_use": {"probabilities": {"filter": 0.0, "boost": 1.0, "ignore": 0.0}, "confidence": 1.0},
        }})
    );
    // The same request again asks nothing about itself again.
    rdm.call(inputs! { "request" => request }).unwrap();
    assert_eq!((asked(&model, "heating_use").len(), asked(&model, "heating").len()), (1, 9));
    // A dimension the request ignores counts for nothing, though the request mentions nothing else.
    rdm.call(inputs! { "request" => "Do whatever is best." }).unwrap();
    assert_eq!(offered_back(&model, 2).len(), 9);
    // A filter that would leave no name is let go: a house with no heating is offered its lights.
    let model = client();
    let near = Near::new(about()).top(3).uses(true);
    let lights = Rdm::new(operate())
        .tools_from(steps().into_iter().take(6))
        .fit(near)
        .model(model.clone());
    lights.call(inputs! { "request" => request }).unwrap();
    assert_eq!(offered(&model)[0], ["hall light", "kitchen light", "porch light", "finish"]);
}

#[test]
fn how_a_dimension_is_used_has_weights_that_are_read_when_it_is() {
    // A model that is torn about how the request uses heating: a filter, by a little.
    let script = Mutex::new(VecDeque::new());
    let torn = Arc::new(Scripted::new(move |request: &Request| {
        let answer = |(name, question): (&String, &Question)| {
            let answer = match name.as_str() {
                "heating_use" => Answer::choice([("filter", 0.5), ("boost", 0.4), ("ignore", 0.1)]),
                _ => says(name, question, &request.state["inputs"], &script),
            };
            (name.clone(), answer)
        };
        request.questions.iter().map(answer).collect::<Vec<_>>()
    }));
    let request = "A lamp would be nice, and cooler: it must be heating.";
    let near = Near::new(about()).top(3).uses(true);
    let rdm = Rdm::new(operate()).tools_from(steps()).fit(near.clone()).model(torn.clone());
    rdm.call(inputs! { "request" => request }).unwrap();
    assert_eq!(offered(&torn)[0], ["cooler", "warmer", "finish"]);
    // With a filter weighted down, heating is a boost, and counts for a little less than the lights do: what the
    // Python module offers for the same answers and the same weight. Nothing is asked again.
    near.set_use_params("heating", Params::weights([("filter", 0.5)]));
    rdm.call(inputs! { "request" => request }).unwrap();
    assert_eq!(offered(&torn)[2], ["hall light", "kitchen light", "porch light", "finish"]);
    assert_eq!((asked(&torn, "heating_use").len(), asked(&torn, "heating").len()), (1, 9));
    assert_eq!(
        near.all_use_params(),
        IndexMap::from([("heating".to_owned(), Params::weights([("filter", 0.5)]))])
    );
    near.unset_use_params("heating");
    rdm.call(inputs! { "request" => request }).unwrap();
    assert_eq!(offered(&torn)[4], ["cooler", "warmer", "finish"]);
}

#[test]
fn near_is_about_the_map_and_leaves_any_other_choice_alone() {
    let options = ["hall lamp", "attic lamp", "two degrees cooler", "one degree warmer"];
    let pick = Signature::new("Pick a step.").input("request").with_output(
        "step",
        Output::choice("Which step does `inputs.request` ask for?", options).or_none("None of these."),
    );
    let model = client();
    let rdm = Rdm::new(pick).fit(Near::new(about()).top(2)).model(model.clone());
    rdm.call(inputs! { "request" => "Cooler, please." }).unwrap();
    // A choice that is not the map is for `Fit`, where it is too long. Nothing was asked about dimensions.
    assert_eq!(asked(&model, "step")[0].questions["step"].labels().len(), 5);
    assert!(asked(&model, "heating").is_empty());
}

#[test]
fn a_request_that_is_not_a_text_is_asked_about_as_json() {
    let ask = Signature::new("Operate a house.")
        .input_typed("request", "dict")
        .input("observation")
        .tool("next", "What next?");
    let model = client();
    let rdm = Rdm::new(ask)
        .tools_from(steps())
        .fit(Near::new(about()).top(3))
        .model(model.clone());
    let wanted = json!({"wants": "cooler", "by": 2, "rooms": ["hall", null], "share": 0.5});
    rdm.call(inputs! { "request" => wanted }).unwrap();
    // As the Python module writes it: `json.dumps`, with a space after each comma and colon.
    assert!(texts(&model, "heating").contains(&json!(r#"{"wants": "cooler", "by": 2, "rooms": ["hall", null], "share": 0.5}"#)));
    assert_eq!(offered(&model)[0], ["cooler", "warmer", "finish"]);
}

#[test]
fn a_request_that_is_not_a_text_is_written_as_python_writes_json() {
    // What the Python module asked about for each of these requests: `json.dumps`, which writes whatever is not
    // plain ASCII as an escape. The text is what the model reads, and the name the labels are kept under.
    let written = [
        (
            json!({"wants": "café", "emoji": "🙂", "quote": "say \"x\"", "line": "a\nb\tc", "ctl": "\u{1}\u{7f}", "slash": "a/b\\c"}),
            r#"{"wants": "caf\u00e9", "emoji": "\ud83d\ude42", "quote": "say \"x\"", "line": "a\nb\tc", "ctl": "\u0001\u007f", "slash": "a/b\\c"}"#,
        ),
        (
            json!(["hall", null, true, 1.5, 2.0, 1e-7, 1e20, 12345678901234567890_u64]),
            r#"["hall", null, true, 1.5, 2.0, 1e-07, 1e+20, 12345678901234567890]"#,
        ),
        (json!(5), "5"),
        (json!(2.0), "2.0"),
        (json!(true), "true"),
        (Json::Null, "null"),
        (
            json!({"nested": {"deep": [1, {"x": []}], "empty": {}}}),
            r#"{"nested": {"deep": [1, {"x": []}], "empty": {}}}"#,
        ),
        // A text is asked about as it is.
        (json!("plain é text"), "plain é text"),
    ];
    for (request, text) in written {
        let ask = Signature::new("Operate a house.")
            .input_typed("request", "dict")
            .input("observation")
            .tool("next", "What next?");
        let model = client();
        let near = Near::new(about()).top(3);
        let rdm = Rdm::new(ask).tools_from(steps()).fit(near.clone()).model(model.clone());
        rdm.call(inputs! { "request" => request }).unwrap();
        assert_eq!(near.answers().keys().next().map(String::as_str), Some(text));
        assert!(texts(&model, "heating").contains(&json!(text)));
    }
}

#[test]
fn where_there_is_nothing_to_compare_the_score_is_nought_and_not_less() {
    // With no dimension yet every text is nowhere, and two of them share nothing: 0, which is not written `-0`.
    let model = client();
    let near = Near::new(blank()).model(model.clone());
    let nowhere = near.embed(["Cooler.", "A lamp."]).unwrap();
    assert_eq!(format!("{:.2}", near.score(&nowhere[0], &nowhere[1])), "0.00");
    let thing = coordinates([("heating", 1.0)]);
    // What the Python module gives for each measure.
    for (measure, score) in [
        ("dot", "0.0"),
        ("cosine", "0.0"),
        ("euclidean", "1.0"),
        ("manhattan", "1.0"),
        ("jaccard", "0.0"),
    ] {
        let measure: Measure = measure.parse().unwrap();
        assert_eq!(format!("{:?}", measure.score(&Coordinates::new(), &thing)), score, "{measure:?}");
        assert_eq!(
            format!(
                "{:?}",
                measure.score(&coordinates([("heating", 0.0)]), &coordinates([("heating", 0.0)]))
            ),
            score
        );
    }
    let request = coordinates([("heating", 0.5), ("lights", 0.25)]);
    let other = coordinates([("doors", 1.0)]);
    assert_eq!(Measure::Dot.score(&request, &other).to_string(), "0");
    assert_eq!(Measure::Euclidean.score(&request, &other), 0.6414298263637128);
    assert_eq!(Measure::Manhattan.score(&request, &other), 0.5714285714285714);
    // A score that was kept with no level likely at all is at the bottom of its rubric.
    let kept = json!({
        "questions": {"urgency": "How soon does `inputs.request` need doing?"},
        "answers": {"Whenever.": {"urgency": {"probabilities": {}, "confidence": 1.0}}},
    });
    let urgent = blank().score(
        "urgency",
        "How soon does `inputs.request` need doing?",
        ["whenever", "this week", "today"],
    );
    let near = Near::new(urgent)
        .with_labels(serde_json::from_value(kept).unwrap())
        .model(model.clone());
    assert_eq!(format!("{:?}", near.embed(["Whenever."]).unwrap()[0]["urgency"]), "0.0");
    assert!(model.requests().is_empty());
}

// Dimensions written while the module runs

#[test]
fn what_will_not_do_as_a_dimension_is_said_as_the_python_module_says_it() {
    let said = "A dimension is a name, written as one word and not `request`, and a question; got";
    let near = Near::new(about());
    let refused = |name: &str, question: &str| match near.add(name, question).unwrap_err() {
        Error::Tool(message) => message,
        other => panic!("{other:?} is not a tool's failure"),
    };
    // What the Python module said of each of these names, which is how Python writes a text: what cannot be seen
    // is written as an escape.
    for (name, written) in [
        ("a\nb", r"'a\nb'"),
        ("tab\tin", r"'tab\tin'"),
        ("nul\0l", r"'nul\x00l'"),
        ("del\u{7f}", r"'del\x7f'"),
        ("nb\u{a0}sp", r"'nb\xa0sp'"),
        ("zw\u{200b}sp", r"'zw\u200bsp'"),
        ("x\u{ad}", r"'x\xad'"),
        ("it's", r#""it's""#),
        ("q\"q", r#"'q"q'"#),
        ("it's \"both\"", r#"'it\'s "both"'"#),
        ("back\\slash", r"'back\\slash'"),
        ("9", "'9'"),
        ("", "''"),
        ("a-b", "'a-b'"),
        ("🙂", "'🙂'"),
    ] {
        assert_eq!(refused(name, "A question?"), format!("{said} {written} and 'A question?'."));
    }
    for (question, written) in [
        ("tab\tin", r"'tab\tin'"),
        ("a\u{1}b", r"'a\x01b'"),
        ("nb\u{a0}sp", r"'nb\xa0sp'"),
        ("é", "'é'"),
        ("it's", r#""it's""#),
        ("line\nbreak", r"'line\nbreak'"),
        ("\u{200b}", r"'\u200b'"),
    ] {
        assert_eq!(refused("request", question), format!("{said} 'request' and {written}."));
    }
    // And what it took: a name is one word in any writing, with whatever is round it left off.
    for (name, kept) in [
        ("été", "été"),
        ("a１", "a１"),
        ("class", "class"),
        ("ａ", "ａ"),
        ("ª", "ª"),
        (" heating\n", "heating"),
        ("sep\u{2028}", "sep"),
        ("fs\u{1c}", "fs"),
    ] {
        let near = Near::new(about());
        near.add(name, "A question?").unwrap();
        assert!(near.questions().contains_key(kept), "{name:?}");
    }
}

#[test]
fn a_dimension_is_a_name_and_a_question_that_can_be_written_while_the_module_runs() {
    // With no dimension yet there is nothing to ask, and a map is offered whole.
    let model = client();
    let near = Near::new(blank()).top(2).model(model.clone());
    let rdm = house(&near, &model);
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert!(near.questions().is_empty());
    assert_eq!(near.embed(["Cooler."]).unwrap(), [Coordinates::new()]);
    assert!(asked(&model, "heating").is_empty());
    assert_eq!(offered(&model)[0].len(), 9);
    assert_eq!(near.show(), "There is no dimension yet.");
    // A dimension is written as a name and a question. The names on the map are asked it when next they are needed.
    assert_eq!(
        near.add("heating", "Is `inputs.request` about how warm it is?").unwrap(),
        "`heating` is now a dimension: Is `inputs.request` about how warm it is?"
    );
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(offered_back(&model, 2), ["cooler", "warmer", "finish"]);
    assert_eq!(asked(&model, "heating").len(), 9);
    // Only a text that lacks an answer is asked. With another dimension, each text is asked once more.
    near.add("lights", " Is `inputs.request` about a light? ").unwrap();
    assert_eq!(
        near.embed([COOLER, "A lamp, and cooler."]).unwrap(),
        [
            coordinates([("heating", 0.9), ("lights", 0.0)]),
            coordinates([("heating", 0.9), ("lights", 0.8)])
        ]
    );
    assert_eq!((asked(&model, "lights").len(), asked(&model, "heating").len()), (2, 9 + 2));
    // The questions are plain data; a name there already is takes the new question, and what was answered to the old one goes.
    assert_eq!(
        json!(near.questions()),
        json!({"heating": "Is `inputs.request` about how warm it is?", "lights": "Is `inputs.request` about a light?"})
    );
    assert_eq!(
        near.show(),
        "heating: Is `inputs.request` about how warm it is?\nlights: Is `inputs.request` about a light?"
    );
    assert_eq!(
        near.add("lights", "Is `inputs.request` about a lamp?").unwrap(),
        "`lights` now asks: Is `inputs.request` about a lamp?"
    );
    assert_eq!(json!(near.answers()["A lamp, and cooler."]), json!({"heating": 0.9}));
    // The same question again changes nothing, and what was answered to it stays.
    near.add("heating", "Is `inputs.request` about how warm it is?").unwrap();
    assert_eq!(json!(near.answers()["A lamp, and cooler."]), json!({"heating": 0.9}));
    // Removing a dimension removes its answers and its threshold with it.
    near.set_params("lights", Params::threshold(0.7));
    assert_eq!(near.remove("lights"), "`lights` is no longer a dimension.");
    assert!(near.all_params().is_empty());
    assert_eq!(near.embed(["A lamp, and cooler."]).unwrap(), [coordinates([("heating", 0.9)])]);
    assert_eq!(
        near.remove("lights"),
        "There is no dimension `lights`: the dimensions are ['heating']."
    );
    // What a writer wrote that will not do is its tool's failure, said as the Python module says it.
    let refused = |name: &str, question: &str| match near.add(name, question).unwrap_err() {
        Error::Tool(message) => message,
        other => panic!("{other:?} is not a tool's failure"),
    };
    let said = "A dimension is a name, written as one word and not `request`, and a question; got";
    assert_eq!(
        refused("two words", "A question?"),
        format!("{said} 'two words' and 'A question?'.")
    );
    assert_eq!(refused("request", "A question?"), format!("{said} 'request' and 'A question?'."));
    assert_eq!(refused("empty", " "), format!("{said} 'empty' and ''."));
    assert_eq!(refused("it's", "say \"x\""), format!("{said} \"it's\" and 'say \"x\"'."));
    // A name a call keeps for itself would stop every request that asked it.
    assert!(refused("state", "A question?").starts_with("A dimension cannot be named `state`"));
    assert_eq!(near.questions().len(), 1);
}

#[test]
fn a_dimension_keeps_the_kind_it_was_built_with_and_one_that_is_written_is_a_yes_or_no() {
    let sorted = blank().noul("heating", "Is `inputs.request` about how warm it is?").score(
        "urgency",
        "How soon does `inputs.request` need doing?",
        ["whenever", "this week", "today"],
    );
    let model = client();
    let near = Near::new(sorted).model(model.clone());
    near.remove("heating");
    near.add("heating", "Is `inputs.request` about how warm it is?").unwrap();
    near.remove("urgency");
    near.add("urgency", "How soon?").unwrap();
    near.add("doors", "Is `inputs.request` about a door or its lock?").unwrap();
    // As the Python module has it: the questions in the order they were written, the coordinates in the order of
    // the signature it was built with, and `urgency` still a score.
    assert!(near.questions().keys().eq(["heating", "urgency", "doors"]));
    assert_eq!(near.columns(), ["heating", "urgency", "doors"]);
    assert_eq!(near.dimensions().output("urgency").unwrap().kind(), Some(Kind::Score));
    assert_eq!(near.dimensions().output("doors").unwrap().kind(), Some(Kind::Noul));
    assert_eq!(
        near.embed(["Cooler, today."]).unwrap(),
        [coordinates([("heating", 0.9), ("urgency", 1.0), ("doors", 0.0)])]
    );
    assert_eq!(near.show().lines().nth(1), Some("urgency: How soon?"));
}

#[test]
fn a_writer_writes_a_dimension_through_a_tool_and_the_decision_model_applies_it() {
    let near = Near::new(about()).top(2);
    // `add_in` takes a name and a question, which is what a language model writes. Here the writer is a function.
    let label = {
        let near = near.clone();
        Tool::new(
            "label",
            "Have a new label made: nothing listed is near what is asked.",
            move |call| {
                let _request: String = call.arg("request")?;
                near.add_in("doors", "Is `inputs.request` about a door or its lock?", call)
            },
        )
    };
    let doors = [
        step("close door", "Close the front door."),
        step("lock door", "Lock the front door."),
    ];
    let model = client_as("jev-test", &["label", "lock door", "finish"]);
    let rdm = Rdm::new(operate())
        .tools_from(steps().into_iter().chain(doors))
        .tool(label)
        .fit(near.clone())
        .always(["label"])
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => "Lock the front door." }).unwrap();
    // No dimension was about a door, so the request shared nothing with any name and the whole map was offered.
    assert_eq!(offered(&model)[0].len(), 12);
    // The decision model pointed at `label`, and the writer wrote a dimension: its name and its question.
    assert_eq!(near.questions()["doors"], "Is `inputs.request` about a door or its lock?");
    // The decision model then said of every name on the map whether the new label applies, and the map was cut by it.
    assert_eq!(offered(&model)[1], ["close door", "lock door", "label", "finish"]);
    let done = result.steps("next");
    assert_eq!(
        done[0].1,
        "`doors` is now a dimension: Is `inputs.request` about a door or its lock?"
    );
    assert_eq!(done[1], ("lock door".to_owned(), json!("did lock door")));
    assert_eq!(result.termination_reason, Reason::Final);
    // The change is in the trajectory, on the request whose tool made it, and `show` is what a writer reads first.
    assert_eq!(
        result.trajectory[0].changes,
        Some(vec![json!({"by": "near", "did": "add", "name": "doors"})])
    );
    assert_eq!(
        serde_json::to_string(&result.trajectory[0].changes).unwrap(),
        r#"[{"by":"near","did":"add","name":"doors"}]"#
    );
    assert_eq!(
        near.show().lines().last(),
        Some("doors: Is `inputs.request` about a door or its lock?")
    );
}

#[test]
fn what_a_writer_gets_wrong_is_its_observation_and_removing_is_recorded_too() {
    let near = Near::new(about()).top(2);
    let label = {
        let near = near.clone();
        Tool::new("label", "Have a new label made.", move |call| {
            near.add_in("two words", "A question?", call)
        })
    };
    let unlabel = {
        let near = near.clone();
        Tool::new(
            "unlabel",
            "Have a label taken away.",
            move |call| Ok(near.remove_in("lights", call)),
        )
    };
    let model = client_as("jev-test", &["label", "unlabel", "unlabel", "finish"]);
    let rdm = Rdm::new(operate()).tool(label).tool(unlabel).fit(near.clone()).model(model);
    let result = rdm.call(inputs! { "request" => "Lock the front door." }).unwrap();
    let said: Vec<Json> = result.steps("next").into_iter().map(|(_, said)| said).collect();
    assert_eq!(
        said,
        [
            json!(
                "Execution error in label: A dimension is a name, written as one word and not `request`, and a question; got 'two \
                 words' and 'A question?'."
            ),
            json!("`lights` is no longer a dimension."),
            json!("There is no dimension `lights`: the dimensions are ['heating']."),
            Json::Null,
        ]
    );
    // What changed nothing is not a change.
    let changes: Vec<_> = result.trajectory.iter().map(|entry| entry.changes.clone()).collect();
    assert_eq!(
        changes,
        [
            None,
            Some(vec![json!({"by": "near", "did": "remove", "name": "lights"})]),
            None,
            None
        ]
    );
}

// Thresholds

#[test]
fn a_label_is_the_probability_until_its_dimension_has_a_threshold_and_a_threshold_asks_nothing_again() {
    let model = client();
    let near = Near::new(about()).model(model.clone());
    let stuffy = ["It is dim and a bit stuffy."];
    assert_eq!(near.embed(stuffy).unwrap(), [coordinates([("heating", 0.6), ("lights", 0.5)])]);
    // With a threshold, an answer under it counts as nothing. It is read when the label is, so nothing is asked again.
    near.set_params("heating", Params::threshold(0.75));
    assert_eq!(near.embed(stuffy).unwrap(), [coordinates([("heating", 0.0), ("lights", 0.5)])]);
    near.set_params("heating", Params::threshold(0.6));
    assert_eq!(near.embed(stuffy).unwrap(), [coordinates([("heating", 0.6), ("lights", 0.5)])]);
    assert_eq!(near.params_of("heating"), Some(Params::threshold(0.6)));
    assert_eq!(near.unset_params("heating"), Some(Params::threshold(0.6)));
    assert_eq!(model.requests().len(), 1);
    assert_eq!(
        json!(near.answers()),
        json!({"It is dim and a bit stuffy.": {"heating": 0.6, "lights": 0.5}})
    );
    // Each text as a row of numbers, in the order of the columns.
    assert_eq!(near.rows(stuffy).unwrap(), [[0.6, 0.5]]);
}

#[test]
fn labels_are_one_models_answers_and_are_dropped_when_another_is_asked() {
    let (first, second) = (client_as("jev-test", &[]), client_as("jev-next", &[]));
    let near = Near::new(about()).model(first.clone());
    near.embed(["Make it cooler."]).unwrap();
    near.embed(["Make it cooler."]).unwrap();
    assert_eq!((first.requests().len(), near.labels().model.as_deref()), (1, Some("jev-test")));
    // Another model would answer otherwise, so what the first answered is not read as its own.
    near.set_model(second.clone());
    near.embed(["Make it cooler."]).unwrap();
    assert_eq!((second.requests().len(), near.labels().model.as_deref()), (1, Some("jev-next")));
    assert!(near.answers().keys().eq(["Make it cooler."]));
    // In a run the model that answers is the module's: the labels are its answers from then on.
    let rdm = house(&near.clone().top(3), &first);
    rdm.call(inputs! { "request" => "Make it cooler." }).unwrap();
    assert_eq!((near.labels().model.as_deref(), near.answers().len()), (Some("jev-test"), 9));
    assert_eq!(texts(&first, "heating").len(), 1 + 9);
}

#[test]
fn a_copy_has_its_own_labels_and_reads_them_under_its_own_thresholds() {
    // "Dim and stuffy" is 0.6 about heating and 0.5 about lights, so the one name offered is a heating step.
    let model = client();
    let near = Near::new(about()).top(1);
    let stuffy = "It is dim and a bit stuffy.";
    let first = |rdm: &Rdm| rdm.call(inputs! { "request" => stuffy }).unwrap().pointed("next")[0].clone();
    assert_eq!(first(&house(&near, &model)), "cooler");
    // A copy with a threshold of its own: heating under it counts as nothing, and a light is offered. It has a
    // copy of every answer, so it asks nothing, and the one it was copied from is as it was.
    let tuned = near.copy();
    tuned.set_params("heating", Params::threshold(0.75));
    assert_eq!(first(&house(&tuned, &model)), "hall light");
    assert_eq!(asked(&model, "heating").len(), 9);
    assert!(near.all_params().is_empty());
    assert_eq!(tuned.embed([stuffy]).unwrap(), [coordinates([("heating", 0.0), ("lights", 0.5)])]);
    assert_eq!(near.embed([stuffy]).unwrap(), [coordinates([("heating", 0.6), ("lights", 0.5)])]);
    // A dimension written into one is not in the other.
    tuned.add("doors", "Is `inputs.request` about a door or its lock?").unwrap();
    assert_eq!((tuned.questions().len(), near.questions().len()), (3, 2));
    // A clone is the same one: what a module that was given it asks is read by the one that was kept.
    assert_eq!(near.clone().remove("lights"), "`lights` is no longer a dimension.");
    assert!(near.questions().keys().eq(["heating"]));
}

#[test]
fn a_copy_of_a_module_writes_its_labels_into_a_near_of_its_own() {
    let model = client();
    let near = Near::new(about()).top(3);
    let rdm = house(&near, &model);
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(near.answers().len(), 9);
    // The copy has every answer the original had, so it asks about its request only, and keeps that itself.
    let twin = rdm.copy();
    twin.call(inputs! { "request" => "It is dim: a lamp, please." }).unwrap();
    assert_eq!((asked(&model, "heating").len(), near.answers().len()), (10, 9));
    rdm.call(inputs! { "request" => "It is dim: a lamp, please." }).unwrap();
    assert_eq!((asked(&model, "heating").len(), near.answers().len()), (11, 10));
}

#[test]
fn a_copy_of_a_module_writes_a_dimension_into_a_near_of_its_own() {
    // Tools that hold the `Near` they were made with: one that writes a dimension, one that takes one away.
    let near = Near::new(about()).top(2);
    let (adding, removing) = (near.clone(), near.clone());
    let label = Tool::new("label", "Have a new label made.", move |call| {
        adding.add_in("doors", "Is `inputs.request` about a door or its lock?", call)
    });
    let unlabel = Tool::new("unlabel", "Have a label taken away.", move |call| {
        Ok(removing.remove_in("lights", call))
    });
    let doors = [("close door", "Close the front door."), ("lock door", "Lock the front door.")];
    let doors = doors.map(|(name, description)| Tool::new(name, description, move |_| Ok(format!("did {name}"))));
    let rdm = Rdm::new(operate())
        .tools_from(steps().into_iter().chain(doors))
        .tool(label)
        .tool(unlabel)
        .fit(near.clone())
        .always(["label", "unlabel"]);

    // A copy of the module has the same tools, and a `Near` of its own. A dimension its tools write
    // in a run of the copy is the copy's: the copy's next request is cut by it.
    let twin = rdm.copy();
    let model = client_as("jev-test", &["label", "unlabel", "lock door", "finish"]);
    let result = twin
        .model(model.clone())
        .call(inputs! { "request" => "Lock the front door." })
        .unwrap();
    let done = result.steps("next");
    assert_eq!(
        done[0].1,
        "`doors` is now a dimension: Is `inputs.request` about a door or its lock?"
    );
    assert_eq!(done[1].1, "`lights` is no longer a dimension.");
    assert_eq!(done[2], ("lock door".to_owned(), json!("did lock door")));
    assert_eq!(offered(&model)[1], ["close door", "lock door", "label", "unlabel", "finish"]);
    assert_eq!(asked(&model, "doors").len(), 1 + 10);
    // The original was not run, and its dimensions are as they were.
    assert!(near.questions().keys().eq(["heating", "lights"]));
    assert!(near.answers().is_empty());
    // Outside a run there is no module to ask: a handle writes into the `Near` it is.
    near.add("doors", "Is `inputs.request` about a door?").unwrap();
    assert!(near.questions().keys().eq(["heating", "lights", "doors"]));
}

// A program that is saved, loaded and copied

#[test]
fn the_labels_are_saved_and_loaded_with_the_program() {
    let program = |signature: Signature| {
        let near = Near::new(signature).top(2);
        (Rdm::new(operate()).tools_from(steps()).fit(near.clone()), near)
    };
    let (rdm, near) = program(blank());
    let model = client();
    near.add("heating", "Is `inputs.request` about how warm it is?").unwrap();
    near.add("doors", "Is `inputs.request` about a door or its lock?").unwrap();
    near.set_params("doors", Params::threshold(0.7));
    rdm.set_model(model.clone());
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    let written = serde_json::to_string(&near.saved()).unwrap();
    assert!(written.starts_with(r#"{"labels":{"questions":{"heating":"Is `inputs.request` about how warm it is?","doors":"#));
    assert!(written.ends_with(r#""model":"jev-test"},"dimensions":{"doors":{"threshold":0.7}},"uses":{}}"#));
    // A program built the same way is given the dimensions that were written, their thresholds, and every answer.
    let (fresh_rdm, fresh) = program(blank());
    let again = client();
    fresh.load(serde_json::from_str::<Saved>(&written).unwrap());
    assert_eq!(fresh.questions(), near.questions());
    assert!(fresh.dimensions().outputs().keys().eq(["heating", "doors"]));
    assert_eq!(fresh.all_params(), IndexMap::from([("doors".to_owned(), Params::threshold(0.7))]));
    assert_eq!(fresh.answers(), near.answers());
    assert_eq!(fresh.saved(), near.saved());
    // So it asks nothing about the map again: the first request is the one that points.
    fresh_rdm.set_model(again.clone());
    fresh_rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(offered(&again)[0], ["cooler", "warmer", "finish"]);
    assert!(asked(&again, "heating").is_empty());
    // A program built with other dimensions has, once loaded, the ones that were saved.
    let (_, other) = program(about());
    other.load(serde_json::from_str::<Saved>(&written).unwrap());
    assert!(other.questions().keys().eq(["heating", "doors"]));
    assert!(other.dimensions().outputs().keys().eq(["heating", "doors"]));
}

#[test]
fn the_labels_are_plain_data_as_the_python_module_keeps_them() {
    // What the Python module kept for a house, read as it wrote it.
    let kept = json!({
        "questions": {"heating": "Is `inputs.request` about how warm it is?", "urgency": "How soon does `inputs.request` need doing?"},
        "answers": {
            "cooler: Make it one degree cooler.": {"heating": 0.9, "urgency": {"probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}, "confidence": 1.0}},
            "warmer: Make it one degree warmer.": {"heating": 0.9, "urgency": 0.2},
        },
        "uses": {},
        "model": "jev-test",
    });
    let labels: Labels = serde_json::from_value(kept.clone()).unwrap();
    assert_eq!(labels.answers["cooler: Make it one degree cooler."]["heating"], Given::Noul(0.9));
    assert_eq!(serde_json::to_string(&labels).unwrap(), serde_json::to_string(&kept).unwrap());
    let sorted = blank()
        .noul("heating", "Is `inputs.request` about how warm it is?")
        .noul("lights", "Is `inputs.request` about a light?")
        .score(
            "urgency",
            "How soon does `inputs.request` need doing?",
            ["whenever", "this week", "today"],
        );
    let model = client();
    let near = Near::new(sorted)
        .params("lights", Params::threshold(0.7))
        .with_labels(labels.clone())
        .model(model.clone());
    assert_eq!(near.labels(), labels);
    // The dimensions are the ones the labels hold, and the threshold of one that is gone went with it.
    assert_eq!(near.columns(), ["heating", "urgency"]);
    assert!(near.all_params().is_empty());
    // What was kept is read without asking. An answer that is not of its dimension's kind is no answer: it is asked again.
    let names = ["cooler: Make it one degree cooler.", "warmer: Make it one degree warmer."];
    assert_eq!(
        near.embed(names).unwrap(),
        [
            coordinates([("heating", 0.9), ("urgency", 1.0)]),
            coordinates([("heating", 0.9), ("urgency", 0.0)])
        ]
    );
    assert_eq!(texts(&model, "urgency"), ["warmer: Make it one degree warmer."]);
    // With nothing in them there is nothing to ask about.
    near.set_labels(Labels::default());
    assert_eq!(near.embed(names).unwrap(), [Coordinates::new(), Coordinates::new()]);
}

#[test]
fn the_coordinates_are_rows_of_numbers_for_a_search_that_takes_vectors() {
    let model = client();
    let near = Near::new(about()).measure(Measure::Cosine).model(model.clone());
    let documents = ["The lamp in the hall is out.", "Make it two degrees cooler.", "Nothing of either."];
    // Each text is a row of numbers, one for each coordinate, in a fixed order.
    assert_eq!(near.columns(), ["heating", "lights"]);
    assert_eq!(near.rows(&documents[..2]).unwrap(), [[0.0, 0.8], [0.9, 0.0]]);
    let corpus = near.rows(documents).unwrap();
    // A search over them, as an index of plain vectors would do it: the cosine of two rows.
    let search = |query: &str| {
        let wanted = &near.rows([query]).unwrap()[0];
        let cosine = |row: &Vec<f64>| {
            let dot = |one: &[f64], other: &[f64]| one.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
            let lengths = dot(wanted, wanted).sqrt() * dot(row, row).sqrt();
            if lengths == 0.0 { 0.0 } else { dot(wanted, row) / lengths }
        };
        let best = (0..corpus.len()).max_by(|one, other| cosine(&corpus[*one]).total_cmp(&cosine(&corpus[*other])));
        documents[best.unwrap()]
    };
    assert_eq!(search("It is stuffy in here."), "Make it two degrees cooler.");
    assert_eq!(search("It is dim."), "The lamp in the hall is out.");
    assert_eq!(model.requests().len(), 5);
}

// With other fits

#[test]
fn several_fits_are_applied_in_order_and_near_leaves_another_fits_questions_alone() {
    // A fit that keeps the first names of a long map, after asking a question of its own about it, as `Fit` does.
    fn shortlist(signature: Signature, inputs: rdm::Inputs, call: &rdm::Call) -> rdm::Result<(Signature, rdm::Inputs)> {
        let Some(name) = signature.pointer().map(str::to_owned) else {
            return Ok((signature, inputs)); // its own question, and Near's: nothing to cut
        };
        let output = signature.output(&name).unwrap();
        let options = output.options().unwrap();
        let (always, measured): (Vec<_>, Vec<_>) = options.iter().cloned().partition(|option| output.stays(option));
        if measured.len() <= 4 {
            return Ok((signature, inputs));
        }
        let own = Signature::new("Shortlist.").input("request").with_output(
            "group",
            Output::choice("Which of these is `inputs.request` about?", measured.iter().take(4)).mark("own", true),
        );
        let request = call.input("request")?;
        call.sub_with(Ask::new(inputs! { "request" => request }).signature(own).answers_only())?;
        let kept = measured.into_iter().take(4).chain(always);
        Ok((signature.with_options(&name, kept), inputs))
    }
    // Near cut the map to what is nearest, and the next fit was given what Near left, which fits.
    let model = client();
    let rdm = Rdm::new(operate())
        .tools_from(steps())
        .fit(Near::new(about()).top(2))
        .fit_fn(shortlist)
        .always(["hall light"])
        .model(model.clone());
    rdm.call(inputs! { "request" => COOLER }).unwrap();
    // Both leave what is always offered.
    assert_eq!(offered(&model)[0], ["cooler", "warmer", "hall light", "finish"]);
    assert!(asked(&model, "group").is_empty());
    // The other way round, the first fit's own question about a long map is left alone by Near, which is given
    // what that fit left: four names, none of them about heating, so nothing sets them apart.
    let model = client();
    let rdm = Rdm::new(operate())
        .tools_from(steps())
        .fit_fn(shortlist)
        .fit(Near::new(about()).top(2))
        .always(["hall light"])
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(asked(&model, "group")[0].questions["group"].labels().len(), 4);
    assert_eq!(
        offered(&model)[0],
        [
            "kitchen light",
            "porch light",
            "attic light",
            "cellar light",
            "hall light",
            "finish"
        ]
    );
    // A request about a light is nearer some of what was left than the rest.
    let result = rdm.call(inputs! { "request" => "A lamp, please." }).unwrap();
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(offered_back(&model, 2), ["kitchen light", "porch light", "hall light", "finish"]);
    assert!(
        offered(&model)
            .iter()
            .all(|options| options.ends_with(&["hall light".to_owned(), "finish".to_owned()]))
    );
}

/// With `Fit` itself, where it is built too: what the Python module asked for the same map, both ways round.
#[cfg(feature = "fit")]
#[test]
fn near_and_fit_are_applied_in_the_order_they_were_given() {
    use rdm::fit::Fit;

    /// Every request the model was asked: each question's name and what it offered.
    fn questions(model: &Scripted) -> Json {
        let of = |request: Request| -> Vec<Json> {
            let each = request.questions.iter();
            each.map(|(name, question)| json!([name, question.labels()])).collect()
        };
        json!(model.requests().into_iter().map(of).collect::<Vec<_>>())
    }
    // Near cut the map to what is nearest, and Fit was given what Near left, which fits. Both leave what is always offered.
    let model = client();
    let rdm = Rdm::new(operate())
        .tools_from(steps())
        .fit(Near::new(about()).top(2))
        .fit(Fit::new().max_options(4))
        .always(["hall light"])
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(offered(&model), [["cooler", "warmer", "hall light", "finish"]; 2]);
    assert_eq!(result.pointed("next"), ["cooler", "finish"]);
    // The request and the seven names that are measured, each asked once, and the two turns: nothing of Fit's.
    assert_eq!((asked(&model, "heating").len(), model.requests().len()), (8, 10));
    // The other way round, Fit's own questions about a long map are left alone by Near, and the run still ends.
    // Fit left two names, which is no more than Near offers, so Near asked nothing.
    let model = client();
    let rdm = Rdm::new(operate())
        .tools_from(steps())
        .fit(Fit::new().max_options(4))
        .fit(Near::new(about()).top(2))
        .always(["hall light"])
        .model(model.clone());
    let result = rdm.call(inputs! { "request" => COOLER }).unwrap();
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(result.pointed("next"), ["kitchen light", "finish"]);
    let turn = json!([
        [
            ["next_0", ["kitchen light", "porch light", "hall light", "finish"]],
            ["next_1", ["attic light", "cellar light", "hall light", "finish"]],
            ["next_2", ["study light", "cooler", "hall light", "finish"]],
        ],
        [
            ["next_0", ["kitchen light", "attic light", "hall light", "finish"]],
            ["next_1", ["study light", "warmer", "hall light", "finish"]],
        ],
        [["next", ["kitchen light", "study light", "hall light", "finish"]]],
    ]);
    let both: Vec<Json> = turn.as_array().unwrap().iter().chain(turn.as_array().unwrap()).cloned().collect();
    assert_eq!(questions(&model), json!(both));
}

#[test]
fn a_tool_embeds_inside_a_run_and_what_it_asks_are_sub_calls_of_its_request() {
    // A search by named dimensions as a tool: the documents nearest the request, asked about in the run.
    let documents = ["The lamp in the hall is out.", "Make it two degrees cooler.", "Nothing of either."];
    let near = Near::new(about()).measure(Measure::Cosine);
    let search = {
        let near = near.clone();
        Tool::new("search", "Find the note nearest the request.", move |call| {
            let request: String = call.arg("request")?;
            let embedded = near.embed_in(std::iter::once(request.as_str()).chain(documents), call)?;
            let scores = embedded[1..].iter().map(|document| near.score(&embedded[0], document));
            let best = scores.enumerate().max_by(|one, other| one.1.total_cmp(&other.1));
            Ok(best.map(|(position, _)| documents[position]))
        })
    };
    let ask = blank().noul("search", "Does `inputs.request` ask for something to be found?");
    let model = Arc::new(Scripted::new(|request: &Request| {
        let text = request.state["inputs"]["request"].as_str().unwrap_or_default().to_lowercase();
        let answer = |name: &String| {
            (
                name.clone(),
                Answer::noul(if name == "search" { 0.9 } else { likely(&WORDS, name, &text) }),
            )
        };
        request.questions.keys().map(answer).collect::<Vec<_>>()
    }));
    let rdm = Rdm::new(ask).tool(search).model(model.clone());
    let result = rdm.call(inputs! { "request" => "It is stuffy in here." }).unwrap();
    assert_eq!(result.observations["search"], "Make it two degrees cooler.");
    // One request for each text, one level down, under the tool that asked, for the answers only.
    let labelled = entries(&result.trajectory, "heating");
    assert_eq!(labelled.len(), 4);
    for entry in labelled {
        assert_eq!(
            (entry.depth, entry.parent, entry.tool_name.as_deref()),
            (1, Some(0), Some("search"))
        );
    }
    assert_eq!((result.trajectory.len(), result.termination_reason), (5, Reason::Final));
    // Under a limit, a text that could not be asked about is nowhere and is not kept, and the run says a limit
    // was reached. The documents are kept already, so the request was the one text to ask about.
    let rdm = Rdm::new(rdm.signature().clone())
        .tool(rdm.tools().get("search").unwrap())
        .model(model)
        .max_predictor_calls(1);
    let result = rdm.call(inputs! { "request" => "It is dim." }).unwrap();
    assert_eq!(result.termination_reason, Reason::MaxPredictorCalls);
    assert_eq!((result.trajectory.len(), near.answers().len()), (2, 4));
    assert!(!near.answers().contains_key("It is dim."));
}

#[test]
fn runs_at_once_share_the_labels() {
    let model = client();
    let near = Near::new(about()).top(3);
    let rdm = house(&near, &model);
    let requests = [
        COOLER,
        "It is dim: a lamp, please.",
        "Warmer, by two degrees.",
        "Do whatever is best.",
    ];
    let results = rdm.batch(requests.map(|request| inputs! { "request" => request }));
    assert!(
        results
            .iter()
            .all(|result| result.as_ref().unwrap().termination_reason == Reason::Final)
    );
    // Every text has its answers, whichever run asked: the eight names and the four requests.
    assert_eq!(near.answers().len(), 8 + 4);
    let pointed: Vec<String> = results
        .iter()
        .map(|result| result.as_ref().unwrap().pointed("next")[0].clone())
        .collect();
    assert_eq!(pointed, ["cooler", "hall light", "cooler", "hall light"]);
    // From then on a run asks nothing about the map.
    let before = asked(&model, "heating").len();
    rdm.call(inputs! { "request" => "Cooler." }).unwrap();
    assert_eq!(asked(&model, "heating").len(), before + 1);
}

// The two notebook cells, beside what the Python module asked and printed for them

/// A stand-in for the decision model a notebook cell runs: a dimension is as likely as the words of the text
/// make it, and `point` says where the pointer points, given the inputs and what is offered.
fn stand_in(words: &'static [(&'static str, &'static [(&'static str, f64)])], point: fn(&Json, &[String]) -> String) -> Arc<Scripted> {
    let answers = move |request: &Request| {
        let inputs = &request.state["inputs"];
        let answer = |(name, question): (&String, &Question)| {
            let answer = match question.kind {
                Kind::Noul => {
                    let about = inputs.as_object().and_then(|inputs| inputs.values().next());
                    let about = about.and_then(Json::as_str).unwrap_or_default().to_lowercase();
                    Answer::noul(likely(words, name, &about))
                }
                _ => {
                    let labels = question.labels();
                    Answer::chose(&point(inputs, &labels), labels.iter().map(String::as_str))
                }
            };
            (name.clone(), answer)
        };
        request.questions.iter().map(answer).collect::<Vec<_>>()
    };
    Arc::new(Scripted::new(answers).named("jev-test"))
}

/// Requests as they are written on the wire, in an order that does not depend on which was asked first: the
/// sub-calls of one batch are asked at once.
fn wire(requests: impl IntoIterator<Item = Json>) -> Vec<String> {
    let mut written: Vec<String> = requests.into_iter().map(|request| request.to_string()).collect();
    written.sort();
    written
}

fn captured(lines: &str) -> Vec<Json> {
    lines.lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}

/// What Python printed, with a list or a dict written as Rust writes one.
fn as_rust_prints(printed: &str) -> String {
    printed.replace('\'', "\"")
}

#[test]
fn the_near_cell_a_map_with_labels() {
    const HOUSE_WORDS: [(&str, &[(&str, f64)]); 3] = [
        ("screen", &[("tv", 0.95)]),
        ("light", &[("lights", 0.95)]),
        ("entry", &[("door", 0.9), ("lock", 0.8)]),
    ];
    // The decision model the page is running: here one that closes the door before it locks it.
    let decision_model = stand_in(&HOUSE_WORDS, |inputs, offered| {
        let (house, has) = (&inputs["house"], |name: &str| offered.iter().any(|label| label == name));
        if has("door closed") && house["door"] == "open" {
            "door closed".to_owned()
        } else if has("lock locked") && house["lock"] == "unlocked" {
            "lock locked".to_owned()
        } else {
            "finish".to_owned()
        }
    });

    // What the form sets: the request, and where the house starts.
    let task = "Lock the front door.";
    let house = Shared::new(IndexMap::from([
        ("tv", "off"),
        ("lights", "on"),
        ("door", "open"),
        ("lock", "unlocked"),
    ]));
    // What `show_house` draws: the house on the right, and what was last done.
    let drawn = Shared::new(Vec::new());

    // Set one device to one state. The house pushes back: an open door cannot be locked.
    let switch = |device: &'static str, to: &'static str| {
        let (house, drawn) = (house.clone(), drawn.clone());
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
            let done = house.update(|house| {
                if device == "lock" && to == "locked" && house["door"] == "open" {
                    return "lock: refused, the door is open".to_owned();
                }
                house.insert(device, to);
                format!("{device}: now {to}")
            });
            drawn.update(|drawn| drawn.push(done.clone())); // redraws the house on the right
            Ok(done)
        })
    };

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let steps = [
        ("tv", "on"),
        ("tv", "off"),
        ("lights", "on"),
        ("lights", "off"),
        ("door", "closed"),
        ("lock", "locked"),
    ];
    let steps = steps.map(|(device, to)| switch(device, to));

    let about = Signature::new("Say what a text is about.")
        .input("task")
        .noul("screen", "Is `inputs.task` about the TV?")
        .noul("light", "Is `inputs.task` about the lights?")
        .noul("entry", "Is `inputs.task` about the door or its lock?");

    let near = Near::new(about).top(2); // of the six steps, the two nearest a request are offered, with `finish`
    let guided = Rdm::new(operate).tools_from(steps).fit(near.clone()).model(decision_model.clone());

    drawn.update(|drawn| drawn.push(task.to_owned()));
    let result = guided.call(inputs! { "task" => task, "house" => &house }).unwrap();
    let printed = [
        format!("pointed at: {:?}", result.pointed("next")),
        format!("{} {}", house.get()["door"], house.get()["lock"]),
        format!(
            "{} texts were given labels, each asked once: the request and the six names",
            near.answers().len()
        ),
    ];

    assert_eq!(printed.join("\n"), as_rust_prints(NEAR_PRINTED));
    assert_eq!(drawn.get(), [task, "door: now closed", "lock: now locked"]);
    // The model was asked what the Python module asked it, request for request, and nothing more.
    let asked = decision_model.requests().into_iter().map(|request| json!(request));
    assert_eq!(wire(asked), wire(captured(NEAR_REQUESTS)));
    // The trajectory showed what Python's did, in its order: the request, what the fit asked under it, the turns after.
    let shown: Vec<&Json> = result.trajectory.iter().map(|entry| &entry.inputs).collect();
    assert_eq!(
        json!(shown).to_string(),
        serde_json::from_str::<Json>(NEAR_SHOWN).unwrap().to_string()
    );
    let places: Vec<_> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.iteration, entry.parent))
        .collect();
    assert_eq!(places[..2], [(0, 0, None), (1, 0, Some(0))]);
    assert_eq!(places[8..], [(0, 1, Some(0)), (0, 2, Some(8))]);
    // The labels are the ones it kept, key for key.
    assert_eq!(
        serde_json::to_string(&near.labels()).unwrap(),
        serde_json::from_str::<Json>(NEAR_LABELS).unwrap().to_string()
    );
}

#[test]
fn the_dimensions_cell_embeddings_you_can_read() {
    const DESK_WORDS: [(&str, &[(&str, f64)]); 4] = [
        ("money", &[("charge", 0.95), ("invoice", 0.9), ("receipt", 0.9), ("prices", 0.4)]),
        ("access", &[("password", 0.95), ("login", 0.9), ("locked out", 0.9)]),
        ("fault", &[("fault", 0.95), ("outage", 0.8), ("bug", 0.9), ("fix", 0.3)]),
        ("buying", &[("prices", 0.9), ("plans", 0.9), ("sales", 0.85)]),
    ];
    // The decision model the page is running: here one that refunds, where it can, and then stops.
    let decision_model = stand_in(&DESK_WORDS, |inputs, offered| {
        let refund = offered.iter().find(|label| *label == "refund").unwrap_or(&offered[0]);
        if inputs["observation"] == "" {
            refund.clone()
        } else {
            "finish".to_owned()
        }
    });

    // What the form sets: the email.
    let email = "I got charged twice this month, pls fix asap";

    let handle = Signature::new("Handle an email to a support desk, one step at a time.")
        .input("email")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should be done next about `inputs.email`?");

    // One name on the map: a description for the model, and a thing for the code to do.
    let step = |name: &'static str, description: &'static str| Tool::new(name, description, move |_| Ok(format!("{name}: done")));

    // The map: eight things a support desk can do.
    let steps = [
        step("refund", "Return a charge to the customer's card."),
        step("send invoice", "Send a copy of an invoice or a receipt."),
        step("reset password", "Send a link to reset the password of an account."),
        step("unlock account", "Unlock an account that is locked out after failed logins."),
        step("file a bug", "Report a fault in the app to the engineers."),
        step("check status page", "Look up whether there is an outage right now."),
        step("quote a plan", "Send the prices of the plans for a team."),
        step("book a demo", "Offer a call with sales to see the product."),
    ];

    // The dimensions: each output is one, a name and a question. The input is named after the
    // request's input that says what is wanted, and a name on the map is asked as "name: description".
    let about = Signature::new("Say what a text is about.")
        .input("email")
        .noul("money", "Is `inputs.email` about a charge, a payment or an invoice?")
        .noul("access", "Is `inputs.email` about logging in or a password?")
        .noul("fault", "Is `inputs.email` about something in the app that is broken?")
        .noul("buying", "Is `inputs.email` about prices, plans or buying?");

    // Of the eight names, a request is offered the three nearest it, with `finish`.
    let near = Near::new(about).top(3).measure(Measure::Cosine).model(decision_model.clone());
    let desk = Rdm::new(handle)
        .tools_from(steps.clone())
        .fit(near.clone())
        .model(decision_model.clone());

    let result = desk.call(inputs! { "email" => email }).unwrap();
    let mut printed = vec![format!("{email} -> {:?}", result.pointed("next")), String::new()];

    // The embedding, which is part of the program: what each text answered, by the name of the dimension.
    for (text, answered) in near.answers() {
        let labels: Vec<String> = answered.iter().map(|(name, value)| format!("{name} {value:.2}")).collect();
        let name: String = text.split(':').next().unwrap().chars().take(44).collect();
        printed.push(format!("{} | {name}", labels.join(" ")));
    }
    printed.push(String::new());

    // Used alone it is a search by named dimensions: the vector of what is wanted is written by hand, and can be read.
    let wanted = coordinates([("money", 1.0), ("access", 0.0), ("fault", 0.0), ("buying", 0.5)]);
    let names: Vec<String> = steps.iter().map(|tool| text(tool.name(), tool.description())).collect();
    let mut ranked: Vec<(&String, Coordinates)> = names.iter().zip(near.embed(&names).unwrap()).collect();
    ranked.sort_by(|one, other| near.score(&wanted, &other.1).total_cmp(&near.score(&wanted, &one.1)));
    printed.push(format!("nearest to {wanted:?}"));
    for (name, vector) in &ranked[..3] {
        printed.push(format!("  {:.2} {}", near.score(&wanted, vector), name.split(':').next().unwrap()));
    }

    assert_eq!(printed.join("\n"), as_rust_prints(DESK_PRINTED));
    // The model was asked what the Python module asked it, request for request: the search by hand asked nothing,
    // since every name had been asked when the map was first needed.
    let asked = decision_model.requests().into_iter().map(|request| json!(request));
    assert_eq!(wire(asked), wire(captured(DESK_REQUESTS)));
    assert_eq!(
        serde_json::to_string(&near.labels()).unwrap(),
        serde_json::from_str::<Json>(DESK_LABELS).unwrap().to_string()
    );
    // Every number has a name, so it can be read why a name was offered: the email and `refund` are both about money.
    let offered = &decision_model
        .requests()
        .into_iter()
        .find(|request| request.questions.contains_key("next"))
        .unwrap()
        .questions["next"];
    assert_eq!(offered.labels(), ["refund", "send invoice", "quote a plan", "finish"]);
    assert_eq!(near.answers()[email]["money"], near.answers()[&names[0]]["money"]);
}

// What the Python module printed and asked for the two cells, run on the same stand-in model (its requests in
// the order it happened to ask them), and the labels it kept.

const NEAR_PRINTED: &str = r##"pointed at: ['door closed', 'lock locked', 'finish']
closed locked
7 texts were given labels, each asked once: the request and the six names"##;

const NEAR_REQUESTS: &str = r##"{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "Lock the front door."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "tv on: Set the tv to on."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "tv off: Set the tv to off."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "lights on: Set the lights to on."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "door closed: Set the door to closed."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "lights off: Set the lights to off."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `task` (str):", "inputs": {"task": "lock locked: Set the lock to locked."}}, "questions": {"screen": {"instructions": "Is `inputs.task` about the TV?", "type": "noul"}, "light": {"instructions": "Is `inputs.task` about the lights?", "type": "noul"}, "entry": {"instructions": "Is `inputs.task` about the door or its lock?", "type": "noul"}}}
{"state": {"instructions": "Operate a smart home, one step at a time.", "input_fields": "1. `task` (str): \n2. `house` (dict): \n3. `observation` (str): What the last step did.", "inputs": {"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}, "observation": ""}}, "questions": {"next": {"instructions": "What should happen next for `inputs.task`?", "type": "choice", "criteria": {"door closed": "Set the door to closed.", "lock locked": "Set the lock to locked.", "finish": "Stop: nothing more should be done."}}}}
{"state": {"instructions": "Operate a smart home, one step at a time.", "input_fields": "1. `task` (str): \n2. `house` (dict): \n3. `observation` (str): What the last step did.", "inputs": {"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}, "observation": "door: now closed"}}, "questions": {"next": {"instructions": "What should happen next for `inputs.task`?", "type": "choice", "criteria": {"door closed": "Set the door to closed.", "lock locked": "Set the lock to locked.", "finish": "Stop: nothing more should be done."}}}}
{"state": {"instructions": "Operate a smart home, one step at a time.", "input_fields": "1. `task` (str): \n2. `house` (dict): \n3. `observation` (str): What the last step did.", "inputs": {"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "closed", "lock": "locked"}, "observation": "lock: now locked"}}, "questions": {"next": {"instructions": "What should happen next for `inputs.task`?", "type": "choice", "criteria": {"door closed": "Set the door to closed.", "lock locked": "Set the lock to locked.", "finish": "Stop: nothing more should be done."}}}}"##;

const NEAR_SHOWN: &str = r##"[{"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}, "observation": ""}, {"task": "Lock the front door."}, {"task": "tv on: Set the tv to on."}, {"task": "tv off: Set the tv to off."}, {"task": "lights on: Set the lights to on."}, {"task": "lights off: Set the lights to off."}, {"task": "door closed: Set the door to closed."}, {"task": "lock locked: Set the lock to locked."}, {"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}, "observation": "door: now closed"}, {"task": "Lock the front door.", "house": {"tv": "off", "lights": "on", "door": "closed", "lock": "locked"}, "observation": "lock: now locked"}]"##;

const NEAR_LABELS: &str = r##"{"questions": {"screen": "Is `inputs.task` about the TV?", "light": "Is `inputs.task` about the lights?", "entry": "Is `inputs.task` about the door or its lock?"}, "answers": {"Lock the front door.": {"screen": 0.0, "light": 0.0, "entry": 0.9}, "tv on: Set the tv to on.": {"screen": 0.95, "light": 0.0, "entry": 0.0}, "tv off: Set the tv to off.": {"screen": 0.95, "light": 0.0, "entry": 0.0}, "lights on: Set the lights to on.": {"screen": 0.0, "light": 0.95, "entry": 0.0}, "lights off: Set the lights to off.": {"screen": 0.0, "light": 0.95, "entry": 0.0}, "door closed: Set the door to closed.": {"screen": 0.0, "light": 0.0, "entry": 0.9}, "lock locked: Set the lock to locked.": {"screen": 0.0, "light": 0.0, "entry": 0.8}}, "uses": {}, "model": "jev-test"}"##;

const DESK_PRINTED: &str = r##"I got charged twice this month, pls fix asap -> ['refund', 'finish']

money 0.95 access 0.00 fault 0.30 buying 0.00 | I got charged twice this month, pls fix asap
money 0.95 access 0.00 fault 0.00 buying 0.00 | refund
money 0.90 access 0.00 fault 0.00 buying 0.00 | send invoice
money 0.00 access 0.95 fault 0.00 buying 0.00 | reset password
money 0.00 access 0.90 fault 0.00 buying 0.00 | unlock account
money 0.00 access 0.00 fault 0.95 buying 0.00 | file a bug
money 0.00 access 0.00 fault 0.80 buying 0.00 | check status page
money 0.40 access 0.00 fault 0.00 buying 0.90 | quote a plan
money 0.00 access 0.00 fault 0.00 buying 0.85 | book a demo

nearest to {'money': 1.0, 'access': 0.0, 'fault': 0.0, 'buying': 0.5}
  0.89 refund
  0.89 send invoice
  0.77 quote a plan"##;

const DESK_REQUESTS: &str = r##"{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "I got charged twice this month, pls fix asap"}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "refund: Return a charge to the customer's card."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "send invoice: Send a copy of an invoice or a receipt."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "reset password: Send a link to reset the password of an account."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "file a bug: Report a fault in the app to the engineers."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "unlock account: Unlock an account that is locked out after failed logins."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "quote a plan: Send the prices of the plans for a team."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "check status page: Look up whether there is an outage right now."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Say what a text is about.", "input_fields": "1. `email` (str):", "inputs": {"email": "book a demo: Offer a call with sales to see the product."}}, "questions": {"money": {"instructions": "Is `inputs.email` about a charge, a payment or an invoice?", "type": "noul"}, "access": {"instructions": "Is `inputs.email` about logging in or a password?", "type": "noul"}, "fault": {"instructions": "Is `inputs.email` about something in the app that is broken?", "type": "noul"}, "buying": {"instructions": "Is `inputs.email` about prices, plans or buying?", "type": "noul"}}}
{"state": {"instructions": "Handle an email to a support desk, one step at a time.", "input_fields": "1. `email` (str): \n2. `observation` (str): What the last step did.", "inputs": {"email": "I got charged twice this month, pls fix asap", "observation": ""}}, "questions": {"next": {"instructions": "What should be done next about `inputs.email`?", "type": "choice", "criteria": {"refund": "Return a charge to the customer's card.", "send invoice": "Send a copy of an invoice or a receipt.", "quote a plan": "Send the prices of the plans for a team.", "finish": "Stop: nothing more should be done."}}}}
{"state": {"instructions": "Handle an email to a support desk, one step at a time.", "input_fields": "1. `email` (str): \n2. `observation` (str): What the last step did.", "inputs": {"email": "I got charged twice this month, pls fix asap", "observation": "refund: done"}}, "questions": {"next": {"instructions": "What should be done next about `inputs.email`?", "type": "choice", "criteria": {"refund": "Return a charge to the customer's card.", "send invoice": "Send a copy of an invoice or a receipt.", "quote a plan": "Send the prices of the plans for a team.", "finish": "Stop: nothing more should be done."}}}}"##;

const DESK_LABELS: &str = r##"{"questions": {"money": "Is `inputs.email` about a charge, a payment or an invoice?", "access": "Is `inputs.email` about logging in or a password?", "fault": "Is `inputs.email` about something in the app that is broken?", "buying": "Is `inputs.email` about prices, plans or buying?"}, "answers": {"I got charged twice this month, pls fix asap": {"money": 0.95, "access": 0.0, "fault": 0.3, "buying": 0.0}, "refund: Return a charge to the customer's card.": {"money": 0.95, "access": 0.0, "fault": 0.0, "buying": 0.0}, "send invoice: Send a copy of an invoice or a receipt.": {"money": 0.9, "access": 0.0, "fault": 0.0, "buying": 0.0}, "reset password: Send a link to reset the password of an account.": {"money": 0.0, "access": 0.95, "fault": 0.0, "buying": 0.0}, "unlock account: Unlock an account that is locked out after failed logins.": {"money": 0.0, "access": 0.9, "fault": 0.0, "buying": 0.0}, "file a bug: Report a fault in the app to the engineers.": {"money": 0.0, "access": 0.0, "fault": 0.95, "buying": 0.0}, "check status page: Look up whether there is an outage right now.": {"money": 0.0, "access": 0.0, "fault": 0.8, "buying": 0.0}, "quote a plan: Send the prices of the plans for a team.": {"money": 0.4, "access": 0.0, "fault": 0.0, "buying": 0.9}, "book a demo: Offer a call with sales to see the product.": {"money": 0.0, "access": 0.0, "fault": 0.0, "buying": 0.85}}, "uses": {}, "model": "jev-test"}"##;
