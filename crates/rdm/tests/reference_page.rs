#![cfg(all(feature = "filter", feature = "fit", feature = "near", feature = "lm", feature = "repl"))]
//! The examples of the `dspy.experimental.RDM` reference page, each as the Rust it is here.
//!
//! The page (`docs/docs/api/experimental/RDM.md` of the DSPy fork) is one program in fifteen
//! blocks. Every `print` on it is followed by the lines it prints, as comments, and the page is
//! only right if the program prints them: `tests/predict/test_rdm_docs.py` runs the blocks top
//! to bottom with stand-ins for the decision model and the generative model. This file does the
//! same for the Rust. Each block is a function that reads beside its Python, each has a test
//! of its own, in the order of the page, and `the_page_prints_what_it_says` runs them as the
//! one program they are and sets what was printed beside the fourteen lines the page documents.
//!
//! What a test asserts beyond the printed line is what the Python module did for the same
//! block with the same stand-ins: the reason the run ended, the shape of its trajectory, and
//! every request a stand-in was put. The requests are in `reference_page/requests.jsonl`,
//! written down by `reference_page/python/page.py` from a run of the Python module, and a test
//! sets the body of each request the Rust sends beside the one the Python sent.
//!
//! Where the Rust is deliberately another way, the block is written the Rust way and the
//! assertion is on the outcome:
//!
//! - an input a tool changes is live: the house is a [`Shared`], given as `&house`;
//! - a tool is given the call and reads its arguments from it, `call.arg("device")`, where the
//!   Python fills parameters by name;
//! - a sub-call is made through the call, `call.sub(..)`, `call.sub_with(..)`, `call.batch(..)`,
//!   where the Python calls `call.rdm`;
//! - a language model completes a signature through `lm::Generate`, where the page has
//!   `dspy.Predict` and `dspy.RLM`;
//! - a threshold is a parameter the module holds by output name, `set_params`, where the page
//!   sets it on a predictor.
//!
//! Two blocks do not come out as the Python's do, and their tests say where: the twelfth
//! (`Fit` cuts the text into forty parts where the Python cuts thirty-five) and the ninth (what
//! the request after `write` shows as its observation).

#[allow(dead_code, unused_macros, unused_imports)]
#[path = "../examples/common/py.rs"]
mod py;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use rdm::filter::Filter;
use rdm::fit::{self, Fit, GAP, NEEDED};
use rdm::lm::{Generate, Lm, ScriptedLm, input_of};
use rdm::near::Near;
use rdm::repl::Repl;
use rdm::systemone::{self, Answer, Kind, Question, Request, Response};
use rdm::{
    Ask, DecisionModel, Entry, Json, Output, Params, Prediction, RESERVED, Rdm, Reason, Scripted, Shared, Signature, Source, Tool,
    as_observation, inputs,
};
use serde_json::{Map, json};

// ---------------------------------------------------------------------------------------------
// The stand-ins, as `tests/predict/test_rdm_docs.py` has them.
// ---------------------------------------------------------------------------------------------

/// What the stand-in's rule gives for one question: a truth, or a text.
enum Said {
    Truth(bool),
    Text(String),
}

impl Said {
    /// Python's `0.9 if answer else 0.1`: a text is a yes unless it is empty.
    fn yes(&self) -> bool {
        match self {
            Said::Truth(truth) => *truth,
            Said::Text(text) => !text.is_empty(),
        }
    }

    /// Python's `str(answer)`, which a choice's labels are set beside.
    fn text(&self) -> &str {
        match self {
            Said::Truth(true) => "True",
            Said::Truth(false) => "False",
            Said::Text(text) => text,
        }
    }
}

/// The steps a stand-in points at for each task, in turn.
type Plans = Mutex<HashMap<&'static str, VecDeque<&'static str>>>;

const LOCK_UP: [&str; 4] = ["lock locked", "door closed", "lock locked", "finish"];

/// `Jev`: a stand-in System One client, answering each question by a rule written for the
/// page's examples. A yes is 0.9 and a no 0.1, and a choice is the option chosen, as certain.
/// Each has its own steps to take, so a task one module was put can be put to another, and
/// each keeps the requests it was put.
#[derive(Clone)]
struct Jev {
    asked: Arc<Scripted>,
    plans: Arc<Plans>,
}

impl Jev {
    fn new() -> Self {
        let plans: Arc<Plans> = Arc::new(Mutex::new(HashMap::from([
            ("Lock the front door.", VecDeque::from(LOCK_UP)),
            (
                "Good night: everything off, and lock up.",
                VecDeque::from(["tv off", "lights off", "lock locked", "door closed", "lock locked", "finish"]),
            ),
            ("Night!", VecDeque::from(["good night", "finish"])),
            (
                "Bring the heat down to 18 degrees.",
                VecDeque::from(["write", "cool_to_18", "finish"]),
            ),
        ])));
        let planned = plans.clone();
        let answers = move |request: &Request| {
            let mut answers = IndexMap::new();
            for (name, question) in &request.questions {
                let said = Jev::answer(&planned, &request.state["inputs"], name, question)?;
                let evidence = match question.kind {
                    Kind::Noul => Answer::noul(if said.yes() { 0.9 } else { 0.1 }),
                    _ => Answer::chose(said.text(), question.labels().iter().map(String::as_str)),
                };
                answers.insert(name.clone(), evidence);
            }
            Ok(answers)
        };
        Jev {
            asked: Arc::new(Scripted::fallible(answers).named("jev-stand-in")),
            plans,
        }
    }

    /// `Jev.answer`.
    fn answer(plans: &Plans, inputs: &Json, name: &str, question: &Question) -> Result<Said, systemone::Error> {
        let none = Map::new();
        let inputs = inputs.as_object().unwrap_or(&none);
        let shown: Map<String, Json> = inputs
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "question" | "questions"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let shown = as_observation(&Json::Object(shown)); // json.dumps
        let request = inputs.get("task").and_then(Json::as_str).unwrap_or("");
        if name == "next" {
            let next = plans.lock().unwrap().get_mut(request).and_then(VecDeque::pop_front);
            let none_left = || systemone::Error::Evidence(format!("The stand-in has no step left to take for {request:?}."));
            return next.map(|next| Said::Text(next.to_owned())).ok_or_else(none_left);
        }
        if name == "category" {
            let known = ["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"];
            let option = question.labels().into_iter().find(|option| known.contains(&option.as_str()));
            let none_known = || systemone::Error::Evidence("The stand-in knows none of the categories on offer.".to_owned());
            return option.map(Said::Text).ok_or_else(none_known);
        }
        let lower = shown.to_lowercase();
        Ok(match name {
            "screen" => Said::Truth(lower.contains("tv")),
            "light" => Said::Truth(lower.contains("light")),
            "entry" => Said::Truth(lower.contains("door") || lower.contains("lock")),
            "switch" => Said::Truth(true),
            "device" => Said::Text("tv".to_owned()),
            "to" => Said::Text(if request.contains("off") { "off" } else { "on" }.to_owned()),
            "each" => Said::Truth(true),
            "split" => Said::Truth(request.contains(", and ")),
            "act" => Said::Truth(request.starts_with("Turn")),
            "chat" => Said::Truth(request.contains("rain") && !request.contains(", and ")),
            _ => Said::Truth(shown.contains("preferably only one")),
        })
    }

    fn requests(&self) -> Vec<Request> {
        self.asked.requests()
    }

    /// The requests it was put, each as the body a decision model is sent.
    fn sent(&self) -> Vec<String> {
        let body = |request: &Request| json!({"state": request.state, "questions": request.questions}).to_string();
        self.requests().iter().map(body).collect()
    }

    /// Forget the requests it was put so far.
    fn forget(&self) {
        self.asked.clear();
    }

    /// Give it the steps of a task to take once more.
    fn plan(&self, task: &'static str, steps: &[&'static str]) {
        self.plans.lock().unwrap().insert(task, steps.iter().copied().collect());
    }
}

impl DecisionModel for Jev {
    fn decide(&self, request: &Request) -> Result<Response, systemone::Error> {
        self.asked.decide(request)
    }

    fn name(&self) -> String {
        self.asked.name()
    }
}

const COOL_TO_18: &str = "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return f'it is {heat()} degrees'";

/// `llm`: a stand-in generative model for the page's generative examples, answering by the
/// task it is shown. The Python one answers a `dspy.Predict` and one step of a `dspy.RLM`;
/// this one answers a `Generate`, which asks for its outputs as one JSON object.
#[derive(Clone)]
struct Llm {
    lm: Lm,
    asked: Arc<ScriptedLm>,
}

fn llm() -> Llm {
    let asked = Arc::new(ScriptedLm::new(|_system, prompt| {
        let written = match input_of(prompt, "task").unwrap_or_default() {
            "Turn the TV off, and tell me whether it will rain." => json!({"parts": ["Turn the TV off.", "Will it rain today?"]}),
            "Will it rain today?" => json!({"reply": "I cannot see the weather from here."}),
            "Bring the heat down to 18 degrees." => {
                json!({"name": "cool_to_18", "description": "Cool the house until it is 18 degrees.", "code": COOL_TO_18})
            }
            _ => json!({}),
        };
        written.to_string()
    }));
    Llm {
        lm: Lm::shared(asked.clone(), "lm-stand-in"),
        asked,
    }
}

/// What the Python module sent a stand-in in one block of the page, each request as the body
/// it is: `reference_page/python/page.py` wrote them down. Block 16 is the four modules of the
/// last block called once more, after their thresholds were set.
fn python(block: usize, module: &str) -> Vec<String> {
    let lines = include_str!("reference_page/requests.jsonl").lines();
    let lines = lines.map(|line| serde_json::from_str::<Json>(line).expect("a line of requests.jsonl is JSON"));
    lines
        .filter(|line| line["block"] == json!(block) && line["module"] == module)
        .map(|line| json!({"state": line["state"], "questions": line["questions"]}).to_string())
        .collect()
}

/// Requests in an order of their own, for those that were asked at once.
fn sorted(mut requests: Vec<String>) -> Vec<String> {
    requests.sort();
    requests
}

/// Of the bodies of some requests, those that ask one question.
fn asking(requests: Vec<String>, name: &str) -> Vec<String> {
    let asks = |request: &String| serde_json::from_str::<Json>(request).is_ok_and(|request| request["questions"].get(name).is_some());
    requests.into_iter().filter(asks).collect()
}

// ---------------------------------------------------------------------------------------------
// The page: what its blocks share, and what they print.
// ---------------------------------------------------------------------------------------------

/// The house is one dict on the page, changed by the tools of every block.
type House = Shared<Map<String, Json>>;

struct Page {
    house: House,
    printed: Mutex<Vec<String>>,
}

impl Page {
    /// The page before its first block: `house = {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}`.
    fn new() -> Self {
        Page::with(json!({"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}))
    }

    /// The page at a later block, with the house as the blocks before it leave it.
    fn with(house: Json) -> Self {
        Page {
            house: Shared::new(house.as_object().cloned().unwrap_or_default()),
            printed: Mutex::new(Vec::new()),
        }
    }

    /// Python's `print`: each value as its `str`, a space between them.
    fn print(&self, values: &[Json]) {
        let values: Vec<String> = values.iter().map(py::str).collect();
        self.printed.lock().unwrap().push(py::line(&values));
    }

    fn printed(&self) -> Vec<String> {
        self.printed.lock().unwrap().clone()
    }

    /// `house[device]`.
    fn house(&self, device: &str) -> Json {
        self.house.lock()[device].clone()
    }

    /// `house.update(...)`.
    fn update(&self, devices: &[(&str, &str)]) {
        let mut house = self.house.lock();
        for (device, state) in devices {
            house.insert((*device).to_owned(), json!(state));
        }
    }
}

/// Where a request is in its run: its depth, its turn, the request it is under, and the tool that asked it.
type Link<'a> = (usize, usize, Option<usize>, Option<&'a str>);

fn links(result: &Prediction) -> Vec<Link<'_>> {
    let entries = result.trajectory.iter();
    entries
        .map(|entry: &Entry| (entry.depth, entry.iteration, entry.parent, entry.tool_name.as_deref()))
        .collect()
}

/// The options a request offered for a choice.
fn offered(request: &Request, name: &str) -> Vec<String> {
    request.questions[name].labels()
}

// ---------------------------------------------------------------------------------------------
// One request, and what its answers run (RDM.md:14-44)
// ---------------------------------------------------------------------------------------------

/// Set one device to one state.
fn switch(house: &House, device: &str, to: &str) -> String {
    let mut house = house.lock();
    if device == "lock" && to == "locked" && house["door"] == "open" {
        return "lock: refused, the door is open".to_owned();
    }
    house.insert(device.to_owned(), json!(to));
    format!("{device}: now {to}")
}

/// `class Act(dspy.Signature)`, and `act = RDM(Act, tools=[switch])` with a stand-in of its own.
fn act(page: &Page) -> (Rdm, Jev) {
    let signature = Signature::new("Operate a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul(
            "switch",
            "Does `inputs.task` ask for one device in `inputs.house` to be set to another state?",
        )
        .choice("device", "Which device does `inputs.task` mean?", ["tv", "lights", "door", "lock"])
        .choice(
            "to",
            "Which state does `inputs.task` ask for?",
            ["on", "off", "open", "closed", "locked", "unlocked"],
        );
    let house = page.house.clone();
    let switch = Tool::new("switch", "Set one device to one state.", move |call| {
        let (device, to): (String, String) = (call.arg("device")?, call.arg("to")?);
        Ok(switch(&house, &device, &to))
    });
    let act = Rdm::new(signature).tool(switch);
    let jev = Jev::new();
    act.set_model(jev.clone());
    (act, jev)
}

fn one_request(page: &Page, act: &Rdm) -> Prediction {
    let result = act.call(inputs! { "task" => "Turn the TV on.", "house" => &page.house }).unwrap();
    page.print(&[json!(result.observations)]);
    result
}

#[test]
fn e01_one_request_and_what_its_answers_run() {
    let page = Page::new();
    let (act, jev) = act(&page);
    let result = one_request(&page, &act);
    assert_eq!(page.printed(), ["{'switch': 'tv: now on'}"]);

    // One request with three questions, and it is the request the Python module sends.
    assert_eq!(jev.sent(), python(1, "act"));
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(
        (result.probability("switch"), result.chosen("device"), result.chosen("to")),
        (Some(0.9), Some("tv"), Some("on"))
    );
    assert_eq!(page.house("tv"), "on");
    // The run is that request. `device` and `to` are a `Literal` on the page, which the Python
    // records as the plain value; here each is a choice, and is recorded with its probabilities.
    let entry = json!({
        "depth": 0,
        "iteration": 0,
        "parent": null,
        "tool_name": null,
        "inputs": {"task": "Turn the TV on.", "house": {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}},
        "tokens": 0,
        "model": "jev-stand-in",
        "outputs": {
            "switch": {"confidence": 0.8, "value": true, "probability": 0.9},
            "device": {"confidence": 1.0, "value": "tv", "probabilities": {"tv": 1.0, "lights": 0.0, "door": 0.0, "lock": 0.0}},
            "to": {
                "confidence": 1.0,
                "value": "on",
                "probabilities": {"on": 1.0, "off": 0.0, "open": 0.0, "closed": 0.0, "locked": 0.0, "unlocked": 0.0},
            },
        },
        "observations": {"switch": "tv: now on"},
    });
    assert_eq!(json!(result.trajectory), json!([entry]));
}

// ---------------------------------------------------------------------------------------------
// A decision at every step (RDM.md:52-73)
// ---------------------------------------------------------------------------------------------

/// `class Operate(dspy.Signature)`.
fn operate() -> Signature {
    Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?")
}

/// `steps`: a tool for each thing the house can be told to do, under its name.
fn steps(page: &Page) -> Vec<Tool> {
    let each = [
        ("tv", "on"),
        ("tv", "off"),
        ("lights", "on"),
        ("lights", "off"),
        ("door", "closed"),
        ("lock", "locked"),
    ];
    let step = |(device, to): (&'static str, &'static str)| {
        let house = page.house.clone();
        Tool::new(format!("{device} {to}"), format!("Set the {device} to {to}."), move |_| {
            Ok(switch(&house, device, to))
        })
    };
    each.into_iter().map(step).collect()
}

/// `rdm = RDM(Operate, tools=steps)` with a stand-in of its own.
fn rdm(page: &Page) -> (Rdm, Jev) {
    let rdm = Rdm::new(operate()).tools_from(steps(page));
    let jev = Jev::new();
    rdm.set_model(jev.clone());
    (rdm, jev)
}

fn a_decision_at_every_step(page: &Page, rdm: &Rdm) -> Prediction {
    let result = rdm
        .call(inputs! { "task" => "Lock the front door.", "house" => &page.house })
        .unwrap();
    page.print(&[page.house("door"), page.house("lock")]);
    result
}

#[test]
fn e02_a_decision_at_every_step() {
    // The first block left the TV on.
    let page = Page::with(json!({"tv": "on", "lights": "on", "door": "open", "lock": "unlocked"}));
    let (rdm, jev) = rdm(&page);
    let result = a_decision_at_every_step(&page, &rdm);
    assert_eq!(page.printed(), ["closed locked"]);

    // "`result.observations["next"]` holds what was done, most often:"
    assert_eq!(
        py::str(&result.observations["next"]),
        "[['lock locked', 'lock: refused, the door is open'], ['door closed', 'door: now closed'], ['lock locked', 'lock: now locked'], ['finish', None]]"
    );
    assert_eq!(result.termination_reason, Reason::Final);
    // The call's own answer is its first.
    assert_eq!(result.chosen("next"), Some("lock locked"));

    // Four requests, each offering the tools by name and description, with `finish`: the same
    // closed question about the house as it then is, and what the last step did.
    assert_eq!(jev.sent(), python(2, "rdm"));
    assert_eq!(
        offered(&jev.requests()[0], "next"),
        ["tv on", "tv off", "lights on", "lights off", "door closed", "lock locked", "finish"]
    );
    // It is a loop: every turn is at depth 0, under the turn before it, and named for the tool that turn ran.
    assert_eq!(
        links(&result),
        [
            (0, 0, None, None),
            (0, 1, Some(0), Some("lock locked")),
            (0, 2, Some(1), Some("door closed")),
            (0, 3, Some(2), Some("lock locked")),
        ]
    );
    // The last turn as the Python records it: pointing at `finish` runs nothing.
    let probabilities =
        json!({"tv on": 0.0, "tv off": 0.0, "lights on": 0.0, "lights off": 0.0, "door closed": 0.0, "lock locked": 0.0, "finish": 1.0});
    let last = json!({
        "depth": 0,
        "iteration": 3,
        "parent": 2,
        "tool_name": "lock locked",
        "inputs": {
            "task": "Lock the front door.",
            "house": {"tv": "on", "lights": "on", "door": "closed", "lock": "locked"},
            "observation": "lock: now locked",
        },
        "tokens": 0,
        "model": "jev-stand-in",
        "outputs": {"next": {"confidence": 1.0, "value": "finish", "probabilities": probabilities}},
        "observations": {},
    });
    assert_eq!(json!(result.trajectory[3]), last);
}

// ---------------------------------------------------------------------------------------------
// Keeping what was done (RDM.md:87-103)
// ---------------------------------------------------------------------------------------------

fn keeping_what_was_done(page: &Page, rdm: &Rdm) -> (Prediction, Prediction) {
    page.update(&[("tv", "on"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let result = rdm
        .call(inputs! { "task" => "Good night: everything off, and lock up.", "house" => &page.house })
        .unwrap();
    let worked =
        |(name, observed): &(String, Json)| name != "finish" && !observed.as_str().is_some_and(|observed| observed.contains("refused"));
    let done: Vec<String> = result.steps("next").into_iter().filter(worked).map(|(name, _)| name).collect();
    page.print(&[json!(page.house.get())]);

    let description = format!("Good night: everything off, and lock up ({}).", done.join(", "));
    // Take the steps that worked, in the order they were taken.
    let good_night = Tool::new("good night", description, move |call| {
        done.iter().map(|name| call.run(name)).collect::<rdm::Result<Vec<Json>>>()
    });
    rdm.tools().insert(good_night);
    page.update(&[("tv", "on"), ("lights", "on"), ("door", "open"), ("lock", "unlocked")]);
    let night = rdm.call(inputs! { "task" => "Night!", "house" => &page.house }).unwrap();
    page.print(&[json!(page.house.get())]);
    (result, night)
}

#[test]
fn e03_keeping_what_was_done() {
    // The house as the second block left it; this block sets it again before each run.
    let page = Page::with(json!({"tv": "on", "lights": "on", "door": "closed", "lock": "locked"}));
    let (rdm, jev) = rdm(&page);
    let (result, night) = keeping_what_was_done(&page, &rdm);
    assert_eq!(
        page.printed(),
        [
            "{'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}",
            "{'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}",
        ]
    );

    // The first run took six turns, one of them refused.
    assert_eq!(
        result.pointed("next"),
        ["tv off", "lights off", "lock locked", "door closed", "lock locked", "finish"]
    );
    assert_eq!(result.steps("next")[2].1, "lock: refused, the door is open");
    // What worked is kept under one name, which is on the module's map from the next run.
    assert_eq!(
        rdm.tools().names(),
        [
            "tv on",
            "tv off",
            "lights on",
            "lights off",
            "door closed",
            "lock locked",
            "good night"
        ]
    );
    assert_eq!(
        rdm.tools().get("good night").unwrap().description(),
        "Good night: everything off, and lock up (tv off, lights off, door closed, lock locked)."
    );
    // The second run took two: one point does what took several.
    assert_eq!(
        py::str(&night.observations["next"]),
        "[['good night', ['tv: now off', 'lights: now off', 'door: now closed', 'lock: now locked']], ['finish', None]]"
    );
    assert_eq!(links(&night), [(0, 0, None, None), (0, 1, Some(0), Some("good night"))]);
    assert_eq!(
        (result.termination_reason, night.termination_reason),
        (Reason::Final, Reason::Final)
    );

    // Eight requests, as the Python module sends them: the second run's offer `good night` before
    // `finish`, and the request after it shows the list the tool returned, as JSON.
    assert_eq!(jev.sent(), python(3, "rdm"));
    let after = &jev.requests()[7];
    assert_eq!(
        offered(after, "next"),
        [
            "tv on",
            "tv off",
            "lights on",
            "lights off",
            "door closed",
            "lock locked",
            "good night",
            "finish"
        ]
    );
    assert_eq!(
        after.state["inputs"]["observation"],
        r#"["tv: now off", "lights: now off", "door: now closed", "lock: now locked"]"#
    );
}

// ---------------------------------------------------------------------------------------------
// Recursion: a tool that asks the module again (RDM.md:111-136)
// ---------------------------------------------------------------------------------------------

/// `zen = codecs.decode(this.s, "rot13")`: the Zen of Python. Importing `this` prints it, which
/// the Python test reads past; nothing prints it here.
const ZEN: &str = "The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!";

const QUESTION: &str = "How many obvious ways should there be to do something?";

/// The line of the Zen that answers the question.
const FOUND: &str = "There should be one-- and preferably only one --obvious way to do it.";

/// `class Search(dspy.Signature)`.
fn search_for() -> Signature {
    Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?")
}

/// `search = RDM(Search, tools=[look], max_depth=6)` with a stand-in of its own.
fn search() -> (Rdm, Jev) {
    let look = Tool::new("look", "On a yes, ask the same question about each half, one level down.", |call| {
        let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() == 1 {
            return Ok(vec![lines[0].to_owned()]);
        }
        let halves = [&lines[..lines.len() / 2], &lines[lines.len() / 2..]];
        let mut found = Vec::new();
        for half in halves {
            let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
        }
        Ok(found)
    });
    let search = Rdm::new(search_for()).tool(look).max_depth(6);
    let jev = Jev::new();
    search.set_model(jev.clone());
    (search, jev)
}

fn recursion(page: &Page, search: &Rdm) -> Prediction {
    let result = search.call(inputs! { "text" => ZEN, "question" => QUESTION }).unwrap();
    page.print(&[result.observations["look"].clone()]);
    result
}

#[test]
fn e04_recursion_a_tool_that_asks_the_module_again() {
    let page = Page::new();
    let (search, jev) = search();
    let result = recursion(&page, &search);
    assert_eq!(
        page.printed(),
        ["['There should be one-- and preferably only one --obvious way to do it.']"]
    );
    assert_eq!(result.termination_reason, Reason::Final);

    // Eleven requests, each a yes or a no about at most the text it was given, and each the
    // request the Python module sends: the halves, in the order they were asked about.
    assert_eq!((ZEN.chars().count(), ZEN.lines().count()), (856, 21));
    assert_eq!(jev.sent(), python(4, "search"));
    // A yes asks about each half one level down, under the request that said it, and a no ends
    // there. The line is five levels down, which is why the module is given six.
    assert_eq!(
        links(&result),
        [
            (0, 0, None, None),
            (1, 0, Some(0), Some("look")),
            (1, 0, Some(0), Some("look")),
            (2, 0, Some(2), Some("look")),
            (3, 0, Some(3), Some("look")),
            (3, 0, Some(3), Some("look")),
            (4, 0, Some(5), Some("look")),
            (4, 0, Some(5), Some("look")),
            (5, 0, Some(7), Some("look")),
            (5, 0, Some(7), Some("look")),
            (2, 0, Some(2), Some("look")),
        ]
    );
    let yeses: Vec<usize> = (0..11).filter(|at| result.trajectory[*at].value("look") == true).collect();
    assert_eq!(yeses, [0, 2, 3, 5, 7, 9]);
    assert_eq!(result.trajectory[9].inputs["text"], FOUND);
}

// ---------------------------------------------------------------------------------------------
// Several sub-calls at once, other questions, and state no request shows (RDM.md:142-160)
// ---------------------------------------------------------------------------------------------

/// `class Ask(dspy.Signature)`, and `lines = RDM(Ask, tools=[each])` with a stand-in of its own.
fn lines() -> (Rdm, Jev) {
    let ask = Signature::new("Decide whether to look for the answer to a question.")
        .input("question")
        .noul("each", "Could one line of a text state the answer to `inputs.question`?");
    let each = Tool::new(
        "each",
        "Put the question to every line of the text at once, for the answers only.",
        |call| {
            let question: String = call.arg("question")?;
            let text = call.state()["text"].as_str().unwrap_or_default();
            let lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
            let every = lines.iter().map(|line| {
                Ask::new(inputs! { "text" => line, "question" => &question })
                    .signature(search_for())
                    .max_depth(0)
            });
            let verdicts = call.batch(every)?;
            // The yeses, likeliest first.
            let said_yes = Filter::new().rated(lines.iter().zip(verdicts.iter().map(|verdict| verdict.noul("look"))));
            let found: Vec<String> = said_yes.into_iter().map(|(line, _probability)| line.to_string()).collect();
            Ok(found)
        },
    );
    let lines = Rdm::new(ask).tool(each);
    let jev = Jev::new();
    lines.set_model(jev.clone());
    (lines, jev)
}

fn several_at_once(page: &Page, lines: &Rdm) -> Prediction {
    let result = lines
        .call_with(Ask::new(inputs! { "question" => QUESTION }).state(json!({"text": ZEN})))
        .unwrap();
    page.print(&[result.observations["each"].clone()]);
    result
}

#[test]
fn e05_several_sub_calls_at_once_and_state_no_request_shows() {
    let page = Page::new();
    let (lines, jev) = lines();
    let result = several_at_once(&page, &lines);
    assert_eq!(
        page.printed(),
        ["['There should be one-- and preferably only one --obvious way to do it.']"]
    );
    assert_eq!(result.termination_reason, Reason::Final);

    // "No request shows the whole text: the first shows the question alone, and each sub-call
    // one line." Twenty-one requests, the Python module's: twenty of them asked at once, so in
    // an order of their own.
    let (sent, python) = (jev.sent(), python(5, "lines"));
    assert_eq!(sent[0], python[0]);
    assert_eq!(jev.requests()[0].state["inputs"], json!({"question": QUESTION}));
    assert_eq!(sorted(sent), sorted(python));
    // The sub-calls are in the run in the order the tool gave them: one level down, under the
    // request whose tool made them, with other questions, and for their answers only.
    let mut under = vec![(0, 0, None, None)];
    under.resize(21, (1, 0, Some(0), Some("each")));
    assert_eq!(links(&result), under);
    let shown: Vec<&str> = result.trajectory[1..]
        .iter()
        .filter_map(|entry| entry.inputs["text"].as_str())
        .collect();
    assert_eq!(shown, ZEN.lines().filter(|line| !line.is_empty()).collect::<Vec<_>>());
    let ran_nothing = |entry: &Entry| entry.observations.as_ref().is_none_or(IndexMap::is_empty);
    assert!(result.trajectory[1..].iter().all(ran_nothing));
    let yeses: Vec<&Entry> = result.trajectory.iter().filter(|entry| entry.value("look") == true).collect();
    assert_eq!((yeses.len(), &yeses[0].inputs["text"]), (1, &json!(FOUND)));
}

// ---------------------------------------------------------------------------------------------
// Choosing the next question (RDM.md:168-194)
// ---------------------------------------------------------------------------------------------

/// `taxonomy[category]`: the categories under one, where it has any.
fn taxonomy(category: &str) -> Option<&'static [&'static str]> {
    match category {
        "Products" => Some(&["Animals & Pet Supplies", "Furniture", "Kitchen"]),
        "Animals & Pet Supplies" => Some(&["Cat Supplies", "Dog Supplies"]),
        "Cat Supplies" => Some(&["Cat Beds", "Cat Toys", "Cat Window Perches"]),
        _ => None,
    }
}

/// Which category under this one fits the listing?
fn question(category: &str) -> Signature {
    let options = taxonomy(category).unwrap_or_default().iter().map(|child| (*child, ""));
    Signature::parse("listing -> category").unwrap().with_output(
        "category",
        Output::choice("Which category best matches the product in `inputs.listing`?", options),
    )
}

const LISTING: &str = "A padded shelf that suction-cups to a window, so a cat can sun itself.";

/// `classify = RDM(question("Products"), tools=[category])` with a stand-in of its own.
fn classify() -> (Rdm, Jev) {
    let category = Tool::new(
        "category",
        "Put the question under the category chosen, one level down. Comes to the leaf it ends at.",
        |call| {
            let (category, listing): (String, String) = (call.arg("category")?, call.arg("listing")?);
            if taxonomy(&category).is_none() {
                return Ok(json!(category));
            }
            let below = call.sub_with(Ask::new(inputs! { "listing" => listing }).signature(question(&category)))?;
            Ok(below.observations["category"].clone())
        },
    );
    let classify = Rdm::new(question("Products")).tool(category);
    let jev = Jev::new();
    classify.set_model(jev.clone());
    (classify, jev)
}

fn choosing_the_next_question(page: &Page, classify: &Rdm) -> Prediction {
    let result = classify.call(inputs! { "listing" => LISTING }).unwrap();
    page.print(&[result.observations["category"].clone()]);
    result
}

#[test]
fn e06_choosing_the_next_question() {
    let page = Page::new();
    let (classify, jev) = classify();
    let result = choosing_the_next_question(&page, &classify);
    assert_eq!(page.printed(), ["Cat Window Perches"]);
    assert_eq!(result.termination_reason, Reason::Final);
    // "The call's own answer, `.category`, is its own choice: the first."
    assert_eq!(result.chosen("category"), Some("Animals & Pet Supplies"));

    // Three requests, the Python module's, each with the options the answer before it decided.
    assert_eq!(jev.sent(), python(6, "classify"));
    let options: Vec<Vec<String>> = jev.requests().iter().map(|request| offered(request, "category")).collect();
    assert_eq!(
        options,
        [
            vec!["Animals & Pet Supplies", "Furniture", "Kitchen"],
            vec!["Cat Supplies", "Dog Supplies"],
            vec!["Cat Beds", "Cat Toys", "Cat Window Perches"],
        ]
    );
    assert_eq!(
        links(&result),
        [
            (0, 0, None, None),
            (1, 0, Some(0), Some("category")),
            (2, 0, Some(1), Some("category"))
        ]
    );
    // Each level's own choice, and the leaf every level's tool came to.
    let chose: Vec<Json> = result.trajectory.iter().map(|entry| entry.value("category")).collect();
    assert_eq!(chose, ["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"]);
    let came_to = |entry: &Entry| entry.observations.as_ref().map(|observed| observed["category"].clone());
    assert!(
        result
            .trajectory
            .iter()
            .all(|entry| came_to(entry) == Some(json!("Cat Window Perches")))
    );
}

// ---------------------------------------------------------------------------------------------
// Sorting and filtering (RDM.md:202-216)
// ---------------------------------------------------------------------------------------------

/// `wide = RDM(question("Products"), tools=[category])` with a stand-in of its own.
fn wide() -> (Rdm, Jev) {
    let likeliest = Filter::new().top(2);
    let category = Tool::new(
        "category",
        "Follow the two likeliest categories, and come to every leaf they reach.",
        move |call| {
            let listing: String = call.arg("listing")?;
            let mut leaves = Vec::new();
            for (option, _probability) in likeliest.options(call.answer("category")) {
                if taxonomy(&option).is_none() {
                    leaves.push(option);
                } else {
                    let below = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(question(&option)))?;
                    leaves.extend(below.observed::<Vec<String>>("category").unwrap_or_default());
                }
            }
            Ok(leaves)
        },
    );
    let wide = Rdm::new(question("Products")).tool(category);
    let jev = Jev::new();
    wide.set_model(jev.clone());
    (wide, jev)
}

fn sorting_and_filtering(page: &Page, wide: &Rdm) -> Prediction {
    let result = wide.call(inputs! { "listing" => LISTING }).unwrap();
    let leaves = result.observed::<Vec<String>>("category").unwrap_or_default();
    page.print(&[json!(leaves.iter().any(|leaf| leaf == "Cat Window Perches"))]);
    result
}

#[test]
fn e07_sorting_and_filtering() {
    let page = Page::new();
    let (wide, jev) = wide();
    let result = sorting_and_filtering(&page, &wide);
    assert_eq!(page.printed(), ["True"]);
    assert_eq!(result.termination_reason, Reason::Final);

    // Every leaf the two likeliest reach comes back. The stand-in is certain, so the second
    // likeliest is as likely as nothing, and is followed all the same: `top` is how many go on.
    assert_eq!(
        result.observations["category"],
        json!(["Cat Window Perches", "Cat Beds", "Dog Supplies", "Furniture"])
    );
    // Three requests, the Python module's: the categories that have none under them ask nothing.
    assert_eq!(jev.sent(), python(7, "wide"));
    assert_eq!(
        links(&result),
        [
            (0, 0, None, None),
            (1, 0, Some(0), Some("category")),
            (2, 0, Some(1), Some("category"))
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// With a generative model (RDM.md:224-249)
// ---------------------------------------------------------------------------------------------

/// `class Route(dspy.Signature)`, and `route = RDM(Route, tools=[split, act, chat])` with a
/// stand-in of its own. `act` is the module of the first block.
fn route(act: &Rdm, llm: &Llm) -> (Rdm, Jev) {
    let signature = Signature::new("Route a request to a smart home.")
        .input("task")
        .input_typed("house", "dict")
        .noul("split", "Does `inputs.task` ask for more than one distinct thing?")
        .noul("act", "Is `inputs.task` one thing to do to the house?")
        .noul("chat", "Is `inputs.task` only a question to answer in conversation?");

    let write = Generate::parse("task -> parts: list[str]").unwrap().lm(llm.lm.clone());
    let reply = Generate::parse("task -> reply").unwrap().lm(llm.lm.clone());

    let split = Tool::new(
        "split",
        "Each part the generative model writes is a request of its own, routed one level down.",
        move |call| {
            let task: String = call.arg("task")?;
            let parts: Vec<String> = write.call_in(call, inputs! { "task" => task })?.read("parts")?;
            let mut routed = Vec::new();
            for part in parts {
                routed.push(call.sub(inputs! { "task" => part, "house" => call.input("house")? })?);
            }
            Ok(routed)
        },
    );
    let route = Rdm::new(signature)
        .tool(split)
        .tool(Tool::module("act", "", act))
        .tool(reply.tool("chat", ""));
    let jev = Jev::new();
    route.set_model(jev.clone());
    (route, jev)
}

fn with_a_generative_model(page: &Page, route: &Rdm) -> Prediction {
    let result = route
        .call(inputs! { "task" => "Turn the TV off, and tell me whether it will rain.", "house" => &page.house })
        .unwrap();
    page.print(&[page.house("tv")]);
    result
}

#[test]
fn e08_with_a_generative_model() {
    // The house as the third block left it.
    let page = Page::with(json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"}));
    let ((act, acts), llm) = (act(&page), llm());
    let (route, jev) = route(&act, &llm);
    let result = with_a_generative_model(&page, &route);
    assert_eq!(page.printed(), ["off"]);
    assert_eq!(result.termination_reason, Reason::Final);

    // Five requests to decision models and two to the generative one. `route` is asked about
    // the whole request and then about each part that was written, by its own stand-in; `act`
    // is another module with a model of its own, which setting the model on `route` leaves
    // alone. All are the requests the Python module sends.
    assert_eq!(jev.sent(), python(8, "route"));
    assert_eq!(sorted(acts.sent()), sorted(python(8, "act")));
    assert_eq!(llm.asked.requests().len(), 2);

    // The stand-in says of the whole request both that it is several things and that it is one
    // thing to do, so `split` and `act` both run on it, at once: `act` joins the run as a
    // sub-call under the name it has on the map, and so does each part, with `act` under the first.
    let whole = "Turn the TV off, and tell me whether it will rain.";
    let task = |entry: &Entry| entry.inputs["task"].as_str().unwrap_or_default().to_owned();
    let above = |entry: &Entry| entry.parent.map(|parent| task(&result.trajectory[parent]));
    let mut run: Vec<(usize, Option<&str>, String, Option<String>)> = result
        .trajectory
        .iter()
        .map(|entry| (entry.depth, entry.tool_name.as_deref(), task(entry), above(entry)))
        .collect();
    run.sort();
    assert_eq!(
        run,
        [
            (0, None, whole.to_owned(), None),
            (1, Some("act"), whole.to_owned(), Some(whole.to_owned())),
            (1, Some("split"), "Turn the TV off.".to_owned(), Some(whole.to_owned())),
            (1, Some("split"), "Will it rain today?".to_owned(), Some(whole.to_owned())),
            (2, Some("act"), "Turn the TV off.".to_owned(), Some("Turn the TV off.".to_owned())),
        ]
    );
    // What each tool came to. A module that is a tool returns its prediction, and so does the
    // generative one, which the Python shows as `Prediction(reply=...)` and this as its outputs.
    let parts = result.observations["split"].as_array().unwrap();
    assert_eq!(result.observations["act"]["observations"], json!({"switch": "tv: now off"}));
    assert_eq!(parts[0]["observations"]["act"]["observations"], json!({"switch": "tv: now off"}));
    assert_eq!(
        parts[1]["observations"],
        json!({"chat": {"reply": "I cannot see the weather from here."}})
    );

    // On the page the TV is off before this block starts, so its line would be printed whatever
    // the run did. Turned on, it is turned off: the same request, and the same line.
    page.update(&[("tv", "on")]);
    with_a_generative_model(&page, &route);
    assert_eq!(page.printed(), ["off", "off"]);
}

// ---------------------------------------------------------------------------------------------
// A REPL: pointing at code a language model wrote (RDM.md:257-290)
// ---------------------------------------------------------------------------------------------

/// What the ninth block makes, and later blocks use.
struct Cool {
    repl: Repl,
    write: Tool,
    cool: Rdm,
    jev: Jev,
}

/// `class Write(dspy.Signature)` and the writer over it. The page's writer is a `dspy.RLM`
/// that is given `heat` and `repl.define` to call, and whose one output is what `define`
/// returned. Here it is one `Generate` whose outputs are the function, and the REPL defines
/// what it gave: so the instructions do not say to call `define`, and the outputs are the
/// function's name, description and code.
fn writer(llm: &Llm) -> Generate {
    Generate::new(
        "Write one Python function that does what the request still needs.\n\nWhere the function runs, `cooler()` makes it one degree cooler and `heat()` returns the degrees it is now.",
    )
    .input("task")
    .input_typed("house", "dict")
    .output_described("name", "The function's name.")
    .output_described("description", "One sentence for a menu: what choosing it does.")
    .output_described("code", "The function's source.")
    .lm(llm.lm.clone())
}

const WRITE: &str = "Have a new tool written in code and put on this list: nothing listed does what is still needed.";

/// `repl = RDM.Repl(tools=[cooler, heat])`, `write`, and `cool = RDM(Operate, tools=[write, repl])`
/// with a stand-in of its own.
fn cool(page: &Page, llm: &Llm) -> Cool {
    page.house.lock().insert("heat".to_owned(), json!(21));

    let (cooled, read) = (page.house.clone(), page.house.clone());
    // The sandbox, and what written code may call in it.
    let repl = Repl::new()
        .function("cooler", "Make it one degree cooler.", move |_| {
            let mut house = cooled.lock();
            let heat = house["heat"].as_i64().unwrap_or_default() - 1;
            house.insert("heat".to_owned(), json!(heat));
            Ok(format!("heat: now {heat}"))
        })
        .function("heat", "How warm it is, in degrees.", move |_| Ok(read.lock()["heat"].clone()));

    let write = repl.writer_from(&writer(llm).tool("write", WRITE));

    let cool = Rdm::new(operate()).tool(write.clone()).source(repl.clone());
    let jev = Jev::new();
    cool.set_model(jev.clone());
    Cool { repl, write, cool, jev }
}

fn a_repl(page: &Page, cool: &Rdm) -> Prediction {
    let result = cool
        .call(inputs! { "task" => "Bring the heat down to 18 degrees.", "house" => &page.house })
        .unwrap();
    page.print(&[page.house("heat")]);
    result
}

/// What `define` says of the function the stand-in writes.
const WROTE: &str = "`cool_to_18` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Cool the house until it is 18 degrees.";

#[test]
fn e09_a_repl_pointing_at_code_a_language_model_wrote() {
    // The house as the eighth block left it.
    let page = Page::with(json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"}));
    let llm = llm();
    let Cool { repl, cool, jev, .. } = cool(&page, &llm);
    let result = a_repl(&page, &cool);
    assert_eq!(page.printed(), ["18"]);
    assert_eq!(result.termination_reason, Reason::Final);

    // The decision model points at `write`, the language model writes a function, its name is
    // on the map from the next request, and the decision model points at it.
    assert_eq!(result.pointed("next"), ["write", "cool_to_18", "finish"]);
    assert_eq!(
        links(&result),
        [
            (0, 0, None, None),
            (0, 1, Some(0), Some("write")),
            (0, 2, Some(1), Some("cool_to_18"))
        ]
    );
    let offers: Vec<Vec<String>> = jev.requests().iter().map(|request| offered(request, "next")).collect();
    assert_eq!(
        offers,
        [
            vec!["write", "finish"],
            vec!["write", "cool_to_18", "finish"],
            vec!["write", "cool_to_18", "finish"]
        ]
    );
    assert_eq!(result.steps("next")[1].1, "it is 18 degrees");
    assert_eq!(llm.asked.requests().len(), 1);

    // What was written is the program's, and the request whose tool wrote it says so. The
    // REPL's functions are not among the module's own tools.
    assert_eq!(
        json!(repl.written()),
        json!({"cool_to_18": {"description": "Cool the house until it is 18 degrees.", "code": COOL_TO_18}})
    );
    assert_eq!(
        json!(result.trajectory[0].changes),
        json!([{"by": "repl", "did": "define", "name": "cool_to_18"}])
    );
    assert_eq!(cool.tools().names(), ["write"]);

    // Three requests. The first and the third are the Python module's. The second differs in
    // what it shows as the observation: here what `define` said, which is what `write` returns;
    // in the Python the whole prediction of the `dspy.RLM` that wrote, of which that sentence
    // is the output, as the JSON of its repr.
    let (sent, python) = (jev.sent(), python(9, "cool"));
    assert_eq!((&sent[0], &sent[2]), (&python[0], &python[2]));
    assert_eq!(result.steps("next")[0].1, WROTE);
    assert_eq!(jev.requests()[1].state["inputs"]["observation"], WROTE);
    let in_python: Json = serde_json::from_str(&python[1]).unwrap();
    let observed = in_python["state"]["inputs"]["observation"].as_str().unwrap();
    assert!(observed.starts_with(r#""Prediction(\n    wrote='`cool_to_18` was written"#) && observed.contains(WROTE));
    assert_eq!(in_python["questions"], json!(jev.requests()[1].questions));
}

// ---------------------------------------------------------------------------------------------
// A note is a function (RDM.md:298-302)
// ---------------------------------------------------------------------------------------------

fn a_note_is_a_function(page: &Page, repl: &Repl) {
    repl.define(
        "boiler",
        "What was found wrong with the boiler.",
        "def boiler():\n    return 'The pilot light is out, and the gas valve is open.'",
    )
    .unwrap();
    page.print(&[json!(repl.execute("print(boiler())").unwrap())]);
}

#[test]
fn e10_a_note_is_a_function() {
    let page = Page::with(json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"}));
    let Cool { repl, cool, .. } = cool(&page, &llm());
    a_repl(&page, &cool);
    a_note_is_a_function(&page, &repl);
    assert_eq!(page.printed(), ["18", "The pilot light is out, and the gas valve is open."]);

    // "`boiler` is now a name on `cool`'s map beside the function written above."
    let on_the_map: Vec<String> = repl.offers().iter().map(|tool| tool.name().to_owned()).collect();
    assert_eq!(on_the_map, ["cool_to_18", "boiler"]);
    // What the REPL holds, as the Python's `repl.show()` gives it for the writer to read.
    assert_eq!(
        repl.show(),
        "cool_to_18: Cool the house until it is 18 degrees.\ndef cool_to_18():\n    while heat() > 18:\n        cooler()\n    return f'it is {heat()} degrees'\n\nboiler: What was found wrong with the boiler.\ndef boiler():\n    return 'The pilot light is out, and the gas valve is open.'"
    );
}

// ---------------------------------------------------------------------------------------------
// Several REPLs (RDM.md:314-324)
// ---------------------------------------------------------------------------------------------

/// `home = RDM(Operate, tools=[room("heating", ...), room("lights", ...)])`, and the
/// environments its rooms were opened in.
///
/// The page's `repl.spawn(name, description)` keeps the description with the environment, and
/// the REPL keeps the environments it opened, to shut them down with itself. Here `spawn`
/// takes the name alone, the description is the tool's, and an environment is its own to keep:
/// there is nothing to shut down, a REPL ends when it is dropped.
fn several_repls(repl: &Repl, write: &Tool, llm: &Llm) -> (Rdm, Vec<Repl>) {
    let mut environments = Vec::new();
    // An environment, a tool that writes into it, and a module over the two: one name on a map.
    let mut room = |name: &str, description: &str| {
        let environment = repl.spawn(name);
        let writes = environment.writer_from(&writer(llm).tool("write", write.description()));
        environments.push(environment.clone());
        Tool::module(name, description, &Rdm::new(operate()).tool(writes).source(environment))
    };
    let home = Rdm::new(operate())
        .tool(room("heating", "Anything about the temperature."))
        .tool(room("lights", "Anything about the lights."));
    (home, environments)
}

#[test]
fn e11_several_repls() {
    let page = Page::with(json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked"}));
    let llm = llm();
    let Cool { repl, write, cool, .. } = cool(&page, &llm);
    a_repl(&page, &cool);
    // The page builds `home` and does not call it, and nothing is printed.
    let (home, environments) = several_repls(&repl, &write, &llm);
    assert_eq!(home.tools().names(), ["heating", "lights"]);
    let rooms = home.tools().all();
    assert_eq!(
        rooms.iter().map(|room| (room.name(), room.description())).collect::<Vec<_>>(),
        [
            ("heating", "Anything about the temperature."),
            ("lights", "Anything about the lights.")
        ]
    );
    assert_eq!(environments.iter().map(Repl::name).collect::<Vec<_>>(), ["heating", "lights"]);
    // "isolated from the first: a function written in one does not exist in another", and the
    // functions of the host are in each.
    for environment in &environments {
        assert_eq!((environment.written().len(), environment.functions()), (0, repl.functions()));
    }
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["cool_to_18"]);
    assert!(llm.asked.requests().len() == 1 && page.printed() == ["18"]);
}

// ---------------------------------------------------------------------------------------------
// Inputs of any size (RDM.md:332-337)
// ---------------------------------------------------------------------------------------------

/// `covered = RDM(Search, fit=RDM.Fit(max_chars=400))` with a stand-in of its own.
fn covered() -> (Rdm, Jev) {
    let covered = Rdm::new(search_for()).fit(Fit::new().max_chars(400)); // no request shows more than 400 characters of the text
    let jev = Jev::new();
    covered.set_model(jev.clone());
    (covered, jev)
}

fn inputs_of_any_size(page: &Page, covered: &Rdm) -> Prediction {
    let result = covered.call(inputs! { "text" => ZEN, "question" => QUESTION }).unwrap();
    page.print(&[result.value("look")]);
    result
}

/// The one part of the Zen the stand-in says is needed.
const NEEDED_PART: &str = "preferably only one --obvious ";

#[test]
fn e12_inputs_of_any_size() {
    let page = Page::new();
    let (covered, jev) = covered();
    let result = inputs_of_any_size(&page, &covered);
    assert_eq!(page.printed(), ["True"]);
    assert_eq!(result.termination_reason, Reason::Final);
    assert!(result.observations.is_empty());

    // The request is in the run before what `Fit` asked for it, and each thing `Fit` asked is
    // an answers-only sub-call under it, in no tool's name: `needed`, about one part of the text.
    let parts = &result.trajectory[1..];
    assert_eq!(links(&result)[0], (0, 0, None, None));
    assert!(links(&result)[1..].iter().all(|link| *link == (1, 0, Some(0), None)));
    let asks_needed = |entry: &Entry| entry.outputs.as_ref().is_some_and(|outputs| outputs.keys().eq([NEEDED]));
    assert!(parts.iter().all(asks_needed));
    // The parts are the text, each asked about once, and one is said to be needed.
    let text = |entry: &Entry| entry.inputs["text"].as_str().unwrap_or_default().to_owned();
    assert_eq!(parts.iter().map(text).collect::<String>(), ZEN);
    let needed: Vec<String> = parts.iter().filter(|part| part.value(NEEDED) == true).map(text).collect();
    assert_eq!(needed, [NEEDED_PART]);
    // The inputs are 958 characters as a request shows them, and no request shows more than
    // 400: the last shows the part that is needed, and of the rest what there is room for, the
    // earliest first, with a mark where parts were left out.
    let shown = text(&result.trajectory[0]);
    assert_eq!(fit::size(&json!({"text": ZEN, "question": QUESTION})), 958);
    assert!(jev.requests().iter().all(|request| fit::size(&request.state["inputs"]) <= 400));
    assert!(shown.starts_with("The Zen of Python, by Tim Peters\n\nBeautiful is better than ugly.\n") && shown.ends_with(NEEDED_PART));
    assert_eq!(asking(jev.sent(), "look").len(), 1);
    assert_eq!(jev.requests().last().unwrap().state["inputs"]["text"], shown.as_str());

    // Where this is not the Python's run. A part may take 33 characters. The Python measures a
    // line as it is, so a line of 33 with its line break is one part; `Fit` here measures it as
    // the JSON a request shows, where the line break is two characters and the quotes two
    // more, and cuts that line in two. So the text is 40 parts here and 35 there, the run 41
    // requests against 36, and what fills the room beside the needed part is not the same.
    let in_python = python(12, "covered");
    assert_eq!((parts.len(), asking(in_python.clone(), NEEDED).len()), (40, 35));
    assert_eq!((result.trajectory.len(), jev.requests().len(), in_python.len()), (41, 41, 36));
    assert_eq!(text(&parts[0]), "The Zen of Python, by Tim ");
    assert!(
        in_python
            .iter()
            .any(|request| request.contains(r#""text":"The Zen of Python, by Tim Peters\n""#))
    );
    let rest = "Explicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated.\n";
    assert_eq!(
        shown,
        format!("The Zen of Python, by Tim Peters\n\nBeautiful is better than ugly.\n{rest}{GAP}purity.\n{GAP}{NEEDED_PART}")
    );
    let shown_in_python: Json = serde_json::from_str(&asking(in_python, "look")[0]).unwrap();
    assert_eq!(
        shown_in_python["state"]["inputs"]["text"],
        format!(
            "The Zen of Python, by Tim Peters\n\nBeautiful is better than ugly.\n{rest}Flat is better than nested.\n{GAP}{NEEDED_PART}"
        )
    );
}

// ---------------------------------------------------------------------------------------------
// A map with labels (RDM.md:345-362)
// ---------------------------------------------------------------------------------------------

/// `class About(dspy.Signature)`, `near = RDM.Near(About, top=2)`, and
/// `guided = RDM(Operate, tools=steps, fit=near)` with a stand-in of its own.
fn guided(page: &Page) -> (Near, Rdm, Jev) {
    let about = Signature::new("Say what a text is about.")
        .input("task")
        .noul("screen", "Is `inputs.task` about the TV?")
        .noul("light", "Is `inputs.task` about the lights?")
        .noul("entry", "Is `inputs.task` about the door or its lock?");

    let near = Near::new(about).top(2);
    let guided = Rdm::new(operate()).tools_from(steps(page)).fit(near.clone());
    let jev = Jev::new();
    guided.set_model(jev.clone());
    (near, guided, jev)
}

fn a_map_with_labels(page: &Page, guided: &Rdm) -> Prediction {
    page.update(&[("door", "open"), ("lock", "unlocked")]);
    let result = guided
        .call(inputs! { "task" => "Lock the front door.", "house" => &page.house })
        .unwrap();
    page.print(&[page.house("door"), page.house("lock")]);
    result
}

/// The house as the ninth block left it, which is how the thirteenth finds it.
fn house_after_the_repl() -> Json {
    json!({"tv": "off", "lights": "off", "door": "closed", "lock": "locked", "heat": 18})
}

#[test]
fn e13_a_map_with_labels() {
    let page = Page::with(house_after_the_repl());
    let (near, guided, jev) = guided(&page);
    let result = a_map_with_labels(&page, &guided);
    assert_eq!(page.printed(), ["closed locked"]);
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(result.pointed("next"), LOCK_UP);

    // Eleven requests, the Python module's. Seven are the three questions put to the request
    // and to each of the six steps, once and at once, so in an order of their own; four are the
    // pointer's turns, in turn.
    let (sent, python) = (jev.sent(), python(13, "guided"));
    assert_eq!(sorted(sent.clone()), sorted(python.clone()));
    assert_eq!(asking(sent, "next"), asking(python, "next"));
    // "of the six only `door closed` and `lock locked` are offered, with `finish`", at every turn.
    let turns: Vec<Request> = jev
        .requests()
        .into_iter()
        .filter(|request| request.questions.contains_key("next"))
        .collect();
    assert_eq!(turns.len(), 4);
    assert!(
        turns
            .iter()
            .all(|turn| offered(turn, "next") == ["door closed", "lock locked", "finish"])
    );
    // What `Near` asks are sub-calls of the run, under the first turn and in no tool's name;
    // the later turns ask nothing more of it.
    let mut run = vec![(0, 0, None, None)];
    run.resize(8, (1, 0, Some(0), None));
    run.extend([
        (0, 1, Some(0), Some("lock locked")),
        (0, 2, Some(8), Some("door closed")),
        (0, 3, Some(9), Some("lock locked")),
    ]);
    assert_eq!(links(&result), run);
    // "What was answered is kept in `near.answers`", by the text asked, and every number has a name.
    let labels = json!({
        "Lock the front door.": {"screen": 0.1, "light": 0.1, "entry": 0.9},
        "tv on: Set the tv to on.": {"screen": 0.9, "light": 0.1, "entry": 0.1},
        "tv off: Set the tv to off.": {"screen": 0.9, "light": 0.1, "entry": 0.1},
        "lights on: Set the lights to on.": {"screen": 0.1, "light": 0.9, "entry": 0.1},
        "lights off: Set the lights to off.": {"screen": 0.1, "light": 0.9, "entry": 0.1},
        "door closed: Set the door to closed.": {"screen": 0.1, "light": 0.1, "entry": 0.9},
        "lock locked: Set the lock to locked.": {"screen": 0.1, "light": 0.1, "entry": 0.9},
    });
    assert_eq!(json!(near.answers()).to_string(), labels.to_string());
    // The labels are coordinates, and a step is as near the request as what their labels share.
    // Reading them again asks nothing.
    let texts = [
        "Lock the front door.",
        "lock locked: Set the lock to locked.",
        "tv on: Set the tv to on.",
    ];
    let at = near.embed(texts).unwrap();
    assert_eq!(json!(at[0]), json!({"screen": 0.1, "light": 0.1, "entry": 0.9}));
    let hundredths = |score: f64| (score * 100.0).round();
    assert_eq!(
        (hundredths(near.score(&at[0], &at[1])), hundredths(near.score(&at[0], &at[2]))),
        (83.0, 19.0)
    );
    assert_eq!(jev.requests().len(), 11);
}

// ---------------------------------------------------------------------------------------------
// A dimension is a name and a question (RDM.md:372-376)
// ---------------------------------------------------------------------------------------------

fn writing_a_dimension(page: &Page, near: &Near) {
    near.add("power", "Is `inputs.task` about switching something on or off?").unwrap();
    page.print(&[json!(near.questions().keys().collect::<Vec<_>>())]);
}

#[test]
fn e14_a_dimension_is_a_name_and_a_question() {
    let page = Page::with(house_after_the_repl());
    let (near, guided, jev) = guided(&page);
    a_map_with_labels(&page, &guided);
    writing_a_dimension(&page, &near);
    assert_eq!(page.printed(), ["closed locked", "['screen', 'light', 'entry', 'power']"]);
    // Writing one asks nothing, and `show` gives what is there for a writer to read.
    assert_eq!(jev.requests().len(), 11);
    assert_eq!(
        near.show(),
        "screen: Is `inputs.task` about the TV?\nlight: Is `inputs.task` about the lights?\nentry: Is `inputs.task` about the door or its lock?\npower: Is `inputs.task` about switching something on or off?"
    );
}

// ---------------------------------------------------------------------------------------------
// Thresholds (RDM.md:442-451)
// ---------------------------------------------------------------------------------------------

/// The four places of the page. There each is a predictor's: the module's own, one set on the
/// module for a question of its sub-calls, `Fit`'s and `Near`'s. Here the first three are the
/// module's, by the name of the output, and the fourth is the `Near`'s, by its dimension.
fn thresholds(search: &Rdm, lines: &Rdm, covered: &Rdm, near: &Near) {
    search.set_params("look", Params::threshold(0.7)); // a question of the module's own signature

    lines.set_params("look", Params::threshold(0.7)); // a question asked only in sub-calls

    covered.set_params(NEEDED, Params::threshold(0.3)); // Fit's question about each part of a long input

    near.set_params("entry", Params::threshold(0.7)); // Near's question about a dimension
}

#[test]
fn e15_thresholds() {
    // The modules of the blocks before, each having run as the page runs it.
    let page = Page::with(house_after_the_repl());
    let ((search, searched), (lines, lined), (covered, covering)) = (search(), lines(), covered());
    let (near, guided, guiding) = guided(&page);
    let before = recursion(&page, &search);
    several_at_once(&page, &lines);
    inputs_of_any_size(&page, &covered);
    a_map_with_labels(&page, &guided);
    writing_a_dimension(&page, &near);
    let asked: Vec<usize> = [&searched, &lined, &covering, &guiding].map(|jev| jev.requests().len()).into();

    // The block sets four thresholds, asks nothing and prints nothing.
    let printed = page.printed();
    thresholds(&search, &lines, &covered, &near);
    assert_eq!(page.printed(), printed);
    assert_eq!(
        [&searched, &lined, &covering, &guiding].map(|jev| jev.requests().len()).to_vec(),
        asked
    );
    assert_eq!(
        [
            search.params_of("look"),
            lines.params_of("look"),
            covered.params_of(NEEDED),
            near.params_of("entry")
        ],
        [0.7, 0.7, 0.3, 0.7].map(|threshold| Some(Params::threshold(threshold)))
    );

    // What each then does, beside the Python module called once more with the same four set
    // (block 16 of the requests). A threshold is never sent: what it changes is what is decided
    // from the answers, and so what a tool or a fit then does.
    for jev in [&searched, &lined, &covering, &guiding] {
        jev.forget();
    }
    guiding.plan("Lock the front door.", &LOCK_UP);

    // A yes at 0.9 is as far above 0.7 as it is: still a yes, and less sure a one.
    let result = recursion(&page, &search);
    assert_eq!(searched.sent(), python(16, "search"));
    assert_eq!(result.observations, before.observations);
    assert_eq!(before.noul("look").map(|look| look.confidence), Some(0.8));
    assert_eq!(
        result.noul("look").map(|look| (look.value, look.confidence)),
        Some((true, 0.2857142857142858))
    );

    // The module's threshold for `look` is applied to the sub-calls that ask it.
    let result = several_at_once(&page, &lines);
    assert_eq!(sorted(lined.sent()), sorted(python(16, "lines")));
    let found = result.trajectory.iter().find(|entry| entry.inputs["text"] == FOUND).unwrap();
    assert_eq!(
        json!(found.outputs),
        json!({"look": {"confidence": 0.2857142857142858, "value": true, "probability": 0.9}})
    );
    assert_eq!(result.noul("each").map(|each| each.confidence), Some(0.8));

    // "`Fit` keeps the likeliest parts that fit until `needed` is given a threshold; then it
    // keeps only the parts said to be needed." The question put to what was kept is the Python's.
    let result = inputs_of_any_size(&page, &covered);
    assert_eq!(result.trajectory[0].inputs["text"], NEEDED_PART);
    let kept = asking(covering.sent(), "look");
    assert_eq!((kept.len(), &kept), (1, &asking(python(16, "covered"), "look")));
    assert_eq!((result.value("look"), result.trajectory.len()), (json!(true), 41));

    // A dimension was added since the texts were given their labels, so each is asked again,
    // all four questions, as the Python asks them. Under the threshold a no to `entry` counts as
    // nothing, and the same two steps are offered.
    let result = a_map_with_labels(&page, &guided);
    assert_eq!(sorted(guiding.sent()), sorted(python(16, "guided")));
    assert_eq!(result.pointed("next"), LOCK_UP);
    let turns = asking(guiding.sent(), "next");
    assert!(turns.len() == 4 && turns.iter().all(|turn| turn.contains(r#""criteria":{"door closed":"#)));
    let texts = [
        "Lock the front door.",
        "lock locked: Set the lock to locked.",
        "tv on: Set the tv to on.",
    ];
    let at = near.embed(texts).unwrap();
    assert_eq!(json!(at[2]), json!({"screen": 0.9, "light": 0.1, "entry": 0.0, "power": 0.1}));
    assert_eq!(
        (near.score(&at[0], &at[1]), near.score(&at[0], &at[2])),
        (0.8400000000000001, 0.11000000000000001)
    );
    assert_eq!(page.printed()[printed.len()..], printed[..4]);
}

// ---------------------------------------------------------------------------------------------
// The page, top to bottom.
// ---------------------------------------------------------------------------------------------

/// What the page says its program prints: the comment lines after each `print`, in order.
const DOCUMENTED: [&str; 14] = [
    "{'switch': 'tv: now on'}",
    "closed locked",
    "{'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}",
    "{'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}",
    "['There should be one-- and preferably only one --obvious way to do it.']",
    "['There should be one-- and preferably only one --obvious way to do it.']",
    "Cat Window Perches",
    "True",
    "off",
    "18",
    "The pilot light is out, and the gas valve is open.",
    "True",
    "closed locked",
    "['screen', 'light', 'entry', 'power']",
];

/// The fifteen blocks as the one program they are on the page: one house, each module with the
/// stand-in it was given where the page makes it, and what a block makes there for those after it.
#[test]
fn the_page_prints_what_it_says() {
    let page = Page::new();

    let (act, _) = act(&page);
    one_request(&page, &act);
    let (rdm, _) = rdm(&page);
    a_decision_at_every_step(&page, &rdm);
    keeping_what_was_done(&page, &rdm);
    let (search, _) = search();
    recursion(&page, &search);
    let (lines, _) = lines();
    several_at_once(&page, &lines);
    let (classify, _) = classify();
    choosing_the_next_question(&page, &classify);
    let (wide, _) = wide();
    sorting_and_filtering(&page, &wide);
    let llm = llm();
    let (route, _) = route(&act, &llm);
    with_a_generative_model(&page, &route);
    let Cool { repl, write, cool, .. } = cool(&page, &llm);
    a_repl(&page, &cool);
    a_note_is_a_function(&page, &repl);
    let (home, _) = several_repls(&repl, &write, &llm);
    let (covered, _) = covered();
    inputs_of_any_size(&page, &covered);
    let (near, guided, _) = guided(&page);
    a_map_with_labels(&page, &guided);
    writing_a_dimension(&page, &near);
    thresholds(&search, &lines, &covered, &near);

    assert_eq!(page.printed(), DOCUMENTED);
    assert_eq!(json!(page.house.get()).to_string(), house_after_the_repl().to_string());
    assert_eq!(home.tools().names(), ["heating", "lights"]);
    assert_eq!(llm.asked.requests().len(), 3);
}

// ---------------------------------------------------------------------------------------------
// What the page says outside its blocks, where no other test says it.
// ---------------------------------------------------------------------------------------------

/// The constructor's table: `max_iters` 20, `max_depth` 3, `max_predictor_calls` 256 and no
/// `max_tokens`, each seen in what a module given none of them does.
#[test]
fn the_limits_a_module_has_unless_it_is_given_others() {
    // "Turns one pointer's loop may take": a loop that is never told to finish takes twenty.
    // Nothing bounds the tokens: these twenty use twenty thousand million.
    let count = Shared::new(0_u64);
    let counted = count.clone();
    let again = Tool::new("again", "Count one more.", move |_| {
        Ok(counted.update(|count| {
            *count += 1;
            *count
        }))
    });
    let always = Scripted::new(|request: &Request| {
        let labels = request.questions["next"].labels();
        [("next", Answer::chose("again", labels.iter().map(String::as_str)))]
    });
    let looping = Signature::new("Count.").input_typed("count", "int").tool("next", "What next?");
    let rdm = Rdm::new(looping).tool(again).model(always.tokens(1_000_000_000));
    let result = rdm.call(inputs! { "count" => &count }).unwrap();
    assert_eq!((result.steps("next").len(), count.get()), (20, 20));
    assert_eq!(
        (result.termination_reason, result.termination_reason.as_str()),
        (Reason::MaxIters, "max_iters")
    );
    assert_eq!(result.tokens(), 20_000_000_000);

    // "The depth at which tools are not run": a tool that always asks one level down runs at
    // depths 0, 1 and 2. The request at depth 3 is answered, and its tool is not run.
    let deeper = Tool::new("deeper", "", |call| {
        let n: u64 = call.arg("n")?;
        Ok(call.sub(inputs! { "n" => n + 1 })?.termination_reason)
    });
    let going = Signature::new("Go on.")
        .input_typed("n", "int")
        .noul("deeper", "Is there further to go from `inputs.n`?");
    let yes = Scripted::new(|_: &Request| [("deeper", Answer::noul(0.9))]);
    let result = Rdm::new(going).tool(deeper).model(yes).call(inputs! { "n" => 0 }).unwrap();
    assert_eq!(result.trajectory.iter().map(|entry| entry.depth).collect::<Vec<_>>(), [0, 1, 2, 3]);
    let leaf = &result.trajectory[3];
    assert!(leaf.value("deeper") == true && leaf.observations.as_ref().is_none_or(IndexMap::is_empty));
    assert_eq!(
        (result.termination_reason, result.termination_reason.as_str()),
        (Reason::MaxDepth, "max_depth")
    );

    // "Requests answered in one run, sub-calls included": of three hundred sub-calls under one
    // request, 255 are answered.
    let all = Tool::new("all", "", |call| {
        let answers = call.batch((0..300).map(|n| Ask::new(inputs! { "n" => n }).max_depth(0)))?;
        Ok(answers.iter().filter(|answer| answer.answered()).count())
    });
    let every = Signature::new("Ask.")
        .input_typed("n", "int")
        .noul("all", "Should every number be asked about?");
    let yes = Scripted::new(|_: &Request| [("all", Answer::noul(0.9))]);
    let result = Rdm::new(every).tool(all).model(yes).call(inputs! { "n" => 1000 }).unwrap();
    assert_eq!(result.observations["all"], 255);
    assert_eq!(result.trajectory.iter().filter(|entry| entry.asked()).count(), 256);
    assert_eq!(
        (result.termination_reason, result.termination_reason.as_str()),
        (Reason::MaxPredictorCalls, "max_predictor_calls")
    );
}

/// The names a prediction has of its own, and a call takes as its own arguments: a signature
/// may use none of them, as an input or as an output.
#[test]
fn the_names_a_signature_may_not_take() {
    assert_eq!(
        RESERVED,
        [
            "signature",
            "state",
            "max_depth",
            "observations",
            "termination_reason",
            "trajectory"
        ]
    );
    for name in RESERVED {
        let as_input = Signature::new("Find.").input(name).noul("found", "Is it there?");
        let as_output = Signature::new("Find.").input("text").noul(name, "Is it there?");
        for signature in [as_input, as_output] {
            let error = Rdm::new(signature).check().unwrap_err();
            assert!(error.is_usage() && error.to_string().contains("reserved by RDM"), "{name}: {error}");
        }
    }
}
