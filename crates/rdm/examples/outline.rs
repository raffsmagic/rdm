//! The shape of an RLM, with no language model. The handbook stays in the program as the run's
//! `state`, and no request ever shows it whole. Plain code turns it into an outline; a request
//! shows one entry and the headings below it, never the text under them. The same question is
//! asked at every node, and a yes asks it again about each entry under that one, all the way out
//! to a passage.
//!
//! ```text
//! cargo run -p rdm --example outline
//! cargo run -p rdm --example outline -- --help
//! RDM_POST=1 cargo run -p rdm --example outline -- --question 2
//! ```
//!
//! The last is the run of the post's section "Self-similarity and external state", printed as
//! the post prints it.

mod common;

use common::{Form, Say, py};
use indexmap::IndexMap;
use rdm::systemone::Question;
use rdm::{Ask, Json, Rdm, Signature, Tool, inputs};
use serde::Deserialize;
use serde_json::json;

/// `{heading: [the headings and passages directly under it]}`, in the order of the handbook.
type Outline = IndexMap<String, Vec<String>>;

/// What the run is given as its state: no request shows it.
#[derive(Deserialize)]
struct State {
    outline: Outline,
}

/// `{heading: [the headings and passages directly under it]}`, made by plain code.
fn outline(markdown: &str, title: &str) -> Outline {
    let (mut under, mut path) = (Outline::from([(title.to_owned(), Vec::new())]), vec![title.to_owned()]);
    for block in markdown.trim().split("\n\n") {
        let depth = block.len() - block.trim_start_matches('#').len();
        let block = if depth > 0 { block.trim_start_matches(['#', ' ']) } else { block };
        if depth > 0 {
            path.truncate(depth);
            path.push(block.to_owned());
            under.insert(block.to_owned(), Vec::new());
        }
        let above = &path[path.len() - if depth > 0 { 2 } else { 1 }];
        under[above].push(block.to_owned());
    }
    under
}

/// Every heading below an entry: all that a request shows of what is under it.
fn headings(under: &Outline, entry: &str) -> Vec<String> {
    let mut below = Vec::new();
    for each in under.get(entry).into_iter().flatten().filter(|each| under.contains_key(*each)) {
        below.push(each.clone());
        below.extend(headings(under, each));
    }
    below
}

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    // No language model is made anywhere in this program.

    // What the form sets: the question put to the handbook.
    let form = Form::new(
        "outline",
        "The shape of an RLM",
        "RDM calls RDM out to a leaf over an outline held in the program, with no language model anywhere.",
    )
    .text(
        "question",
        "Ask the handbook",
        &[
            "Can I work from another country?",
            "What is the nightly hotel limit in London?",
            "Who do I tell about a lost laptop?",
            "How many days of annual leave are there?",
        ],
    )
    .read();
    let question = form.text_of("question");

    let handbook = common::HANDBOOK;

    let under = outline(handbook, "Handbook");

    let navigate = Signature::new("Search a handbook for the answer to a question, one entry at a time.")
        .input_described("entry", "A heading of the handbook, or a passage under one.")
        .input_with("contains", "list[str]", "The headings under `inputs.entry`. Empty for a passage.")
        .input("question")
        .noul(
            "explore",
            "Does `inputs.entry` state the answer to `inputs.question`, or does `inputs.entry` or any \
             heading listed in `inputs.contains` name the topic that `inputs.question` asks about?",
        );

    // Runs on a yes. Asks the same question about each entry under this one, one level down.
    let explore = Tool::new(
        "explore",
        "Runs on a yes. Asks the same question about each entry under this one, one level down.",
        |call| {
            let (entry, question): (String, String) = (call.arg("entry")?, call.arg("question")?);
            let State { outline } = call.state_as()?; // what the run was given as its state: no request shows it
            let Some(under) = outline.get(&entry) else {
                return Ok(vec![entry]); // a leaf: a passage, and the model said yes to it
            };
            py::print!("explored:", entry);
            let mut found = Vec::new();
            for each in under {
                let below = call.sub(inputs! { "entry" => each, "contains" => headings(&outline, each), "question" => &question })?;
                found.extend(below.observed::<Vec<String>>("explore").unwrap_or_default());
            }
            Ok(found)
        },
    );

    let walk = Rdm::new(navigate).tool(explore).max_depth(10).model(decision_model);

    common::show_text(handbook); // draws the handbook, and on it the passages a request showed
    common::show_tree("Handbook", |entry| under.get(entry).cloned().unwrap_or_default()); // draws, in grey, the outline code laid out
    let shown = inputs! { "entry" => "Handbook", "contains" => headings(&under, "Handbook"), "question" => &question };
    let result = walk.call_with(Ask::new(shown).state(json!({"outline": under})))?;
    common::trace(&result);

    if as_the_post() {
        // What the post prints of this run: what was found, the number of requests, and how much of
        // the handbook the largest of them shows, which is its entry and the headings listed under it.
        py::print!(result.observation("explore").cloned().unwrap_or(json!([])));
        let shows =
            |inputs: &Json| inputs["entry"].as_str().unwrap_or_default().chars().count() + py::repr(&inputs["contains"]).chars().count();
        let largest = result.trajectory.iter().map(|request| shows(&request.inputs)).max().unwrap_or(0);
        let (requests, characters) = (result.trajectory.len(), thousands(handbook.chars().count()));
        println!("{requests} requests again, but the largest shows {largest} characters of a {characters}-character handbook");
        return Ok(());
    }

    py::print!();
    py::print!(question);
    let found = result.observation("explore").cloned();
    py::print!("->", found.unwrap_or(json!("the model said the handbook does not cover this")));
    // `len(str(request["inputs"]))`: a request as Python writes it, with its names and its question.
    let largest = result
        .trajectory
        .iter()
        .map(|request| py::repr(&request.inputs).chars().count())
        .max();
    py::print!(
        result.trajectory.len(),
        "requests; the largest shows",
        largest.unwrap_or(0),
        "characters of a",
        handbook.chars().count(),
        "character handbook"
    );
    Ok(())
}

/// Whether the run is to be printed as the post prints it: `RDM_POST` is set to something other than `0`.
fn as_the_post() -> bool {
    std::env::var("RDM_POST").is_ok_and(|value| !value.is_empty() && value != "0")
}

/// A number as the post writes one: `2,555`.
fn thousands(number: usize) -> String {
    let digits = number.to_string();
    let mut written = String::new();
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at).is_multiple_of(3) {
            written.push(',');
        }
        written.push(digit);
    }
    written
}

/// The stand-in decision model, which answers when the environment asks for no real one. It knows
/// what each preset question is about: a word of the question, the heading its answer is under,
/// and how the passage that states it begins. It says yes to that heading, to an entry that lists
/// it, and to that passage, and no to anything else.
fn standin(inputs: &Json, _name: &str, _question: &Question) -> Say {
    const ANSWERS: [(&str, &str, &str); 4] = [
        ("another country", "Working from abroad", "Working from another country"),
        ("hotel", "Hotels", "The nightly limit for accommodation"),
        ("lost", "Reporting an incident", "Report a lost device"),
        ("annual leave", "Annual leave", "Full-time staff receive 25 days"),
    ];
    let question = inputs["question"].as_str().unwrap_or_default().to_lowercase();
    let entry = inputs["entry"].as_str().unwrap_or_default();
    let contains = |heading: &str| {
        inputs["contains"]
            .as_array()
            .is_some_and(|listed| listed.iter().any(|each| each == heading))
    };
    let about = |(word, heading, passage): &(&str, &str, &str)| {
        question.contains(word) && (entry == *heading || contains(heading) || entry.starts_with(passage))
    };
    ANSWERS.iter().any(about).into()
}
