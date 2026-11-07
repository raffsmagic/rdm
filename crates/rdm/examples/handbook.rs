//! Recursion. A tool asks the module again about a piece of what it was given, so the decisions
//! go down through a text that no single request needs to hold. Here: the halving search.
//!
//! ```text
//! cargo run -p rdm --example handbook
//! cargo run -p rdm --example handbook -- --help
//! RDM_POST=1 cargo run -p rdm --example handbook
//! ```
//!
//! The last is the search as the post first writes it, in "A Simple RDM": the whole handbook,
//! and no language model.

mod common;

use common::{Decider, Form, Say, py};
use rdm::lm::{Generate, input_of};
use rdm::systemone::Question;
use rdm::{Json, Params, Rdm, Signature, Tool, inputs};
use serde_json::json;

/// The two halves of a text, by lines. A single line has none.
fn halves(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    let middle = lines.len() / 2;
    if middle == 0 {
        return Vec::new();
    }
    vec![lines[..middle].join("\n"), lines[middle..].join("\n")]
}

/// The questions of the search, which the cell and the post write alike.
fn search() -> Signature {
    Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?")
}

fn main() -> common::Outcome {
    // The form above the cell. It is read before anything is printed, so that `--help` prints only itself.
    let form = Form::new(
        "handbook",
        "With a language model",
        "The halving search again, and at a single line a language model writes the answer.",
    )
    .text(
        "question",
        "Ask the handbook",
        &[
            "What is the nightly hotel limit in London?",
            "How much can I spend on meals per day?",
            "When can I take a taxi?",
            "How often can I claim home office furniture?",
        ],
    )
    .read();

    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    if as_the_post() {
        return post(decision_model, &form.text_of("question"));
    }
    let language_model = common::language_model(writes)?; // the language model the environment asks for
    py::print!(
        "decision model:",
        decision_model.model(),
        "| language model:",
        language_model.model()
    );

    // What the form sets: the question put to the text.
    let question = form.text_of("question");

    let handbook = common::HANDBOOK;
    let section = |heading: &str| handbook.find(heading).ok_or(format!("The handbook has no {heading}."));
    let expenses = &handbook[section("# Expenses")?..section("# Security")?];

    let answer = Generate::parse("passage, question -> answer")?.lm(language_model);

    // Runs on a yes. Asks the same question about each half, one level down: that is the recursion.
    //
    // Down to one line, there is nothing left to halve, and the language model answers from it.
    let look = Tool::new(
        "look",
        "Runs on a yes. Asks the same question about each half, one level down: that is the recursion.\n\n\
         Down to one line, there is nothing left to halve, and the language model answers from it.",
        move |call| {
            let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
            if halves(&text).is_empty() {
                let written = answer.call_in(call, inputs! { "passage" => text, "question" => question })?;
                return Ok(vec![written.read::<String>("answer")?]);
            }
            let mut found = Vec::new();
            for half in halves(&text) {
                let below = call.sub(inputs! { "text" => half, "question" => &question })?;
                found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
            }
            Ok(found)
        },
    );

    let search = Rdm::new(search()).tool(look).max_depth(10).model(decision_model);
    let search = search.params("look", Params::threshold(0.5)); // every cut is a parameter of the module

    common::show_text(expenses); // draws the text, and on it the part each request shows
    common::show_tree(expenses, halves); // draws, in grey, every piece the search could reach
    let result = search.call(inputs! { "text" => expenses, "question" => &question })?;
    common::trace(&result);

    py::print!();
    py::print!(question);
    let found = result.observation("look").cloned();
    py::print!("->", found.unwrap_or(json!("the model said the answer is not in this text")));
    py::print!(result.trajectory.len(), "requests to the decision model, each a yes or a no");
    Ok(())
}

/// Whether to run the search as the post writes it: `RDM_POST` is set to something other than `0`.
fn as_the_post() -> bool {
    std::env::var("RDM_POST").is_ok_and(|value| !value.is_empty() && value != "0")
}

/// The search as the post first writes it, in "A Simple RDM": the whole handbook, every line of
/// it counted, and at one line the line itself is the answer. There is no language model.
fn post(decision_model: Decider, question: &str) -> common::Outcome {
    let look = Tool::new("look", "", |call| {
        let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() <= 1 {
            return Ok(lines.into_iter().map(str::to_owned).collect()); // down to one line, and the model said yes to it
        }
        let middle = lines.len() / 2;
        let mut found: Vec<String> = Vec::new();
        for half in [&lines[..middle], &lines[middle..]] {
            let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
        }
        Ok(found)
    });

    let handbook = common::HANDBOOK; // 71 lines

    let rdm = Rdm::new(search()).tool(look).max_depth(10).model(decision_model);
    let result = rdm.call(inputs! { "text" => handbook, "question" => question })?;
    common::trace(&result);
    py::print!(result.observation("look").cloned().unwrap_or(json!([])));
    py::print!(result.trajectory.len(), "requests, each a yes or a no");
    Ok(())
}

/// What a preset question is about: a word of it, words of the line that states its answer, and
/// the answer as a language model might write it from that line. The first is what the site's
/// own mock writes.
const ANSWERS: [(&str, &str, &str); 4] = [
    ("hotel", "260 euros in London", "260 euros a night in London."),
    ("meals", "45 euros per day", "Up to 45 euros per day when travelling."),
    ("taxi", "Taxis are reimbursed", "When travelling with equipment, or after 22:00."),
    ("furniture", "home office furniture", "Once every two years, up to 400 euros."),
];

/// The stand-in decision model, which answers when the environment asks for no real one: yes
/// to a text that has the line a preset question is answered by, and no to anything else.
fn standin(inputs: &Json, _name: &str, _question: &Question) -> Say {
    let question = inputs["question"].as_str().unwrap_or_default().to_lowercase();
    let text = inputs["text"].as_str().unwrap_or_default();
    let stated = ANSWERS.iter().any(|(word, line, _)| question.contains(word) && text.contains(line));
    stated.into()
}

/// The stand-in language model: the answer to a preset question, from the line that states it.
/// Asked anything else, it gives the passage back.
fn writes(_system: &str, prompt: &str) -> String {
    let question = input_of(prompt, "question").unwrap_or_default().to_lowercase();
    let passage = input_of(prompt, "passage").unwrap_or_default();
    let knows = ANSWERS
        .iter()
        .find(|(word, line, _)| question.contains(word) && passage.contains(line));
    json!({"answer": knows.map_or(passage, |(_, _, written)| written)}).to_string()
}
