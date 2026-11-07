//! Recursion: a tool that asks the module again. Calling `call.sub` inside a tool is a sub-call:
//! one level deeper, in the same run. Here a yes about a text asks the same question about each
//! half of it, down to one line. The text is nineteen aphorisms, small so the run can be read.
//!
//! ```text
//! cargo run -p rdm --example zen
//! cargo run -p rdm --example zen -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};
use serde_json::json;

/// The Zen of Python: what `this.s` is in Python, once it is decoded.
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

/// The two halves of a text, by lines. A single line has none.
fn halves(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    let middle = lines.len() / 2;
    if middle == 0 {
        return Vec::new();
    }
    vec![lines[..middle].join("\n"), lines[middle..].join("\n")]
}

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the question put to the text.
    let form = Form::new(
        "zen",
        "Recursion",
        "A tool asks the module again: a yes about a text asks the same question of each half.",
    )
    .text(
        "question",
        "Ask the Zen of Python",
        &[
            "Is flat better than nested?",
            "What does the text say about namespaces?",
            "What is one honking great idea?",
            "How many obvious ways should there be to do something?",
        ],
    )
    .read();
    let question = form.text_of("question");

    let zen = ZEN;
    println!("{zen}"); // importing `this` prints the Zen of Python, the first time

    let search = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");

    // Runs on a yes. Asks the same question about each half, one level down: that is the recursion.
    let look = Tool::new(
        "look",
        "Runs on a yes. Asks the same question about each half, one level down: that is the recursion.",
        |call| {
            let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
            if halves(&text).is_empty() {
                return Ok(vec![text]); // one line, and the model said yes to it
            }
            let mut found = Vec::new();
            for half in halves(&text) {
                let below = call.sub(inputs! { "text" => half, "question" => &question })?;
                found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
            }
            Ok(found)
        },
    );

    let search = Rdm::new(search).tool(look).max_depth(6).model(decision_model);

    common::show_text(zen); // draws the text, and on it the part each request shows
    common::show_tree(zen, halves); // draws, in grey, every piece the search could reach
    let result = search.call(inputs! { "text" => zen, "question" => &question })?;
    common::trace(&result);

    py::print!();
    py::print!(question);
    let found = result.observation("look").cloned();
    py::print!("->", found.unwrap_or(json!("the model said the answer is not in this text")));
    py::print!(result.trajectory.len(), "requests, each a yes or a no");
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one: yes
/// to a text that has the line a preset question is answered by, and no to anything else. The
/// last of them is the rule the Python tests its reference page with.
fn standin(inputs: &Json, _name: &str, _question: &Question) -> Say {
    const ANSWERS: [(&str, &str); 4] = [
        ("flat", "Flat is better than nested."),
        ("namespaces", "Namespaces are one honking great idea"),
        ("honking", "Namespaces are one honking great idea"),
        ("obvious", "preferably only one"),
    ];
    let question = inputs["question"].as_str().unwrap_or_default().to_lowercase();
    let text = inputs["text"].as_str().unwrap_or_default();
    let stated = ANSWERS.iter().any(|(word, line)| question.contains(word) && text.contains(line));
    stated.into()
}
