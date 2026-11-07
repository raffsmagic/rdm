//! Inputs of any size. A fit takes what is given each request before it is asked. `Fit` cuts
//! down what one request cannot hold, deciding with the same model what to keep: each part it asks
//! about is a sub-call of the run, so it shows in the calls and counts against the limits.
//!
//! ```text
//! cargo run -p rdm --example fit
//! cargo run -p rdm --example fit -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::fit::Fit;
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, inputs};

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the question put to the text.
    let form = Form::new("fit", "Any size", "Fit asks about a long text in parts and keeps the likeliest.")
        .text(
            "question",
            "Ask the handbook",
            &[
                "What is the nightly hotel limit in London?",
                "How many days of annual leave are there?",
                "Who do I tell about a lost laptop?",
            ],
        )
        .read();
    let question = form.text_of("question");

    let handbook = common::HANDBOOK; // 2,555 characters

    let search = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");

    // No request shows more than 900 characters. The text is asked about in parts a third that size,
    // and the likeliest parts are what the question is then put to.
    let covered = Rdm::new(search).fit(Fit::new().max_chars(900).parts(3)).model(decision_model);

    common::show_text(handbook); // draws the text, and on it the part each request shows
    let result = covered.call(inputs! { "text" => handbook, "question" => &question })?;
    common::trace(&result);
    py::print!(question);
    py::print!(
        "stated in the handbook:",
        result.value("look"),
        format!("(p = {:.2})", result.probability("look").unwrap_or_default())
    );
    py::print!(
        result.trajectory.len(),
        "requests, none showing more than 900 characters of the",
        handbook.chars().count()
    );
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one: yes
/// to a text that has the line a preset question is answered by, and no to anything else. The
/// question is `inputs.question` of the search, and of a part `Fit` asks about, among `inputs.questions`.
fn standin(inputs: &Json, _name: &str, _question: &Question) -> Say {
    const ANSWERS: [(&str, &str); 3] = [
        ("hotel", "260 euros in London"),
        ("annual leave", "25 days of annual leave"),
        ("lost", "Report a lost device"),
    ];
    let question = inputs
        .get("question")
        .or(inputs.get("questions"))
        .map(py::str)
        .unwrap_or_default()
        .to_lowercase();
    let text = inputs["text"].as_str().unwrap_or_default();
    let stated = ANSWERS.iter().any(|(word, line)| question.contains(word) && text.contains(line));
    stated.into()
}
