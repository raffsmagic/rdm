//! One request. A request is a signature whose outputs are decisions: the inputs are the state,
//! each output is a question, and every answer comes with a probability. An output named after a
//! tool calls it on a yes, and the tool's arguments are other answers of the same request: the
//! model chooses them, it does not write them.
//!
//! ```text
//! cargo run -p rdm --example triage
//! cargo run -p rdm --example triage -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the message.
    let form = Form::new(
        "triage",
        "One request",
        "State in, typed decisions out. A yes calls the tool of the same name, with an argument the model chose.",
    )
    .text_named(
        "message",
        "A message from a customer",
        &[
            (
                "Charged twice",
                "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
            ),
            (
                "Logged out",
                "The app logs me out every time I open the settings page. Otherwise I love it!",
            ),
            ("Team pricing", "Could you tell me what the team plan costs for twelve people?"),
            ("Thanks", "Just wanted to say thanks, the new export is great."),
        ],
    )
    .read();
    let message = form.text_of("message");

    // Open a refund with a team. `team` is an answer of the request, and `message` one of its inputs.
    let refund = Tool::new(
        "refund",
        "Open a refund with a team. `team` is an answer of the request, and `message` one of its inputs.",
        |call| {
            let (team, _message): (String, String) = (call.arg("team")?, call.arg("message")?);
            Ok(format!("refund opened with {team}"))
        },
    );

    let triage = Signature::new("Triage a message from a customer.")
        .input("message")
        .noul("refund", "Is the customer in `inputs.message` asking for a refund?")
        .choice(
            "team",
            "Which team should handle `inputs.message`?",
            ["billing", "technical support", "sales", "account management"],
        )
        .noul("upset", "Is the customer in `inputs.message` upset?");

    // With no tools, `Rdm::new(triage)` is one predict call: one request, and its answers.
    let triage = Rdm::new(triage).tool(refund).model(decision_model);

    let result = triage.call(inputs! { "message" => message })?; // one request, three questions
    common::trace(&result);
    let p = |name| result.probability(name).unwrap_or_default();
    py::print!("refund:", result.value("refund"), format!("(p = {:.2})", p("refund")));
    py::print!("upset: ", result.value("upset"), format!("(p = {:.2})", p("upset")));
    if result.observations.is_empty() {
        py::print!("no answer selected a tool, so nothing ran");
    } else {
        py::print!(result.observations);
    }
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one: it
/// reads the message for a few words, as a rule and not as a model would.
fn standin(inputs: &Json, name: &str, _question: &Question) -> Say {
    let message = inputs["message"].as_str().unwrap_or_default().to_lowercase();
    let says = |words: &[&str]| words.iter().any(|word| message.contains(word));
    match name {
        "refund" => says(&["money back", "refund"]).into(),
        "upset" => says(&["nobody answers", "money back"]).into(),
        _ if says(&["charged", "refund", "invoice"]) => "billing".into(),
        _ if says(&["logs me out", "error", "crash"]) => "technical support".into(),
        _ if says(&["cost", "price", "plan"]) => "sales".into(),
        _ => "account management".into(),
    }
}
