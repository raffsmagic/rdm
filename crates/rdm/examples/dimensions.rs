//! A map with dimensions: embeddings you can read. Kieran Klaassen's idea for a decision model:
//! in place of an embedding model's 1,536 numbers that mean nothing, ask questions about each
//! thing, and the answers are its vector. Every number has a name, so it can be read why two
//! things are near. `Near` does this to a map: each name is asked the questions once, each
//! request is asked the same ones, and the pointer is offered only the names nearest it.
//! The idea: https://x.com/kieranklaassen/status/2103522599414501550
//!
//! ```text
//! cargo run -p rdm --example dimensions
//! cargo run -p rdm --example dimensions -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::near::{self, Coordinates, Measure, Near, coordinates};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the email.
    let form = Form::new(
        "dimensions",
        "A map with dimensions",
        "Embeddings you can read: each name on the map answers a few questions, and a request is offered the names nearest it.",
    )
    .text_named(
        "email",
        "An email to a support desk",
        &[
            ("Charged twice", "I got charged twice this month, pls fix asap"),
            ("Can't log in", "I can't log in, it says my password is wrong."),
            ("Export is broken", "The export button does nothing since yesterday."),
            ("Team pricing", "What does the team plan cost for twelve people?"),
        ],
    )
    .read();
    let email = form.text_of("email");

    let handle = Signature::new("Handle an email to a support desk, one step at a time.")
        .input("email")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should be done next about `inputs.email`?");

    /// One name on the map: a description for the model, and a thing for the code to do.
    fn step(name: &'static str, desc: &'static str) -> Tool {
        Tool::new(name, desc, move |_call| Ok(format!("{name}: done")))
    }

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

    // Of the eight names, a request is offered the three nearest it, with `finish`. A module's
    // model does not reach its fit, so `near` is given the model too, for when it is used alone.
    let near = Near::new(about).top(3).measure(Measure::Cosine).model(decision_model.clone());
    let desk = Rdm::new(handle).tools_from(steps.clone()).fit(near.clone()).model(decision_model);

    let result = desk.call(inputs! { "email" => &email })?;
    common::trace(&result);
    py::print!(email, "->", result.pointed("next"));
    py::print!();

    // The embedding, which is part of the program: what each text answered, by the name of the dimension.
    for (text, answered) in near.answers() {
        let labels: Vec<String> = answered.iter().map(|(name, value)| format!("{name} {value:.2}")).collect();
        let name: String = text.split(':').next().unwrap_or_default().chars().take(44).collect();
        py::print!(labels.join(" "), "|", name);
    }
    py::print!();

    // Used alone it is a search by named dimensions: the vector of what is wanted is written by hand, and can be read.
    let wanted = coordinates([("money", 1.0), ("access", 0.0), ("fault", 0.0), ("buying", 0.5)]);
    let names: Vec<String> = steps.iter().map(|tool| near::text(tool.name(), tool.description())).collect();
    let mut ranked: Vec<(&String, Coordinates)> = names.iter().zip(near.embed(&names)?).collect();
    ranked.sort_by(|one, other| near.score(&wanted, &other.1).total_cmp(&near.score(&wanted, &one.1)));
    py::print!("nearest to", wanted);
    for (name, vector) in &ranked[..3] {
        py::print!(
            format!("  {:.2}", near.score(&wanted, vector)),
            name.split(':').next().unwrap_or_default()
        );
    }
    Ok(())
}

/// How far a text is about each dimension, by the words in it: a yes for most, and "partly"
/// where a word belongs to two (a price is about buying, and somewhat about money).
const ABOUT: [(&str, &[(&str, f64)]); 4] = [
    (
        "money",
        &[("charge", 0.93), ("refund", 0.93), ("invoice", 0.93), ("price", 0.5), ("cost", 0.5)],
    ),
    ("access", &[("password", 0.93), ("log in", 0.93), ("login", 0.93)]),
    ("fault", &[("fault", 0.93), ("bug", 0.93), ("does nothing", 0.93), ("outage", 0.8)]),
    ("buying", &[("price", 0.93), ("plan", 0.93), ("cost", 0.93), ("sales", 0.93)]),
];

/// The step an email asks for, by a word in it.
const STEPS: [(&str, &str); 9] = [
    ("charge", "refund"),
    ("invoice", "send invoice"),
    ("password", "reset password"),
    ("locked out", "unlock account"),
    ("does nothing", "file a bug"),
    ("outage", "check status page"),
    ("cost", "quote a plan"),
    ("price", "quote a plan"),
    ("demo", "book a demo"),
];

/// The stand-in decision model, which answers when the environment asks for no real one.
///
/// A dimension is answered by the words of the text: the likeliest of those it has, and 0.04
/// with none. Each name on the map is a text too, asked as `name: description`, so what the
/// table of the run shows is these rules read off the eight descriptions and the email.
///
/// To point, it takes the step the email has a word for, once, if that step is among the three
/// it was offered, and then stops.
fn standin(inputs: &Json, name: &str, question: &Question) -> Say {
    let email = inputs["email"].as_str().unwrap_or_default().to_lowercase();
    if let Some((_, words)) = ABOUT.iter().find(|(dimension, _)| *dimension == name) {
        let has = words.iter().filter(|(word, _)| email.contains(word));
        return has.map(|(_, likely)| *likely).fold(0.04, f64::max).into();
    }
    let step = STEPS.iter().find(|(word, _)| email.contains(word)).map(|(_, step)| *step);
    let first = inputs["observation"].as_str().is_some_and(str::is_empty);
    match step {
        Some(step) if first && question.labels().iter().any(|offered| offered == step) => step.into(),
        _ => "finish".into(),
    }
}
