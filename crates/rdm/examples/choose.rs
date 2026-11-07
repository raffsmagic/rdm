//! Choosing the next question. A sub-call can put other questions: a signature given with it
//! replaces the module's own for that call. Here each answer decides which options the next
//! question has: the category chosen at one level is the one whose categories are offered at
//! the next.
//!
//! ```text
//! cargo run -p rdm --example choose
//! cargo run -p rdm --example choose -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::systemone::Question;
use rdm::{Ask, Json, Output, Rdm, Signature, Tool, inputs};
use serde_json::json;

/// `{category: [the categories directly under it]}`.
const TAXONOMY: [(&str, &[&str]); 6] = [
    ("Products", &["Animals & Pet Supplies", "Furniture", "Kitchen"]),
    ("Animals & Pet Supplies", &["Cat Supplies", "Dog Supplies"]),
    ("Cat Supplies", &["Cat Beds", "Cat Toys", "Cat Window Perches"]),
    ("Dog Supplies", &["Dog Beds", "Dog Toys", "Dog Leashes"]),
    ("Furniture", &["Shelves", "Beds", "Sofas"]),
    ("Kitchen", &["Cookware", "Tableware", "Kitchen Knives"]),
];

/// `taxonomy[category]`: the categories under one, and none for a category that is not in it.
fn taxonomy(category: &str) -> Option<&'static [&'static str]> {
    TAXONOMY.iter().find(|(name, _)| *name == category).map(|(_, under)| *under)
}

/// One request: which category under this one fits the listing?
fn ask(category: &str) -> rdm::Result<Signature> {
    let options = taxonomy(category).unwrap_or_default().iter().map(|child| (*child, ""));
    Ok(Signature::parse("listing -> category")?.with_output(
        "category",
        Output::choice("Which category best matches the product in `inputs.listing`?", options),
    ))
}

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the product to find a category for.
    let form = Form::new(
        "choose",
        "Choosing the next question",
        "Each answer decides which options the next question has: a greedy walk down a small taxonomy.",
    )
    .text_named(
        "listing",
        "A product listing",
        &[
            (
                "Cat window shelf",
                "A padded shelf that suction-cups to a window, so a cat can sun itself.",
            ),
            ("Dog tug toy", "A rope tug toy for large dogs."),
            ("Chef's knife", "A chef's knife with a 20 cm blade."),
            ("Sofa", "A three-seat sofa in grey fabric."),
        ],
    )
    .read();
    let listing = form.text_of("listing");

    // Runs on every answer, and is given it by name. Puts the question under the category chosen,
    // one level down, and comes to the leaf it ends at.
    let category = Tool::new(
        "category",
        "Runs on every answer, and is given it by name. Puts the question under the category chosen,\n\
         one level down, and comes to the leaf it ends at.",
        |call| {
            let (category, listing): (String, String) = (call.arg("category")?, call.arg("listing")?);
            if taxonomy(&category).is_none() {
                return Ok(json!(category)); // a leaf
            }
            let below = call.sub_with(Ask::new(inputs! { "listing" => listing }).signature(ask(&category)?))?;
            Ok(below.observations["category"].clone())
        },
    );

    let classify = Rdm::new(ask("Products")?).tool(category).model(decision_model);

    let result = classify.call(inputs! { "listing" => listing })?;
    common::trace(&result);
    // The call's own answer is its own choice: the first. What its tool came to is in `observations`.
    py::print!("first choice:", result.value("category"));
    py::print!("leaf:        ", result.observations["category"]);
    py::print!(
        result.trajectory.len(),
        "requests. The walk is greedy: one wrong choice near the top is not recovered from."
    );
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one. It
/// knows the words of a listing that put it in a category, and chooses the first category on
/// offer that has one of its words in the listing: the first of all, when none has.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    const WORDS: [(&str, &[&str]); 17] = [
        ("Animals & Pet Supplies", &["cat", "dog"]),
        ("Furniture", &["sofa", "shelf", "bed"]),
        ("Kitchen", &["knife", "pan", "plate"]),
        ("Cat Supplies", &["cat"]),
        ("Dog Supplies", &["dog"]),
        ("Cat Beds", &["bed"]),
        ("Cat Toys", &["toy"]),
        ("Cat Window Perches", &["window"]),
        ("Dog Beds", &["bed"]),
        ("Dog Toys", &["toy"]),
        ("Dog Leashes", &["leash"]),
        ("Shelves", &["shelf"]),
        ("Beds", &["bed"]),
        ("Sofas", &["sofa"]),
        ("Cookware", &["pan"]),
        ("Tableware", &["plate"]),
        ("Kitchen Knives", &["knife"]),
    ];
    let listing = inputs["listing"].as_str().unwrap_or_default().to_lowercase();
    let fits = |option: &String| {
        let words = WORDS.iter().find(|(category, _)| category == option).map(|(_, words)| *words);
        words.unwrap_or_default().iter().any(|word| listing.contains(word))
    };
    let options = question.labels();
    options
        .iter()
        .find(|option| fits(option))
        .or(options.first())
        .cloned()
        .unwrap_or_default()
        .into()
}
