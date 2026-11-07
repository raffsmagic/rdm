//! Sorting and filtering. `Filter` sorts what the model rated by its probabilities, likeliest
//! first, and keeps the likeliest. Here a tool follows the two likeliest categories at every level,
//! not only the one chosen: a beam with no score, where every leaf the two likeliest reach comes back.
//!
//! ```text
//! cargo run -p rdm --example filter
//! cargo run -p rdm --example filter -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::filter::Filter;
use rdm::systemone::Question;
use rdm::{Ask, Json, Output, Rdm, Signature, Tool, inputs};

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
        "filter",
        "Sorting and filtering",
        "Filter keeps the two likeliest at every level: a beam with no score.",
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

    let likeliest = Filter::new().top(2); // every option of a choice with its probability, likeliest first, and the two likeliest kept

    // Follow the two likeliest categories, and come to every leaf they reach.
    let category = Tool::new(
        "category",
        "Follow the two likeliest categories, and come to every leaf they reach.",
        move |call| {
            let listing: String = call.arg("listing")?;
            let mut leaves: Vec<Json> = Vec::new();
            // The answer with its probabilities, not only its value.
            for (option, _probability) in likeliest.options(call.answer("category")) {
                if taxonomy(&option).is_none() {
                    leaves.push(option.into());
                    continue;
                }
                let below = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(ask(&option)?))?;
                leaves.extend(below.observations["category"].as_array().cloned().unwrap_or_default());
            }
            Ok(leaves)
        },
    );

    let wide = Rdm::new(ask("Products")?).tool(category).model(decision_model);

    let result = wide.call(inputs! { "listing" => listing })?;
    common::trace(&result);
    py::print!("leaves reached:", result.observations["category"]);
    py::print!(result.trajectory.len(), "requests");
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one. It
/// knows the words of a listing that put it in a category, and chooses the first category on
/// offer that has one of its words in the listing: the first of all, when none has. The one it
/// chooses is at 0.95 and the others share the rest, so the second likeliest is the first of them.
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
