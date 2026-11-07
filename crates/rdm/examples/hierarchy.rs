//! Choosing the next question. A sub-call can put other questions: a signature given with it
//! replaces the module's own for that call, so each answer decides which options the next
//! question has.
//!
//! ```text
//! cargo run -p rdm --example hierarchy
//! cargo run -p rdm --example hierarchy -- --help
//! cargo run -p rdm --example hierarchy -- --listing 4
//! RDM_POST=1 cargo run -p rdm --example hierarchy -- --listing 4
//! ```
//!
//! The fourth listing is the post's own, and the last is the search as the post writes it, in
//! "Hierarchical search, without the beam": plain recursion around one request, with no tool.

mod common;

use std::sync::Arc;

use common::{Decider, Failure, Form, Say, py};
use indexmap::IndexMap;
use rdm::filter::Filter;
use rdm::systemone::Question;
use rdm::{Ask, Json, Output, Rdm, Signature, Tool, inputs};

/// `{category: [the categories directly under it]}`.
type Taxonomy = IndexMap<String, Vec<String>>;

/// `{category: how likely it is}`.
type Odds = IndexMap<String, f64>;

/// One request: which of these categories fits the listing best?
fn ask(options: &[String]) -> rdm::Result<Signature> {
    Ok(Signature::parse("listing -> category")?.with_output(
        "category",
        Output::choice("Which category best matches the product in `inputs.listing`?", options),
    ))
}

/// The last name of a path: `path.rpartition(" > ")[2]`.
fn name(path: &str) -> &str {
    path.rsplit_once(" > ").map_or(path, |(_, name)| name)
}

fn main() -> common::Outcome {
    // The form above the cell. It is read before anything is printed, so that `--help` prints only itself.
    let form = Form::new(
        "hierarchy",
        "Hierarchical search",
        "A walk down Shopify's product taxonomy, following every category at least half as likely as the likeliest.",
    )
    .text_named(
        "listing",
        "A product listing",
        // The first three are ones the small local model finds its way through; the last two want the hosted models.
        &[
            ("Laptop backpack", "Laptop backpack with a padded sleeve, 20 litres."),
            (
                "Board game",
                "A board game for families: a strategy game for 2 to 4 players.",
            ),
            ("Hiking backpack", "Hiking backpack, 40 litres, with a rain cover."),
            (
                "Cat window perch",
                "Furniture listing: a wall-mounted window shelf bed. This padded floating shelf uses suction cups and a washable cushion as a sunny perch for one cat.",
            ),
            (
                "Headphones",
                "Wireless over-ear headphones with active noise cancelling, 30 hours of battery and a folding case.",
            ),
        ],
    )
    .read();

    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    if as_the_post() {
        return post(decision_model, &taxonomy()?, &form.text_of("listing"));
    }
    py::print!("decision model:", decision_model.model());

    // What the form sets: the product to find a category for.
    let listing = form.text_of("listing");

    // The external state: Shopify's product taxonomy, over twelve thousand categories that no request shows whole.
    let taxonomy = Arc::new(taxonomy()?);
    py::print!(taxonomy.len() - 1, "categories");

    let likeliest = Filter::new(); // sorts what a decision rated: every option of a choice, not only the one chosen

    // Runs on every answer. Follows each category at least half as likely as the likeliest, by
    // asking the module again about the categories under it: that is the recursion. There is no
    // beam width and no path score. Returns the leaves reached.
    let category = {
        let taxonomy = taxonomy.clone();
        Tool::new(
            "category",
            "Runs on every answer. Follows each category at least half as likely as the likeliest, by\n\
             asking the module again about the categories under it: that is the recursion. There is no\n\
             beam width and no path score. Returns the leaves reached.",
            move |call| {
                let listing: String = call.arg("listing")?;
                let rated = likeliest.options(call.answer("category")); // (category, probability) pairs, likeliest first
                let likely: Vec<&String> = rated
                    .iter()
                    .filter(|(_, probability)| *probability >= rated[0].1 / 2.0)
                    .map(|(each, _)| each)
                    .collect();
                common::show_branch(&rated, &likely); // draws this step
                // Where the page draws the step, it is printed here as the post prints it: the category
                // whose categories were on offer, and the ones followed.
                let above = rated[0].0.rsplit_once(" > ").map_or("All products", |(above, _)| above);
                py::print!(name(above), "->", likely.iter().map(|each| name(each)).collect::<Vec<_>>());
                let mut leaves: Vec<String> = Vec::new();
                for each in likely {
                    let under = taxonomy.get(each).map(Vec::as_slice).unwrap_or_default();
                    if under.is_empty() {
                        leaves.push(each.clone());
                        continue;
                    }
                    let below = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(ask(under)?))?;
                    leaves.extend(below.observed::<Vec<String>>("category").unwrap_or_default());
                }
                Ok(leaves)
            },
        )
    };

    let search = Rdm::new(ask(&taxonomy["All products"])?)
        .tool(category)
        .max_depth(12)
        .max_predictor_calls(60)
        .model(decision_model);

    let result = search.call(inputs! { "listing" => &listing })?;
    common::trace(&result);
    let found: Vec<String> = result
        .observed("category")
        .ok_or_else(|| Failure::new("The search came to no category: its first request ran no tool."))?;
    // More than one leaf came back: ask once more, among those.
    let best = if found.len() == 1 {
        found[0].clone()
    } else {
        let among = search.call_with(Ask::new(inputs! { "listing" => &listing }).signature(ask(&found)?).max_depth(0))?;
        common::trace(&among);
        py::str(&among.value("category"))
    };

    common::show_leaf(&best);
    py::print!();
    py::print!(found.len(), if found.len() == 1 { "leaf" } else { "leaves" }, "reached");
    py::print!(best);
    Ok(())
}

/// Shopify's product taxonomy: `{category: [the categories directly under it]}`.
///
/// The cell fetches it from
/// `https://raw.githubusercontent.com/Shopify/product-taxonomy/v2026-02/dist/en/categories.txt`.
/// That file is kept beside the examples, in `assets/`, so that no run needs the network for it.
fn taxonomy() -> Result<Taxonomy, Failure> {
    let mut taxonomy = Taxonomy::from([("All products".to_owned(), Vec::new())]);
    for line in common::CATEGORIES.lines().skip(3) {
        // "Animals & Pet Supplies > Pet Supplies > Cat Supplies"
        let path = line.split(" : ").nth(1).ok_or(format!("Not a category: {line}"))?;
        let above = path.rsplit_once(" > ").map_or("All products", |(above, _)| above);
        let under = taxonomy
            .get_mut(above)
            .ok_or(format!("A category before the one it is under: {path}"))?;
        under.push(path.to_owned());
        taxonomy.insert(path.to_owned(), Vec::new());
    }
    Ok(taxonomy)
}

/// Whether to run the search as the post writes it: `RDM_POST` is set to something other than `0`.
fn as_the_post() -> bool {
    std::env::var("RDM_POST").is_ok_and(|value| !value.is_empty() && value != "0")
}

/// The search as the post writes it, in "Hierarchical search, without the beam": two functions
/// around one request, and no tool. `choose` asks how likely each category is, and `classify`
/// follows every one at least half as likely as the likeliest by calling itself; when more than
/// one leaf comes back it asks again among those, at every level.
fn post(decision_model: Decider, taxonomy: &Taxonomy, listing: &str) -> common::Outcome {
    let predict = Rdm::new(Signature::parse("listing -> category")?).model(decision_model); // with no tools, one request

    // One request: how likely each category is to be the best match for the listing.
    let choose = |options: &[String]| -> rdm::Result<Odds> {
        if options.len() == 1 {
            return Ok(Odds::from([(options[0].clone(), 1.0)]));
        }
        let asked = predict.call_with(Ask::new(inputs! { "listing" => listing }).signature(ask(options)?))?;
        common::trace(&asked);
        Ok(asked
            .choice("category")
            .map(|category| category.probabilities.clone())
            .unwrap_or_default())
    };

    fn classify(category: &str, taxonomy: &Taxonomy, choose: &dyn Fn(&[String]) -> rdm::Result<Odds>) -> rdm::Result<String> {
        let under = taxonomy.get(category).map(Vec::as_slice).unwrap_or_default();
        if under.is_empty() {
            return Ok(category.to_owned()); // a leaf
        }
        let odds = choose(under)?;
        let of = |odds: &Odds, each: &String| odds.get(each).copied().unwrap_or(0.0);
        let most = odds.values().copied().fold(0.0, f64::max);
        let likely: Vec<&String> = under.iter().filter(|each| of(&odds, each) >= most / 2.0).collect();
        py::print!(name(category), "->", likely.iter().map(|each| name(each)).collect::<Vec<_>>());
        let mut found = Vec::new();
        for each in likely {
            found.push(classify(each, taxonomy, choose)?); // the same function, one level down
        }
        let odds = choose(&found)?; // more than one came back: ask again, among those
        let mut best = &found[0];
        for each in &found {
            if of(&odds, each) > of(&odds, best) {
                best = each;
            }
        }
        Ok(best.clone())
    }

    py::print!(classify("All products", taxonomy, &choose)?);
    Ok(())
}

/// The stand-in decision model, which answers when the environment asks for no real one. It
/// knows where each preset listing belongs, by a word of it, and chooses the category on the
/// way there: for a listing it does not know, `Uncategorized`, or else the first on offer.
///
/// The post's listing says "bed", and for it the stand-in is of two minds where the post's own
/// run was: `Pet Beds` is likely too, and under it four beds as likely as each other.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    const LEAVES: [(&str, &str); 5] = [
        ("laptop", "Luggage & Bags > Backpacks > Laptop Backpacks"),
        ("hiking", "Luggage & Bags > Backpacks > Hiking Backpacks"),
        ("board game", "Toys & Games > Games > Board Games"),
        (
            "cat",
            "Animals & Pet Supplies > Pet Supplies > Cat Supplies > Cat Furniture > Cat Window Beds & Perches",
        ),
        (
            "headphones",
            "Electronics > Audio > Audio Components > Headphones & Headsets > Headphones > Over-Ear Headphones",
        ),
    ];
    const BEDS: &str = "Animals & Pet Supplies > Pet Supplies > Pet Beds";
    const ALSO: [&str; 4] = ["Hammocks", "Pet Cots", "Pillow Beds", "Radiator Beds"];

    let listing = inputs["listing"].as_str().unwrap_or_default().to_lowercase();
    let options = question.labels();
    let leaf = LEAVES.iter().find(|(word, _)| listing.contains(word)).map_or("", |(_, leaf)| leaf);
    // The option on the way to the leaf, and the options it could be as well.
    let way = options
        .iter()
        .find(|option| leaf == *option || leaf.starts_with(&format!("{option} > ")));
    let bed = |option: &&String| *option == BEDS || ALSO.iter().any(|bed| **option == format!("{BEDS} > {bed}"));
    let also: Vec<&String> = options.iter().filter(bed).filter(|_| listing.contains("bed")).collect();
    if !also.is_empty() {
        let share = if way.is_some() { 0.4 } else { 1.0 } / also.len() as f64;
        let mut odds: Vec<(String, f64)> = way.map(|way| (way.clone(), 0.6)).into_iter().collect();
        odds.extend(also.into_iter().map(|option| (option.clone(), share)));
        return odds.into();
    }
    let unknown = options.iter().find(|option| *option == "Uncategorized");
    way.or(unknown).or(options.first()).cloned().unwrap_or_default().into()
}
