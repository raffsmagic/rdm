#![cfg(feature = "filter")]
//! Filter: sorting what a decision model rated by its probabilities, and keeping the likeliest.

use std::sync::Arc;

use indexmap::IndexMap;
use rdm::filter::Filter;
use rdm::systemone::{Answer, Choice, Decision, Noul, Params, Scripted};
use rdm::{Ask, Output, Prediction, Rdm, Signature, Tool, inputs};
use serde_json::json;

fn things<T: Clone>(kept: &[(T, f64)]) -> Vec<T> {
    kept.iter().map(|(thing, _)| thing.clone()).collect()
}

#[test]
fn it_sorts_by_probability_and_keeps_the_likeliest() {
    // With nothing set it only sorts. What it gives is every thing with its probability, likeliest first.
    let rated = IndexMap::from([("east", 0.1), ("north", 0.5), ("south", 0.4)]);
    assert_eq!(Filter::new().rated(rated.clone()), [("north", 0.5), ("south", 0.4), ("east", 0.1)]);
    // A map that is only lent is read the same: the things are then its keys, lent.
    assert_eq!(things(&Filter::new().rated(&rated)), [&"north", &"south", &"east"]);
    // top: so many at most, a setting like the width of a beam.
    assert_eq!(Filter::new().top(2).rated(rated), [("north", 0.5), ("south", 0.4)]);
    // Things rated alike keep the order they were given in.
    assert_eq!(things(&Filter::new().rated([("a", 0.3), ("b", 0.3), ("c", 0.9)])), ["c", "a", "b"]);
    assert_eq!(Filter::new().top(2).rated(Vec::<(&str, f64)>::new()), []);
    // A probability that is not a number is not likelier than one that is.
    assert_eq!(
        things(&Filter::new().rated([("a", f64::NAN), ("b", 0.2), ("c", 0.7)])),
        ["c", "b", "a"]
    );
}

#[test]
#[should_panic(expected = "top must be at least 1")]
fn a_filter_that_keeps_nothing_is_refused() {
    let _ = Filter::new().top(0);
}

fn noul(value: bool, probability: f64) -> Noul {
    Noul {
        confidence: 0.5,
        value,
        probability,
    }
}

fn way() -> Choice {
    Choice {
        confidence: 0.5,
        value: "north".to_owned(),
        probabilities: IndexMap::from([
            ("north".to_owned(), 0.5),
            ("south".to_owned(), 0.3),
            ("east".to_owned(), 0.05),
            ("None".to_owned(), 0.15),
        ]),
    }
}

#[test]
fn it_reads_what_was_rated_from_the_answers_a_model_gave() {
    // A choice: every option with its probability, each by its label.
    let answer = way();
    assert_eq!(
        Filter::new().top(2).options(&answer),
        [("north".to_owned(), 0.5), ("south".to_owned(), 0.3)]
    );
    // The option of choosing none of the others is one of them, under its label.
    assert_eq!(
        Filter::new().options(&answer)[2..],
        [(rdm::NONE.to_owned(), 0.15), ("east".to_owned(), 0.05)]
    );
    // It is read wherever a tool has it: as a decision, or as an answer that may be absent.
    let decision = Decision::Choice(answer.clone());
    assert_eq!(Filter::new().options(&decision), Filter::new().options(&answer));
    assert_eq!(Filter::new().options(Some(&decision)), Filter::new().options(&answer));
    assert_eq!(Filter::new().options(None::<&Decision>), []);
    assert_eq!(Filter::new().options(Decision::Noul(noul(true, 0.9))), []);

    // A prediction: its yes or no answers that are a yes, by the name of the output.
    let answers = IndexMap::from([
        ("holds_0".to_owned(), Some(Decision::Noul(noul(false, 0.2)))),
        ("holds_1".to_owned(), Some(Decision::Noul(noul(true, 0.9)))),
        ("holds_2".to_owned(), None),
        ("way".to_owned(), Some(decision)),
    ]);
    let prediction = Prediction::of(&Signature::new("Hold."), answers);
    assert_eq!(Filter::new().yeses(&prediction), [("holds_1".to_owned(), 0.9)]);
    assert_eq!(Filter::new().options(prediction.choice("way")).len(), 4);

    // Pairs: anything, each with the yes or no given about it. Only the yeses are kept.
    let records = [json!({"id": 1}), json!({"id": 2}), json!({"id": 3})];
    let verdicts = [noul(false, 0.1), noul(true, 0.7), noul(true, 0.6)];
    let kept = Filter::new().rated(records.iter().zip(&verdicts));
    assert_eq!(kept, [(&json!({"id": 2}), 0.7), (&json!({"id": 3}), 0.6)]);
    // What a yes is was decided when the answer was: a yes below one half is still a yes.
    assert_eq!(Filter::new().rated([("a", noul(true, 0.4)), ("b", noul(false, 0.6))]), [("a", 0.4)]);
    // An answer that is absent, as that of a request a limit stopped is, says yes of nothing.
    assert_eq!(Filter::new().rated([("a", None), ("b", Some(noul(true, 0.8)))]), [("b", 0.8)]);
}

#[test]
fn a_choice_that_came_with_no_probabilities_gives_the_one_option_chosen_as_certain() {
    // The Python module documents this and fails on it. Here it is what the documentation says.
    let bare = Choice {
        confidence: 1.0,
        value: "north".to_owned(),
        probabilities: IndexMap::new(),
    };
    assert_eq!(Filter::new().top(2).options(&bare), [("north".to_owned(), 1.0)]);
}

fn tree(node: &str) -> Option<&'static [&'static str]> {
    Some(match node {
        "root" => &["pets", "furniture", "garden"],
        "pets" => &["cat bed", "dog bed"],
        "furniture" => &["shelf", "sofa"],
        "garden" => &["hose", "spade"],
        _ => return None,
    })
}

fn under(node: &str) -> Signature {
    let options = tree(node).unwrap_or_default().iter().copied();
    Signature::parse("listing -> category")
        .unwrap()
        .with_output("category", Output::choice("Which category fits `inputs.listing`?", options))
}

fn fits() -> Signature {
    Signature::new("Judge one leaf.")
        .input("listing")
        .input("leaf")
        .noul("fits", "Does `inputs.leaf` fit `inputs.listing`?")
}

/// Follow the two likeliest categories, each to its leaves; then keep the leaves that fit, likeliest first.
fn category() -> Tool {
    Tool::new("category", "Follow the two likeliest categories.", |call| {
        let listing: String = call.arg("listing")?;
        let mut leaves: Vec<String> = Vec::new();
        for (option, _) in Filter::new().top(2).options(call.answer("category")) {
            if tree(&option).is_none() {
                leaves.push(option);
                continue;
            }
            let below = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(under(&option)))?;
            leaves.extend(below.observed::<Vec<String>>("category").unwrap_or_default());
        }
        if call.depth() > 0 {
            return Ok(json!(leaves));
        }
        let judged = leaves.iter().map(|leaf| {
            Ask::new(inputs! { "listing" => &listing, "leaf" => leaf })
                .signature(fits())
                .max_depth(0)
        });
        let verdicts = call.batch(judged)?;
        Ok(json!(
            Filter::new().rated(leaves.iter().zip(verdicts.iter().map(|verdict| verdict.noul("fits"))))
        ))
    })
}

/// A stand-in that rates every option of a choice, and says of each leaf how well it fits.
fn odds(fits: &'static [(&'static str, f64)]) -> Arc<Scripted> {
    let rated = |option: &str| match option {
        "pets" => 0.5,
        "furniture" => 0.4,
        "garden" => 0.1,
        "cat bed" => 0.8,
        "dog bed" => 0.2,
        "shelf" => 0.7,
        "sofa" => 0.3,
        "hose" => 0.6,
        _ => 0.4,
    };
    Arc::new(Scripted::new(move |request| {
        let leaf = request.state["inputs"]["leaf"].as_str().unwrap_or_default().to_owned();
        let answer = |name: &String| {
            if name == "fits" {
                return Answer::noul(fits.iter().find(|(known, _)| *known == leaf).map_or(0.1, |(_, p)| *p));
            }
            Answer::choice(request.questions[name].labels().into_iter().map(|label| {
                let probability = rated(&label);
                (label, probability)
            }))
        };
        request
            .questions
            .keys()
            .map(|name| (name.clone(), answer(name)))
            .collect::<Vec<_>>()
    }))
}

#[test]
fn a_tool_follows_the_likeliest_options_and_keeps_what_a_batch_said_yes_to() {
    let model = odds(&[("cat bed", 0.9), ("shelf", 0.6)]);
    let rdm = Rdm::new(under("root")).tool(category()).model(model.clone());
    let result = rdm.call(inputs! { "listing" => "A padded window shelf for a cat." }).unwrap();
    // Two of the three branches were followed, each to its two likeliest leaves, and the third, the garden,
    // was not. Of those four leaves, the two said yes of were kept, likeliest first.
    let requests = model.requests();
    let offered: Vec<Vec<String>> = requests
        .iter()
        .filter_map(|request| Some(request.questions.get("category")?.labels()))
        .collect();
    assert_eq!(
        offered,
        [
            vec!["pets", "furniture", "garden"],
            vec!["cat bed", "dog bed"],
            vec!["shelf", "sofa"]
        ]
    );
    assert_eq!(result.observations["category"], json!([["cat bed", 0.9], ["shelf", 0.6]]));
    assert_eq!(requests.len(), 3 + 4);
}

#[test]
fn the_cut_is_the_threshold_of_the_output_and_not_the_filters() {
    // The model says 0.45 of the leaf that fits and 0.3 of one that does not: at one half, nothing is a yes.
    let rdm = Rdm::new(under("root"))
        .tool(category())
        .model(odds(&[("cat bed", 0.45), ("shelf", 0.3)]));
    let listing = || inputs! { "listing" => "A padded window shelf for a cat." };
    assert_eq!(rdm.call(listing()).unwrap().observations["category"], json!([]));
    // The filter has no cut of its own to tune. The yes is the output's, and its threshold is the module's.
    rdm.set_params("fits", Params::threshold(0.4));
    assert_eq!(rdm.call(listing()).unwrap().observations["category"], json!([["cat bed", 0.45]]));
}

// The notebook's cell: a tool follows the two likeliest categories at every level, not only the
// one chosen, and every leaf they reach comes back.

fn taxonomy(category: &str) -> Option<&'static [&'static str]> {
    Some(match category {
        "Products" => &["Animals & Pet Supplies", "Furniture", "Kitchen"],
        "Animals & Pet Supplies" => &["Cat Supplies", "Dog Supplies"],
        "Cat Supplies" => &["Cat Beds", "Cat Toys", "Cat Window Perches"],
        "Dog Supplies" => &["Dog Beds", "Dog Toys", "Dog Leashes"],
        "Furniture" => &["Shelves", "Beds", "Sofas"],
        "Kitchen" => &["Cookware", "Tableware", "Kitchen Knives"],
        _ => return None,
    })
}

/// One request: which category under this one fits the listing?
fn ask(category: &str) -> Signature {
    let options = taxonomy(category).unwrap_or_default().iter().copied();
    Signature::parse("listing -> category").unwrap().with_output(
        "category",
        Output::choice("Which category best matches the product in `inputs.listing`?", options),
    )
}

#[test]
fn the_filter_cell_reaches_every_leaf_under_the_two_likeliest_categories() {
    let likeliest = Filter::new().top(2); // every option of a choice with its probability, likeliest first, and the two likeliest kept
    // Follow the two likeliest categories, and come to every leaf they reach.
    let category = Tool::new("category", "Follow the two likeliest categories.", move |call| {
        let listing: String = call.arg("listing")?;
        let mut leaves: Vec<String> = Vec::new();
        // The answer with its probabilities, not only its value.
        for (option, _probability) in likeliest.options(call.answer("category")) {
            if taxonomy(&option).is_none() {
                leaves.push(option);
                continue;
            }
            let below = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(ask(&option)))?;
            leaves.extend(below.observed::<Vec<String>>("category").unwrap_or_default());
        }
        Ok(leaves)
    });
    // A stand-in that rates what has to do with cats and windows over what has not.
    let model = Arc::new(Scripted::new(|request| {
        let rated = |label: &str| match label {
            "Animals & Pet Supplies" | "Cat Supplies" | "Cat Window Perches" => 0.6,
            "Furniture" | "Dog Supplies" | "Cat Beds" | "Shelves" | "Dog Beds" => 0.3,
            _ => 0.1,
        };
        let labels = request.questions["category"].labels();
        [(
            "category",
            Answer::choice(labels.into_iter().map(|label| (rated(&label), label)).map(|(p, label)| (label, p))),
        )]
    }));
    let wide = Rdm::new(ask("Products")).tool(category).model(model.clone());
    let listing = "A padded shelf that suction-cups to a window, so a cat can sun itself.";
    let result = wide.call(inputs! { "listing" => listing }).unwrap();
    assert_eq!(result.chosen("category"), Some("Animals & Pet Supplies"));
    assert_eq!(
        result.observations["category"],
        json!(["Cat Window Perches", "Cat Beds", "Dog Beds", "Dog Toys", "Shelves", "Beds"])
    );
    // The kitchen was never asked about: five requests, where following every category takes seven.
    assert_eq!((result.trajectory.len(), model.requests().len()), (5, 5));
    assert!(
        model
            .requests()
            .iter()
            .all(|request| !request.questions["category"].labels().contains(&"Cookware".to_owned()))
    );
    assert_eq!(result.termination_reason, "final");
}

// The notebook's hierarchy cell: with nothing set, the filter sorts every option of a choice, and
// the tool follows each one at least half as likely as the likeliest. There is no beam width.

#[test]
fn the_hierarchy_cell_follows_every_category_at_least_half_as_likely_as_the_likeliest() {
    /// The categories directly under one: none under a leaf.
    fn under(category: &str) -> Vec<String> {
        let under: &[&str] = match category {
            "All products" => &["Furniture", "Luggage & Bags", "Electronics"],
            "Luggage & Bags" => &["Backpacks", "Suitcases"],
            "Electronics" => &["Laptops", "Laptop Bags", "Phones"],
            _ => &[],
        };
        under.iter().map(|each| (*each).to_owned()).collect()
    }
    /// One request: which of these categories fits the listing best?
    fn ask(options: Vec<String>) -> Signature {
        Signature::parse("listing -> category").unwrap().with_output(
            "category",
            Output::choice("Which category best matches the product in `inputs.listing`?", options),
        )
    }
    let likeliest = Filter::new(); // sorts what a decision rated: every option of a choice, not only the one chosen
    let drawn = Arc::new(std::sync::Mutex::new(Vec::new()));
    let category = {
        let drawn = drawn.clone();
        Tool::new("category", "Follows each category likely enough.", move |call| {
            let listing: String = call.arg("listing")?;
            let rated = likeliest.options(call.answer("category")); // (category, probability) pairs, likeliest first
            let likely: Vec<&String> = rated
                .iter()
                .filter(|(_, probability)| *probability >= rated[0].1 / 2.0)
                .map(|(each, _)| each)
                .collect();
            // What the page draws: every option with its probability, as a map, and the ones followed.
            let branch: IndexMap<String, f64> = rated.iter().cloned().collect();
            drawn.lock().unwrap().push((branch, likely.len()));
            let mut leaves: Vec<String> = Vec::new();
            for each in likely {
                let below = under(each);
                if below.is_empty() {
                    leaves.push(each.clone());
                    continue;
                }
                let found = call.sub_with(Ask::new(inputs! { "listing" => &listing }).signature(ask(below)))?;
                leaves.extend(found.observed::<Vec<String>>("category").unwrap_or_default());
            }
            Ok(leaves)
        })
    };
    // Rates electronics over bags at the top, one option clearly under each, and furniture hardly at all.
    let model = Arc::new(Scripted::new(|request| {
        let rated = |label: &str| match label {
            "Electronics" => 0.5,
            "Luggage & Bags" => 0.3,
            "Furniture" => 0.2,
            "Backpacks" | "Laptop Bags" => 0.8,
            _ => 0.1,
        };
        let labels = request.questions["category"].labels();
        [(
            "category",
            Answer::choice(labels.into_iter().map(|label| {
                let probability = rated(&label);
                (label, probability)
            })),
        )]
    }));
    let search = Rdm::new(ask(under("All products")))
        .tool(category)
        .max_depth(12)
        .max_predictor_calls(60)
        .model(model.clone());
    let listing = "Laptop backpack with a padded sleeve, 20 litres.";
    let result = search.call(inputs! { "listing" => listing }).unwrap();
    // Electronics was followed first, as the likeliest, then bags, at 0.3 of its 0.5. Furniture was not.
    let found: Vec<String> = result.observed("category").unwrap();
    assert_eq!(
        (found.clone(), result.trajectory.len()),
        (vec!["Laptop Bags".to_owned(), "Backpacks".to_owned()], 3)
    );
    let drawn = drawn.lock().unwrap();
    assert_eq!(
        drawn[0].0.keys().collect::<Vec<_>>(),
        ["Electronics", "Luggage & Bags", "Furniture"]
    );
    assert_eq!(drawn.iter().map(|(_, followed)| *followed).collect::<Vec<_>>(), [2, 1, 1]);
    // More than one leaf came back: ask once more, among those, for the answer only.
    let best = match found.as_slice() {
        [only] => only.clone(),
        _ => {
            let among = Ask::new(inputs! { "listing" => listing })
                .signature(ask(found.clone()))
                .max_depth(0);
            search.call_with(among).unwrap().chosen("category").unwrap_or_default().to_owned()
        }
    };
    // The two are rated alike, and of options rated alike the first given is the one chosen.
    assert_eq!((best.as_str(), model.requests().len()), ("Laptop Bags", 4));
}
