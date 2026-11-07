//! A copy of a module is a module of its own, and what the core gained for the ports of the
//! notebook's cells: state read as a type, and the option of choosing none beside plain labels.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::{Say, client, scripted};
use rdm::{Ask, Call, Fit, Inputs, Output, Rdm, Shared, Signature, Source, Tool, inputs};
use serde_json::json;

fn operate() -> Signature {
    Signature::new("Go on.")
        .input("request")
        .input("observation")
        .tool("next", "What next?")
}

/// A shelf of names, each a tool. The first puts the second up in its place.
struct Shelf {
    names: Shared<Vec<String>>,
    /// Whether a copy of a module is given a shelf of its own.
    copied: bool,
}

impl Source for Shelf {
    fn offers(&self) -> Vec<Tool> {
        let offer = |name: &String| {
            let (names, first) = (self.names.clone(), name == "first");
            Tool::new(name.clone(), format!("The {name}."), move |_| {
                if first {
                    names.set(vec!["second".to_owned()]);
                    return Ok("put up the second");
                }
                Ok("did the second")
            })
        };
        self.names.lock().iter().map(offer).collect()
    }

    fn copy(&self) -> Option<Arc<dyn Source>> {
        let own = Shelf {
            names: Shared::new(self.names.get()),
            copied: true,
        };
        self.copied.then(|| Arc::new(own) as Arc<dyn Source>)
    }
}

/// What a module's first request offers, after a copy of it has run and put up the second.
fn offered_after_a_copy_ran(copied: bool) -> Vec<String> {
    let names = Shared::new(vec!["first".to_owned()]);
    let model = scripted(["finish"]);
    let original = Rdm::new(operate()).source(Shelf { names, copied }).model(model.clone());
    let copy = original.copy();
    copy.set_model(scripted(["first", "finish"]));
    let result = copy.call(inputs! { "request" => "Go on." }).unwrap();
    assert_eq!(result.pointed("next"), ["first", "finish"]);
    // The original was never run.
    original.call(inputs! { "request" => "Go on." }).unwrap();
    common::offered(&model.requests()[0], "next")
}

#[test]
fn a_copy_has_a_source_of_its_own_where_the_source_says_how_to_copy_it() {
    // The Python module's copy is a deep one: `rdm.deepcopy().sources[0] is not shelf`. A source
    // that keeps something says how it is copied, and a run of the copy then writes into its own.
    assert_eq!(offered_after_a_copy_ran(true), ["first", "finish"]);
    // One that does not say is the same source in both: what either puts up, both offer.
    assert_eq!(offered_after_a_copy_ran(false), ["second", "finish"]);
}

/// A fit that counts the requests it was given, and leaves them as they are.
struct Counting {
    seen: Shared<usize>,
    copied: bool,
}

impl Fit for Counting {
    fn fit(&self, signature: Signature, inputs: Inputs, _: &Call) -> rdm::Result<(Signature, Inputs)> {
        self.seen.update(|seen| *seen += 1);
        Ok((signature, inputs))
    }

    fn copy(&self) -> Option<Arc<dyn Fit>> {
        let own = Counting {
            seen: Shared::new(self.seen.get()),
            copied: true,
        };
        self.copied.then(|| Arc::new(own) as Arc<dyn Fit>)
    }
}

#[test]
fn a_copy_has_a_fit_of_its_own_where_the_fit_says_how_to_copy_it() {
    let ask = Signature::new("Ask.").input("q").noul("found", "Found?");
    for (copied, counted) in [(true, 1), (false, 2)] {
        let seen = Shared::new(0);
        let fit = Counting {
            seen: seen.clone(),
            copied,
        };
        let original = Rdm::new(ask.clone()).fit(fit).model(client(|_, _, _| false));
        original.call(inputs! { "q" => "q" }).unwrap();
        original.copy().call(inputs! { "q" => "q" }).unwrap();
        assert_eq!(seen.get(), counted, "copied: {copied}");
    }
}

#[derive(serde::Deserialize)]
struct Handbook<'a> {
    #[serde(borrow)]
    outline: HashMap<&'a str, Vec<&'a str>>,
}

#[test]
fn a_tool_reads_the_state_as_a_type_of_its_own() {
    // The outline cell reads `call.state["outline"]` at every node. Read as a type that borrows
    // from the state, that copies no text.
    let explore = Tool::new("explore", "", |call| {
        let (entry, handbook): (String, Handbook) = (call.arg("entry")?, call.state_as()?);
        let Some(under) = handbook.outline.get(entry.as_str()) else {
            return Ok(vec![entry]);
        };
        let mut found = Vec::new();
        for each in under {
            found.extend(
                call.sub(inputs! { "entry" => each })?
                    .observed::<Vec<String>>("explore")
                    .unwrap_or_default(),
            );
        }
        Ok(found)
    });
    let navigate = Signature::new("Search a handbook.")
        .input("entry")
        .noul("explore", "Does `inputs.entry` name the topic?");
    let model = client(|inputs, _, _| inputs["entry"] != "Sick leave");
    let rdm = Rdm::new(navigate).tool(explore).model(model);
    let state = json!({"outline": {"Handbook": ["Time off"], "Time off": ["Annual leave", "Sick leave"]}});
    let result = rdm.call_with(Ask::new(inputs! { "entry" => "Handbook" }).state(state)).unwrap();
    assert_eq!(result.observations["explore"], json!(["Annual leave"]));
    // The state is the caller's and not the model's: one that is not of the type is a usage error.
    let error = rdm
        .call_with(Ask::new(inputs! { "entry" => "Handbook" }).state(json!({"outline": 3})))
        .unwrap_err();
    assert!(
        error.is_usage() && error.to_string().starts_with("Tool 'explore' reads the state as a type it is not"),
        "{error:?}"
    );
    assert!(rdm.call(inputs! { "entry" => "Handbook" }).unwrap_err().is_usage());
}

#[test]
fn choosing_none_can_be_offered_beside_plain_labels() {
    // Python's `Choice[("warmer", ""), ("cooler", ""), (None, "...")]`, without writing each option out.
    let home = Signature::new("Carry out a request.").input("request").choice_or_none(
        "thermostat",
        "How should the temperature change?",
        ["warmer", "cooler"],
        "The request is not about temperature.",
    );
    let options = home.output("thermostat").unwrap().options().unwrap();
    let said: Vec<(&str, &str, bool)> = options
        .iter()
        .map(|option| (option.label.as_str(), option.description.as_str(), option.none))
        .collect();
    assert_eq!(
        said,
        [
            ("warmer", "", false),
            ("cooler", "", false),
            ("None", "The request is not about temperature.", true)
        ]
    );
    let same = Output::choice("How should the temperature change?", [("warmer", ""), ("cooler", "")])
        .or_none("The request is not about temperature.");
    assert_eq!(home.output("thermostat"), Some(&same));
    // An output that is not a choice has no options to add one to.
    assert_eq!(Output::noul("Warm?").or_none("Neither."), Output::noul("Warm?"));
    // Chosen, it calls nothing and reads as null.
    let ran = Shared::new(0);
    let counted = ran.clone();
    let thermostat = Tool::new(
        "thermostat",
        "",
        move |_| Ok(counted.update(|ran| std::mem::replace(ran, *ran + 1))),
    );
    let model = client(|inputs, _, _| Say::Pick(if inputs["request"] == "It is freezing." { "warmer" } else { "None" }.to_owned()));
    let rdm = Rdm::new(home).tool(thermostat).model(model.clone());
    let result = rdm.call(inputs! { "request" => "Lights off." }).unwrap();
    assert_eq!((result.value("thermostat"), ran.get()), (json!(null), 0));
    rdm.call(inputs! { "request" => "It is freezing." }).unwrap();
    assert_eq!(ran.get(), 1);
    assert_eq!(
        model.requests()[0].questions["thermostat"].criteria,
        Some(json!({"warmer": null, "cooler": null, "None": "The request is not about temperature."}))
    );
}

#[test]
fn the_deepest_a_run_may_go_is_said() {
    assert_eq!(rdm::DEEPEST, 1024);
}
