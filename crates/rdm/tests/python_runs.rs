//! Whole runs, set beside the Python module's.
//!
//! Two small programs, each run on the Python module under settings drawn at random, on a
//! stand-in decision model whose answers are a hash of what it was shown. What each run came
//! to is in `fixtures/`, a line for each: the answers, the observations, why it ended, and
//! every entry of the trajectory, with each decision as its plain value. The same programs are
//! written here, and every run must come to the same.

use std::sync::Arc;

use rdm::systemone::{Answer, Kind, Params, Request, Scripted};
use rdm::{Ask, Json, Opt, Rdm, Shared, Signature, Tool, inputs};
use serde_json::json;

fn fnv(text: &str) -> u32 {
    text.bytes()
        .fold(2166136261, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(16777619))
}

/// An input as the Python stand-in writes it into what it hashes.
fn show(value: &Json) -> String {
    match value {
        Json::Null => "-".to_owned(),
        Json::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A stand-in whose answer to a question is a hash of its salt, the question's name and
/// `seen(inputs)`: one of `nouls` for a yes or no, and one option or level as certain.
fn client(salt: i64, nouls: &'static [f64], seen: impl Fn(&Json) -> String + Send + Sync + 'static) -> Arc<Scripted> {
    let answers = move |request: &Request| {
        let seen = seen(&request.state["inputs"]);
        let each = request.questions.iter().map(|(name, question)| {
            let hash = fnv(&format!("{salt}|{name}|{seen}")) as usize;
            let labels = question.labels();
            let answer = match question.kind {
                Kind::Noul => Answer::noul(nouls[hash % nouls.len()]),
                Kind::Choice => Answer::chose(&labels[hash % labels.len()], labels.iter().map(String::as_str)),
                Kind::Score => Answer::score((0..labels.len()).map(|at| if at == hash % labels.len() { 1.0 } else { 0.0 })),
            };
            (name.clone(), answer)
        });
        each.collect::<Vec<_>>()
    };
    Arc::new(Scripted::new(answers).named("jev-test").tokens(100))
}

/// Every decision as its plain value, as the fixtures hold them: choosing none is written `"None"` there.
fn plain(value: &Json) -> Json {
    match value {
        Json::Object(map) if map.contains_key("confidence") && map.contains_key("value") => match &map["value"] {
            Json::Null => json!("None"),
            value => value.clone(),
        },
        Json::Object(map) => Json::Object(map.iter().map(|(key, item)| (key.clone(), plain(item))).collect()),
        Json::Array(items) => Json::Array(items.iter().map(plain).collect()),
        other => other.clone(),
    }
}

/// Whether two values are the same, a number being the same whether or not it was written whole.
fn same(ours: &Json, theirs: &Json) -> bool {
    match (ours, theirs) {
        (Json::Number(ours), Json::Number(theirs)) => ours.as_f64() == theirs.as_f64(),
        (Json::Array(ours), Json::Array(theirs)) => {
            ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(ours, theirs)| same(ours, theirs))
        }
        (Json::Object(ours), Json::Object(theirs)) => {
            let found = |(key, ours): (&String, &Json)| theirs.get(key).is_some_and(|theirs| same(ours, theirs));
            ours.len() == theirs.len() && ours.iter().all(found)
        }
        (ours, theirs) => ours == theirs,
    }
}

/// Run every setting of a fixture, and set what it came to beside what the Python module's
/// did. Returns how many runs there were, and how many requests they recorded.
fn beside_python(fixture: &str, run: impl Fn(&Json) -> Json) -> (usize, usize) {
    let (mut runs, mut requests) = (0, 0);
    for line in fixture.lines().filter(|line| !line.trim().is_empty()) {
        let recorded: Json = serde_json::from_str(line).expect("a fixture is JSON");
        let (config, theirs) = (&recorded["config"], &recorded["python"]);
        let ours = run(config);
        for (key, theirs) in theirs.as_object().expect("a run is an object") {
            let ours = &ours[key];
            assert!(
                same(ours, theirs),
                "run {runs}, with {config}: `{key}` is\n  {ours}\nand the Python module's is\n  {theirs}"
            );
        }
        runs += 1;
        requests += theirs["trajectory"].as_array().map_or(0, Vec::len);
    }
    (runs, requests)
}

/// What a run came to, as the fixtures hold one.
fn outcome(result: rdm::Result<rdm::Prediction>, answers: Json, model: &Scripted) -> Json {
    match result {
        Err(error) => json!({"error": format!("{error:?}"), "asked": model.requests().len()}),
        Ok(result) => json!({
            "answers": plain(&answers),
            "next": result.value("next"),
            "observations": plain(&json!(result.observations)),
            "termination_reason": result.termination_reason,
            "trajectory": plain(&json!(result.trajectory)),
            "asked": model.requests().len(),
        }),
    }
}

/// One judgment and a choice of ways, as many as the number allows: where there is one, it is not asked.
fn judge(n: i64) -> Signature {
    let ways: Vec<String> = (0..(n % 3 + 1)).map(|way| format!("way{way}")).collect();
    Signature::new("Judge.")
        .input_typed("n", "int")
        .noul("ok", "OK?")
        .choice("go", "Which way?", ways)
}

/// A pointer over tools that change the world, ask the module again, ask it other questions,
/// and call another module that has limits of its own.
fn walk(config: &Json) -> Json {
    let seen = |inputs: &Json| {
        let world = inputs.get("world").map_or("-".to_owned(), |world| show(&world["k"]));
        format!("{}|{world}|{}", show(&inputs["n"]), show(&inputs["observation"]))
    };
    let model = client(config["salt"].as_i64().unwrap(), &[0.1, 0.9, 0.9, 0.9], seen);
    let limit = |name: &str| config[name].as_u64().unwrap() as usize;
    let bound = config["bound"].as_i64().unwrap();

    let again = Tool::new("again", "", move |call| {
        let n: i64 = call.arg("n")?;
        if n >= bound {
            return Ok(json!("bottom"));
        }
        Ok(json!([call.sub(inputs! { "n" => n + 1 })?.termination_reason]))
    });
    let other = Rdm::new(Signature::new("Count.").input_typed("n", "int").noul("again", "Again?"))
        .tool(again)
        .max_depth(limit("od"))
        .max_predictor_calls(limit("oc"))
        .max_tokens(config["ot"].as_u64())
        .model(model.clone());

    let world = Shared::new(json!({"k": 0}));
    let make = |name: &str| -> Tool {
        let description = format!("Do {name}.");
        match name {
            "tick" => {
                let world = world.clone();
                Tool::new(name, description, move |_| {
                    Ok(world.update(|world| {
                        world["k"] = json!(world["k"].as_i64().unwrap() + 1);
                        world["k"].clone()
                    }))
                })
            }
            "noop" => Tool::new(name, description, |_| Ok("nothing")),
            "dive" => Tool::new(name, description, |call| {
                let n: i64 = call.arg("n")?;
                let ask = Ask::new(inputs! { "n" => n + 1, "world" => call.input("world")? });
                let sub = call.sub_with(if n % 3 == 0 { ask.max_depth(1) } else { ask })?;
                Ok(json!([sub.termination_reason, sub.pointed("next")]))
            }),
            "ask" => Tool::new(name, description, |call| {
                let n: i64 = call.arg("n")?;
                let sub = call.sub_with(Ask::new(inputs! { "n" => n }).signature(judge(n)))?;
                Ok(json!([sub.termination_reason, sub.observation("ok"), sub.observation("go")]))
            }),
            "ok" => {
                let other = other.clone();
                Tool::new(name, description, move |call| {
                    let n: i64 = call.arg("n")?;
                    Ok(call.on(&other).sub(inputs! { "n" => n })?.termination_reason)
                })
            }
            _ => Tool::new(name, description, |call| Ok(format!("went from {}", call.arg::<i64>("n")?))),
        }
    };
    let signature = Signature::new("Walk.")
        .input_typed("n", "int")
        .input_typed("world", "dict")
        .input("observation")
        .tool("next", "What next?");
    let tools = config["tools"].as_array().unwrap().iter().map(|name| make(name.as_str().unwrap()));
    let rdm = Rdm::new(signature)
        .tools_from(tools)
        .finish(config["finish"].as_bool().unwrap())
        .max_iters(limit("max_iters"))
        .max_depth(limit("max_depth"))
        .max_predictor_calls(limit("max_predictor_calls"))
        .max_tokens(config["max_tokens"].as_u64())
        .model(model.clone());
    let result = rdm.call(inputs! { "n" => config["n"].as_i64().unwrap(), "world" => &world });
    let mut outcome = outcome(result, Json::Null, &model);
    outcome["world"] = world.get();
    outcome
}

#[test]
fn a_pointer_over_tools_that_recurse_and_call_another_module_runs_as_it_does_in_python() {
    let (runs, requests) = beside_python(include_str!("fixtures/walk.jsonl"), walk);
    assert_eq!((runs, requests), (80, 245));
}

/// A yes that splits in three, each part with a depth of its own, beside a choice that may
/// come to none, a score, and a question with one answer.
fn tree(config: &Json) -> Json {
    let model = client(config["salt"].as_i64().unwrap(), &[0.9, 0.6, 0.1], |inputs| show(&inputs["n"]));
    let limit = |name: &str| config[name].as_u64().unwrap() as usize;
    let signature = Signature::new("Tree.")
        .input_typed("n", "int")
        .noul("split", "Split?")
        .choice(
            "mode",
            "Which mode?",
            [Opt::new("a", ""), Opt::new("b", "desc b"), Opt::none("Neither.")],
        )
        .score("rate", "How much?", ["low", "mid", "high"]);
    let only = Signature::new("Judge.").input_typed("n", "int").choice("pick", "Pick?", ["only"]);

    let split = Tool::new("split", "", |call| {
        let n: i64 = call.arg("n")?;
        let asks = [2 * n + 1, 2 * n + 2, 2 * n + 1].into_iter().enumerate().map(|(at, part)| {
            let ask = Ask::new(inputs! { "n" => part });
            match (n + at as i64) % 4 {
                0 => ask.max_depth(0),
                1 => ask.max_depth(1),
                _ => ask,
            }
        });
        let subs = call.batch(asks)?;
        let seen = subs
            .iter()
            .map(|sub| json!([sub.termination_reason, sub.noul("split").map(|split| split.value)]));
        Ok(seen.collect::<Vec<_>>())
    });
    let mode = Tool::new("mode", "", move |call| {
        let (mode, n): (String, i64) = (call.arg("mode")?, call.arg("n")?);
        if mode == "b" {
            let sub = call.sub_with(Ask::new(inputs! { "n" => n }).signature(only.clone()))?;
            return Ok(json!([mode, sub.termination_reason, sub.observation("pick")]));
        }
        Ok(json!([mode, n]))
    });
    let pick = Tool::new("pick", "", |call| Ok(format!("picked {}", call.arg::<String>("pick")?)));
    let rate = Tool::new("rate", "", |_| Ok("never"));
    let rdm = Rdm::new(signature)
        .tools_from([split, mode, pick, rate])
        .max_depth(limit("max_depth"))
        .max_predictor_calls(limit("max_predictor_calls"))
        .max_tokens(config["max_tokens"].as_u64())
        .threads(1)
        .model(model.clone());
    if let Some(threshold) = config["threshold"].as_f64() {
        rdm.set_params("split", Params::threshold(threshold));
    }
    let result = rdm.call(inputs! { "n" => config["n"].as_i64().unwrap() });
    let answers = result.as_ref().map_or(
        Json::Null,
        |result| json!({"split": result.value("split"), "mode": result.value("mode"), "rate": result.value("rate")}),
    );
    // The fixtures hold choosing none as the label it has.
    let answers = match answers {
        Json::Object(mut answers) => {
            if answers["mode"].is_null() && result.as_ref().is_ok_and(|result| result.get("mode").is_some()) {
                answers["mode"] = json!("None");
            }
            Json::Object(answers)
        }
        other => other,
    };
    outcome(result, answers, &model)
}

#[test]
fn a_tree_of_batches_with_a_none_a_score_and_a_threshold_runs_as_it_does_in_python() {
    let (runs, requests) = beside_python(include_str!("fixtures/tree.jsonl"), tree);
    assert_eq!((runs, requests), (80, 233));
}
