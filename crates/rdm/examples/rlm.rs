//! An RLM as the writer: a language model in a loop of its own, which tries what it writes.
//!
//! In `repl` the writer is one request, and the function it gives has never been run when it is
//! put on the map. Here the writer is an [`Rlm`]: it writes a step, the step is run, it reads
//! what the step printed, and it writes the next. It defines the function, calls it to try it,
//! and submits it when it does what was asked. Only then is it on the map.
//!
//! The house is real while the writer works, and trying is not acting: a checkpoint is taken
//! when the writing starts and put back when it ends. So the function is tried on the house
//! as it is, the house cools to 18, and it is 21 again when the decision model is asked what
//! happens next. It is the decision model that cools the house, by pointing at the name.
//!
//! This has no cell on the site: the Python gives a `dspy.RLM` the `define` function, with the
//! house's actions taken away from it, so that what it writes cannot be tried.
//!
//! ```text
//! cargo run -p rdm --example rlm
//! cargo run -p rdm --example rlm -- --task 2
//! RDM_TRACE=1 cargo run -p rdm --example rlm      # every step the writer took
//! cargo run -p rdm --example rlm -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::repl::Repl;
use rdm::rlm::Rlm;
use rdm::systemone::Question;
use rdm::{Json, Rdm, Shared, Signature, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    let language_model = common::language_model(writes)?; // the language model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new(
        "rlm",
        "An RLM",
        "The model points at write, a language model writes a function and tries it, and it is on the map from then on.",
    )
    .text(
        "task",
        "Ask the house",
        &["Bring the heat down to 18 degrees.", "Make it three degrees cooler."],
    )
    .house(json!({"tv": "off", "lights": "on", "door": "closed", "lock": "locked", "heat": 21}))
    .read();
    let task = form.text_of("task");
    let house = form.house_of();

    // The sandbox, and what written code may call in it. They are the real ones.
    let (cooled, read) = (house.clone(), house.clone());
    let repl = Repl::new()
        .function("cooler", "Make it one degree cooler.", move |_| Ok(common::cooler(&cooled)))
        .function("heat", "How warm it is, in degrees.", move |_| Ok(common::heat(&read)));

    // The writer: a loop of its own, to six steps, with every step kept for the trace. The
    // house is put back when the writing ends, however it ends.
    let (kept, steps, seen) = (house.clone(), Shared::new(Vec::new()), Shared::new(Vec::new()));
    let (took, saw) = (steps.clone(), seen.clone());
    let look = house.clone();
    let write = Rlm::new("Write one Python function that does what the task still needs.")
        .input("task")
        .input_typed("house", "dict")
        .input_described(
            "observation",
            "What the last step of the house did. If a function failed, why is here.",
        )
        .lm(language_model)
        .max_iters(6)
        .max_llm_calls(0)
        .checkpoint(move || {
            let (before, kept) = (kept.get(), kept.clone());
            move || {
                kept.set(before);
            }
        })
        .watch(move |step| took.lock().push(step.clone()))
        .writer(&repl)
        // The caller's check, and not the writer's word for it: the heat as trying left it.
        .check("changes the heat", move |tried| {
            saw.lock().push(common::heat(&look));
            if tried.printed.is_some() {
                Ok(())
            } else {
                Err("it could not be tried: give it no parameter the request does not name".to_owned())
            }
        })
        .tool(
            "write",
            "Have a new tool written in code and put on this list: nothing listed does what is still needed.",
        );

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let cool = Rdm::new(operate).tool(write).source(repl.clone()).model(decision_model);

    common::show_house(&house, &task);
    let before = common::heat(&house);
    let result = cool.call(inputs! { "task" => &task, "house" => &house })?;
    common::trace(&result);
    if common::tracing() {
        for (at, step) in steps.lock().iter().enumerate() {
            println!("| step {}: {}", at + 1, step.code.replace('\n', "\n|         "));
            println!("|   -> {}", step.output.replace('\n', "\n|      "));
        }
    }
    py::print!("pointed at:", result.pointed("next"));
    py::print!("steps the writer took:", steps.lock().len());
    py::print!("heat while it was tried:", seen.get());
    py::print!("heat:", before, "->", common::heat(&house));
    py::print!("written:", repl.written().keys().collect::<Vec<_>>());
    Ok(())
}

/// The stand-in decision model, as in `repl`: with nothing done yet, `write`, for a task the
/// stand-in language model writes for; when a function was written, that function; and after
/// that, `finish`.
fn standin(inputs: &Json, _name: &str, question: &Question) -> Say {
    let shown = |name: &str| inputs[name].as_str().unwrap_or_default();
    let (task, observation) = (shown("task"), shown("observation"));
    // "`cool_to_18` was written and is now on the list. It has not been run, ..."
    let written = observation.strip_prefix('`').and_then(|rest| rest.split_once("` was written"));
    match written {
        Some((name, _)) if question.labels().iter().any(|label| label == name) => name.into(),
        None if observation.is_empty() && function_for(task).is_some() => "write".into(),
        _ => rdm::FINISH.into(),
    }
}

/// The stand-in language model: the steps of a writer, as the JSON a model would reply with.
/// In its first step it defines the function a task asks for and calls it, to see what it
/// does. In its second it submits it. Asked for the outputs when the turns ran out, which is
/// for a task it writes nothing for, it gives none.
fn writes(_system: &str, prompt: &str) -> String {
    let read = |name: &str| rdm::lm::input_of(prompt, name).unwrap_or_default();
    // `task` is a variable where the code runs, and is not in the prompt: its preview is.
    let task = read("variables_info").split("Preview:\n```\n").nth(1).unwrap_or_default();
    let task = task.split("\n```").next().unwrap_or_default();
    let (name, description, code) = function_for(task).unwrap_or_default();
    if read("iteration").is_empty() {
        return json!({"name": name, "description": description, "code": code}).to_string();
    }
    if read("iteration").starts_with("1/") {
        let reasoning = "Define the function, and call it to see what it does.";
        return json!({"reasoning": reasoning, "code": format!("{code}\n\nprint({name}())")}).to_string();
    }
    let reasoning = "It did what was asked when it was tried. Submit it.";
    json!({"reasoning": reasoning, "code": format!("SUBMIT({name:?}, {description:?})")}).to_string()
}

/// What the stand-in language model writes for a task: its name, its description and its
/// source. For "down to 18 degrees" it is a loop until it is 18, and for "three degrees
/// cooler" a loop of three. A task that asks for neither has none.
fn function_for(task: &str) -> Option<(String, String, String)> {
    const NUMBERS: [&str; 10] = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
    let task = task.to_lowercase();
    let words: Vec<&str> = task
        .split(|each: char| !each.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    let at = words.iter().position(|word| word.starts_with("degree")).filter(|at| *at > 0)?;
    let said = words[at - 1];
    let degrees = said
        .parse()
        .ok()
        .or_else(|| NUMBERS.iter().position(|number| *number == said).map(|at| at + 1))?;
    let returns = "    return f'it is {heat()} degrees'";
    if words.get(at + 1) == Some(&"cooler") {
        let name = format!("cool_by_{said}_degrees");
        let code = format!("def {name}():\n    for _ in range({degrees}):\n        cooler()\n{returns}");
        Some((name, format!("Make it {said} degrees cooler."), code))
    } else if task.contains("down to") {
        let name = format!("cool_to_{degrees}");
        let code = format!("def {name}():\n    while heat() > {degrees}:\n        cooler()\n{returns}");
        Some((name, format!("Cool the house until it is {degrees} degrees."), code))
    } else {
        None
    }
}
