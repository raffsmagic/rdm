//! A REPL: pointing at code a language model wrote. A decision model cannot count, loop or
//! compare, and it can only point at what is on its map. A language model can write. The REPL is
//! what the two share: a function written into it is on the map from the next request.
//!
//! The writer is where this differs from the Python cell. There it is a `dspy.RLM`: a language
//! model in a loop of its own, which writes code that calls `define`, and may call `heat` while
//! it writes. Here the writer is one request to the language model (the example `rlm` is this
//! one with a writer that is a loop, and tries what it writes), a
//! `Generate`, whose outputs are the function as data: the `name`, the `description` and the
//! `code` that `define` takes. `repl.writer_from` makes that a tool that writes: it defines what
//! was given, and what `define` said is the tool's observation. So the instructions no longer
//! say "and define it with `define`", there is no output `wrote`, and the writer cannot read the
//! heat but from the house it is shown. A function that is refused is not written again inside
//! the tool: the refusal is the next request's observation, the writer is shown it in an input
//! `observation` of its own, and the decision model can point at `write` again.
//!
//! ```text
//! cargo run -p rdm --example repl
//! cargo run -p rdm --example repl -- --task 2
//! cargo run -p rdm --example repl -- --help
//! ```

mod common;

use common::{Form, Say, py};
use rdm::lm::Generate;
use rdm::repl::Repl;
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, inputs};
use serde_json::json;

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for
    let language_model = common::language_model(writes)?; // the language model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new(
        "repl",
        "A REPL",
        "The model points at write, a language model writes a function, and it is on the map from then on.",
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

    // The sandbox, and what written code may call in it: `cooler()`, which makes it one degree
    // cooler, and `heat()`, how warm it is, in degrees. `common::cooler` redraws the house too.
    let (cooled, read) = (house.clone(), house.clone());
    let repl = Repl::new()
        .function("cooler", "Make it one degree cooler.", move |_| Ok(common::cooler(&cooled)))
        .function("heat", "How warm it is, in degrees.", move |_| Ok(common::heat(&read)));

    let write = Generate::new(
        "Write one Python function that does what the request still needs.\n\n\
         Where the function runs, `cooler()` makes it one degree cooler and `heat()` returns the degrees it is now.",
    )
    .input("task")
    .input_typed("house", "dict")
    .input_described("observation", "What the last step did. If a function was refused, or failed, why is here.")
    .output_described("name", "The function's name.")
    .output_described("description", "One sentence for a menu: what choosing the function does.")
    .output_described(
        "code",
        "The function's source: one `def` of that name and nothing beside it. A parameter is given the request's answer or input of the same name.",
    );

    let writer = write.lm(language_model); // one request: what it gives, the REPL defines
    let write = repl.writer_from(&writer.tool(
        "write",
        "Have a new tool written in code and put on this list: nothing listed does what is still needed.",
    ));

    let operate = Signature::new("Operate a smart home, one step at a time.")
        .input("task")
        .input_typed("house", "dict")
        .input_described("observation", "What the last step did.")
        .tool("next", "What should happen next for `inputs.task`?");

    let cool = Rdm::new(operate).tool(write).source(repl.clone()).model(decision_model);

    common::show_house(&house, &task);
    let result = cool.call(inputs! { "task" => &task, "house" => &house })?;
    common::trace(&result);
    py::print!("pointed at:", result.pointed("next"));
    py::print!("heat:", common::heat(&house));
    py::print!("written:", repl.written().keys().collect::<Vec<_>>());
    Ok(()) // repl.shutdown(): the REPL goes with the program
}

/// The stand-in decision model, which answers when the environment asks for no real one. It is
/// the plan the Python tests its reference page with (`write`, `cool_to_18`, `finish`), read
/// from what each request shows: with nothing done yet, `write`, for a task the stand-in
/// language model writes for; when `define` has said a function was written, that function;
/// and after that, `finish`.
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

/// The stand-in language model: the function a task asks for, as the JSON a model would reply
/// with. For any other task it gives no function, and `define` refuses what it gave.
fn writes(_system: &str, prompt: &str) -> String {
    let task = rdm::lm::input_of(prompt, "task").unwrap_or_default();
    let (name, description, code) = function_for(task).unwrap_or_default();
    json!({"name": name, "description": description, "code": code}).to_string()
}

/// What the stand-in language model writes for a task: its name, its description and its
/// source. For "down to 18 degrees" it is a loop until it is 18, the function the Python tests
/// its reference page with, and for "three degrees cooler" a loop of three. A task that asks
/// for neither has none.
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
