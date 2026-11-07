//! Rlm: a language model in a loop over the sandbox.
//!
//! - One run: steps, what the model is shown of them, and how a run ends
//! - What code may ask of a language model, and how often
//! - The caller's checks, and a world that is put back
//! - An RLM in a program: a tool, with its tokens on the run
//! - A writer: a function is tried before it is on the map
//! - Decisions from code
#![cfg(feature = "rlm")]

mod common;

use std::sync::Arc;

use common::{client, scripted, shown};
use rdm::lm::{Lm, ScriptedLm, Type, input_of, lm15};
use rdm::repl::{Limits, Repl};
use rdm::rlm::{Rlm, deciding};
use rdm::{Error, Json, Rdm, Shared, Signature, Source, inputs};
use serde_json::json;

/// A language model that writes these cells, one for each step of a run, and the last of
/// them for every step after. Asked for the outputs when the turns ran out, it gives `read`.
fn writes<const N: usize>(cells: [&str; N], read: Json) -> (Lm, Arc<ScriptedLm>) {
    let cells: Vec<String> = cells.iter().map(|cell| (*cell).to_owned()).collect();
    let scripted = Arc::new(
        ScriptedLm::new(move |_system, prompt| {
            let Some(step) = input_of(prompt, "iteration") else {
                return read.to_string();
            };
            let at: usize = step.split('/').next().and_then(|at| at.parse().ok()).unwrap_or(1);
            let cell = cells.get(at - 1).or(cells.last()).cloned().unwrap_or_default();
            json!({"reasoning": format!("step {at}"), "code": cell}).to_string()
        })
        .tokens(100, 10),
    );
    (Lm::shared(scripted.clone(), "scripted"), scripted)
}

/// What a request showed the model under one of its fields.
fn field(scripted: &ScriptedLm, request: usize, name: &str) -> String {
    let asked = &scripted.requests()[request];
    let text = asked.messages.last().and_then(|message| message.text()).unwrap_or_default();
    input_of(&text, name).unwrap_or_default().to_owned()
}

/// What the model was told once, for every request: its instructions.
fn instructions(scripted: &ScriptedLm) -> String {
    match &scripted.requests()[0].system {
        Some(lm15::SystemContent::Text(text)) => text.clone(),
        _ => String::new(),
    }
}

type World = Shared<i64>;

/// A heater: `cooler()` makes it one degree cooler and `heat()` reads it.
fn heater(world: &World) -> Repl {
    let (cooled, read) = (world.clone(), world.clone());
    Repl::new()
        .function("cooler", "Make it one degree cooler.", move |_| {
            Ok(cooled.update(|heat| {
                *heat -= 1;
                format!("heat is {heat}")
            }))
        })
        .function("heat", "How warm it is, in degrees.", move |_| Ok(read.get()))
}

/// What puts a heater back as it was when a run began.
fn put_back(world: &World) -> impl Fn() -> Box<dyn FnOnce() + Send> + Send + Sync + 'static {
    let world = world.clone();
    move || {
        let (before, world) = (world.get(), world.clone());
        Box::new(move || {
            world.set(before);
        })
    }
}

fn failure<T: std::fmt::Debug>(result: rdm::Result<T>) -> String {
    match result {
        Err(Error::Tool(why)) => why,
        other => panic!("expected a failure, and got {other:?}"),
    }
}

// One run: steps, what the model is shown of them, and how a run ends

#[test]
fn a_run_is_steps_and_each_step_is_shown_the_ones_before() {
    let (lm, asked) = writes(["print(len(text))", "SUBMIT(text.upper())"], json!({}));
    let shout = Rlm::new("Say it louder.").input("text").output("loud").lm(lm);
    let ran = shout.call(inputs! { "text" => "quiet" }).unwrap();
    assert_eq!(ran.outputs["loud"], "QUIET");
    assert_eq!(ran.steps.len(), 2);
    assert_eq!(
        (ran.steps[0].code.as_str(), ran.steps[0].output.as_str()),
        ("print(len(text))", "5")
    );
    assert_eq!(ran.steps[1].output, r#"FINAL: {"loud":"QUIET"}"#);
    assert!(!ran.extracted);
    // Two requests, and the tokens of both.
    assert_eq!((asked.requests().len(), ran.tokens()), (2, 220));
    assert_eq!(
        field(&asked, 0, "repl_history"),
        "You have not interacted with the REPL environment yet."
    );
    assert_eq!(field(&asked, 0, "iteration"), "1/20");
    assert_eq!(
        field(&asked, 1, "repl_history"),
        "=== Step 1 ===\nReasoning: step 1\nCode:\n```python\nprint(len(text))\n```\nOutput (1 chars):\n5"
    );
}

#[test]
fn an_input_is_a_variable_and_the_model_is_told_what_it_is_and_not_what_it_holds() {
    let long = "a line of the handbook\n".repeat(400);
    let (lm, asked) = writes(["SUBMIT(len(text.splitlines()))"], json!({}));
    let count = Rlm::new("Count the lines.")
        .input_described("text", "The handbook.")
        .input_typed("limits", "dict")
        .output_typed("lines", Type::Int)
        .lm(lm);
    let ran = count
        .call(inputs! { "text" => long.as_str(), "limits" => json!({"hotel": 260}) })
        .unwrap();
    assert_eq!(ran.outputs["lines"], 400);
    let info = field(&asked, 0, "variables_info");
    assert!(info.starts_with("Variable: `text` (access it in your code)\nType: str\nDescription: The handbook.\nTotal length: 9,200 characters\nPreview:\n```\na line of"));
    assert!(info.contains("Variable: `limits` (access it in your code)\nType: dict\nTotal length: 18 characters"));
    // A thousand characters of it, and not the nine thousand.
    assert!(info.len() < 1600, "the whole input was shown: {} characters", info.len());
}

#[test]
fn code_that_fails_is_a_step_and_the_run_goes_on() {
    let (lm, _) = writes(["print(missing)", "```python\nSUBMIT('recovered')\n```"], json!({}));
    let ran = Rlm::parse("task -> said").unwrap().lm(lm).call(inputs! { "task" => "t" }).unwrap();
    assert!(ran.steps[0].output.starts_with("[Error] NameError"), "{}", ran.steps[0].output);
    // The fence a model puts round code is taken off.
    assert_eq!(
        (ran.steps[1].code.as_str(), ran.outputs["said"].as_str()),
        ("SUBMIT('recovered')", Some("recovered"))
    );
}

#[test]
fn what_failing_code_printed_before_it_failed_is_shown_with_why() {
    let (lm, _) = writes(["print('page one')\nprint(missing)", "SUBMIT('read')"], json!({}));
    let ran = Rlm::parse("task -> said").unwrap().lm(lm).call(inputs! { "task" => "t" }).unwrap();
    assert!(
        ran.steps[0].output.starts_with("page one\n[Error] NameError"),
        "{}",
        ran.steps[0].output
    );
}

#[test]
fn the_last_step_is_told_it_is_the_last() {
    let (lm, asked) = writes(["print(1)"], json!({"said": "read off"}));
    Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .max_iters(2)
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert_eq!(field(&asked, 0, "iteration"), "1/2");
    assert_eq!(
        field(&asked, 1, "iteration"),
        "2/2 - this is the last step: SUBMIT your best answer now"
    );
}

#[test]
fn where_the_decision_model_is_there_judgments_are_its_and_the_guidance_and_rules_can_be_given() {
    let (lm, asked) = writes(["SUBMIT('x')"], json!({}));
    let repl = deciding(Repl::new(), about_work());
    Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .repl(&repl)
        .call(inputs! { "task" => "t" })
        .unwrap();
    let told = instructions(&asked);
    assert!(told.contains("USE decide AND choose FOR JUDGMENTS"), "{told}");
    assert!(told.contains("USE llm_query FOR TEXT"), "{told}");
    assert!(!told.contains("500K"), "{told}");
    // Without the decision model, llm_query is for meaning, as before.
    let (lm, asked) = writes(["SUBMIT('x')"], json!({}));
    Rlm::parse("task -> said").unwrap().lm(lm).call(inputs! { "task" => "t" }).unwrap();
    let told = instructions(&asked);
    assert!(
        told.contains("USE llm_query FOR SEMANTICS") && !told.contains("FOR JUDGMENTS"),
        "{told}"
    );
    // An optimizer's guidance and rules stand in for the defaults.
    let (lm, asked) = writes(["SUBMIT('x')"], json!({}));
    Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .guidance("Answer in one step when the inputs are enough.")
        .rules(["LOOK ONCE", "SUBMIT EARLY"])
        .call(inputs! { "task" => "t" })
        .unwrap();
    let told = instructions(&asked);
    assert!(
        told.contains("Answer in one step when the inputs are enough.\n\n1. LOOK ONCE\n2. SUBMIT EARLY\n\n"),
        "{told}"
    );
    assert!(!told.contains("ITERATIVE") && !told.contains("EXPLORE FIRST"), "{told}");
}

#[test]
fn code_that_prints_nothing_is_told_so() {
    let (lm, _) = writes(["x = 1", "SUBMIT(str(x))"], json!({}));
    let ran = Rlm::parse("task -> said").unwrap().lm(lm).call(inputs! { "task" => "t" }).unwrap();
    assert_eq!(ran.steps[0].output, "(no output - did you forget to print?)");
    // What a step leaves in the REPL, the next has.
    assert_eq!(ran.outputs["said"], "1");
}

#[test]
fn submit_ends_the_step_where_it_is_called() {
    let world = Shared::new(21);
    let (lm, _) = writes(
        ["print('before')\ntry:\n    SUBMIT(heat())\nexcept Exception:\n    cooler()\ncooler()"],
        json!({}),
    );
    let ran = Rlm::parse("task -> heat: int")
        .unwrap()
        .lm(lm)
        .repl(&heater(&world))
        .call(inputs! { "task" => "t" });
    assert_eq!(ran.unwrap().outputs["heat"], 21);
    // Nothing after it ran, and it was not caught.
    assert_eq!(world.get(), 21);
}

#[test]
fn what_is_submitted_is_read_as_its_type_or_the_model_is_told() {
    let (lm, _) = writes(["SUBMIT('many')", "SUBMIT()", "SUBMIT(count='3')"], json!({}));
    let ran = Rlm::parse("task -> count: int")
        .unwrap()
        .lm(lm)
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert_eq!(
        ran.steps[0].output,
        "[Error] What was submitted is not the outputs. `count` is declared `int`, and it is a string."
    );
    assert!(
        ran.steps[1]
            .output
            .starts_with("[Error] TypeError: SUBMIT() missing required argument: 'count'"),
        "{}",
        ran.steps[1].output
    );
    // A whole number written as text is that number.
    assert_eq!(ran.outputs["count"], 3);
}

#[test]
fn out_of_turns_the_outputs_are_read_off_the_steps() {
    let (lm, asked) = writes(["print(2 + 2)"], json!({"sum": 4}));
    let ran = Rlm::parse("task -> sum: int")
        .unwrap()
        .lm(lm)
        .max_iters(3)
        .call(inputs! { "task" => "Add two and two." })
        .unwrap();
    assert_eq!((ran.outputs["sum"].clone(), ran.steps.len(), ran.extracted), (json!(4), 3, true));
    // Three steps and the request that read them: it was shown all three.
    assert_eq!(asked.requests().len(), 4);
    assert!(field(&asked, 3, "repl_history").contains("=== Step 3 ==="));
}

#[test]
fn a_long_output_is_shown_by_its_ends() {
    let (lm, asked) = writes(["print('a' * 60 + 'b' * 60)", "SUBMIT('done')"], json!({}));
    let rlm = Rlm::parse("task -> said").unwrap().lm(lm).max_output_chars(20);
    let ran = rlm.call(inputs! { "task" => "t" }).unwrap();
    // The step keeps what was printed, and the model is shown ten characters from each end.
    assert_eq!(ran.steps[0].output.len(), 120);
    assert!(
        field(&asked, 1, "repl_history").ends_with("Output (120 chars):\naaaaaaaaaa\n\n... (100 characters omitted) ...\n\nbbbbbbbbbb")
    );
}

#[test]
fn a_run_is_refused_before_anything_is_asked_where_it_is_set_up_wrongly() {
    let (lm, asked) = writes(["SUBMIT('x')"], json!({}));
    let usage = |result: rdm::Result<rdm::rlm::Ran>| match result {
        Err(error) if error.is_usage() => error.to_string(),
        other => panic!("expected a usage error, and got {other:?}"),
    };
    let rlm = Rlm::parse("task -> said").unwrap();
    assert!(usage(rlm.call(inputs! { "task" => "t" })).starts_with("The RLM has no language model"));
    let rlm = rlm.lm(lm.clone());
    assert_eq!(
        usage(rlm.call(inputs! { "other" => "t" })),
        "The call has inputs ['other']; its signature takes ['task']."
    );
    // An input is a variable, and a variable cannot have the name of a function.
    let named = Rlm::parse("heat -> said").unwrap().lm(lm).repl(&heater(&Shared::new(21)));
    assert!(usage(named.call(inputs! { "heat" => "t" })).starts_with("The input 'heat' has the name of a function in the REPL"));
    assert!(asked.requests().is_empty());
}

// What code may ask of a language model, and how often

#[test]
fn code_asks_a_language_model_and_is_held_to_a_count() {
    let sub = Arc::new(ScriptedLm::new(|_system, prompt| format!("about {prompt}")).tokens(7, 3));
    let (lm, _) = writes(
        [
            "print(llm_query('heat'))\nprint(llm_query_batched(['doors', 'locks']))",
            "print(llm_query('one too many'))",
            "SUBMIT('done')",
        ],
        json!({}),
    );
    let rlm = Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .sub_lm(Lm::shared(sub.clone(), "sub"))
        .max_llm_calls(3);
    let ran = rlm.call(inputs! { "task" => "t" }).unwrap();
    assert_eq!(ran.steps[0].output, "about heat\n['about doors', 'about locks']");
    assert_eq!(
        ran.steps[1].output,
        "[Error] RuntimeError: LLM call limit exceeded: 3 + 1 > 3. Use Python code for aggregation instead of making more LLM calls."
    );
    // The model that works was asked three times, and the one the code asks, three.
    assert_eq!(sub.requests().len(), 3);
    assert_eq!(ran.tokens(), 3 * 110 + 3 * 10);
}

#[test]
fn with_no_calls_allowed_the_code_is_given_no_way_to_ask() {
    let (lm, asked) = writes(["print(llm_query('heat'))", "SUBMIT('done')"], json!({}));
    let ran = Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .max_llm_calls(0)
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert!(ran.steps[0].output.starts_with("[Error] NameError"), "{}", ran.steps[0].output);
    let told = instructions(&asked);
    assert!(!told.contains("llm_query") && told.contains("4. MINIMIZE RETYPING"), "{told}");
}

#[test]
fn the_model_is_told_the_functions_it_may_call() {
    let world = Shared::new(21);
    let repl = heater(&world);
    repl.define(
        "cool_two",
        "Make it two degrees cooler.",
        "def cool_two():\n    cooler()\n    cooler()",
    )
    .unwrap();
    let (lm, asked) = writes(["cool_two()\nSUBMIT(heat())"], json!({}));
    let rlm = Rlm::parse("task -> heat: int").unwrap().lm(lm).repl(&repl);
    // What was written before is in the scratch as well: a step can call it.
    assert_eq!(rlm.call(inputs! { "task" => "t" }).unwrap().outputs["heat"], 19);
    let told = instructions(&asked);
    assert!(told.contains("- `cooler()` - Make it one degree cooler."), "{told}");
    assert!(
        told.contains("Functions written before, which code may call as well:\n- `cool_two` - Make it two degrees cooler."),
        "{told}"
    );
    assert!(told.contains("- `SUBMIT(heat)` - submit final output when done"), "{told}");
    // Nothing a run defines is left in the REPL it was given.
    assert_eq!(repl.written().len(), 1);
    assert!(repl.execute("SUBMIT(1)").unwrap_err().to_string().starts_with("NameError"));
}

// The caller's checks, and a world that is put back

#[test]
fn a_check_that_fails_is_what_the_model_reads_next() {
    let (lm, asked) = writes(["SUBMIT(7)", "SUBMIT(len(words))"], json!({}));
    let count = Rlm::parse("words: list[str] -> count: int")
        .unwrap()
        .lm(lm)
        .check("counted", |submitted| {
            let said = submitted.run("print(len(words))")?;
            match submitted.get("count") {
                Some(count) if count.as_i64() == said.parse().ok() => Ok(()),
                _ => Err(format!("there are {said} words")),
            }
        });
    let ran = count.call(inputs! { "words" => ["a", "b", "c"] }).unwrap();
    assert_eq!(ran.outputs["count"], 3);
    assert_eq!(ran.steps[0].output, "[Not accepted] counted: there are 3 words");
    assert!(field(&asked, 1, "repl_history").ends_with("[Not accepted] counted: there are 3 words"));
}

#[test]
fn outputs_read_off_the_steps_are_checked_too() {
    let (lm, _) = writes(["print('thinking')"], json!({"count": 7}));
    let count = Rlm::parse("task -> count: int").unwrap().lm(lm).max_iters(2);
    let count = count.check("small", |submitted| {
        if submitted.get("count") == Some(&json!(3)) {
            Ok(())
        } else {
            Err("it is not 3".to_owned())
        }
    });
    assert_eq!(
        failure(count.call(inputs! { "task" => "t" })),
        "The language model did not submit outputs that were accepted in 2 steps, and nor were the outputs read off its \
         steps. [Not accepted] small: it is not 3"
    );
}

#[test]
fn a_checkpoint_is_put_back_however_the_run_ends() {
    let world = Shared::new(21);
    let repl = heater(&world);
    let (lm, _) = writes(["cooler()\ncooler()\nSUBMIT(heat())"], json!({}));
    let tried = Rlm::parse("task -> heat: int")
        .unwrap()
        .lm(lm)
        .repl(&repl)
        .checkpoint(put_back(&world));
    // It was 19 where the code ran, and it is 21 again.
    assert_eq!(tried.call(inputs! { "task" => "t" }).unwrap().outputs["heat"], 19);
    assert_eq!(world.get(), 21);

    // A run that fails puts it back as well.
    let (lm, _) = writes(["cooler()"], json!({"heat": "warm"}));
    let failed = Rlm::parse("task -> heat: int")
        .unwrap()
        .lm(lm)
        .repl(&repl)
        .max_iters(2)
        .checkpoint(put_back(&world));
    failure(failed.call(inputs! { "task" => "t" }));
    assert_eq!(world.get(), 21);

    // With none, trying is acting.
    let (lm, _) = writes(["cooler()\nSUBMIT(heat())"], json!({}));
    Rlm::parse("task -> heat: int")
        .unwrap()
        .lm(lm)
        .repl(&repl)
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert_eq!(world.get(), 20);
}

#[test]
fn each_step_is_told_to_whoever_watches() {
    let seen = Shared::new(Vec::new());
    let sees = seen.clone();
    let (lm, _) = writes(["print(1)", "SUBMIT('done')"], json!({}));
    let rlm = Rlm::parse("task -> said")
        .unwrap()
        .lm(lm)
        .watch(move |step| sees.lock().push(step.output.clone()));
    rlm.call(inputs! { "task" => "t" }).unwrap();
    assert_eq!(*seen.lock(), ["1", r#"FINAL: {"said":"done"}"#]);
}

// An RLM in a program: a tool, with its tokens on the run

#[test]
fn an_rlm_is_a_tool_and_what_it_took_is_the_requests() {
    let (lm, _) = writes(["print(len(text))", "SUBMIT(len(text.split()))"], json!({}));
    let count = Rlm::new("Count the words.").input("text").output_typed("words", Type::Int).lm(lm);
    let route = Signature::new("Route a request.")
        .input("text")
        .noul("count", "Does `inputs.text` need its words counted?");
    let rdm = Rdm::new(route)
        .tool(count.tool("count", "Count the words."))
        .model(client(|_, _, _| true));
    let result = rdm.call(inputs! { "text" => "one two three" }).unwrap();
    // The observation is the outputs, and not the steps that came to them.
    assert_eq!(result.observations["count"], json!({"words": 3}));
    assert_eq!(result.trajectory[0].tool_tokens.as_ref().unwrap()["count"], 220);
}

// A writer: a function is tried before it is on the map

const COOL: &str = "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return heat()";
const WRITE: &str = "Have a new tool written in code: nothing listed does what is still needed.";

fn operate() -> Signature {
    Signature::new("Operate a heater, one step at a time.")
        .input("task")
        .input("observation")
        .tool("next", "What should happen next for `inputs.task`?")
}

fn writer(lm: Lm) -> Rlm {
    Rlm::new("Write one Python function that does what the task asks.")
        .input("task")
        .lm(lm)
        .max_iters(4)
}

#[test]
fn a_function_is_tried_where_trying_changes_nothing_and_then_it_is_on_the_map() {
    let world = Shared::new(21);
    let repl = heater(&world);
    let cell = format!("{COOL}\n\nprint(cool_to_18())");
    let (lm, _) = writes(
        [cell.as_str(), "SUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')"],
        json!({}),
    );
    let write = writer(lm).checkpoint(put_back(&world)).writer(&repl).tool("write", WRITE);
    let model = scripted(["write", "cool_to_18", "finish"]);
    let rdm = Rdm::new(operate()).tool(write).source(repl.clone()).model(model.clone());
    let result = rdm.call(inputs! { "task" => "Bring the heat down to 18 degrees." }).unwrap();

    assert_eq!(result.pointed("next"), ["write", "cool_to_18", "finish"]);
    // What the decision model read after `write`: it is on the list, and has not been run.
    let observed = shown(&model)[1]["observation"].as_str().unwrap().to_owned();
    assert_eq!(
        observed,
        "`cool_to_18` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Bring the heat down to 18 degrees."
    );
    // The source is the one a step defined and tried, with nothing of the step beside it.
    assert_eq!(repl.written()["cool_to_18"].code, COOL);
    assert_eq!(
        result.trajectory[0].changes,
        Some(vec![json!({"by": "repl", "did": "define", "name": "cool_to_18"})])
    );
    // Trying it cooled the house and the house was put back. Pointing at it cooled the house.
    assert_eq!(result.steps("next")[1].1, "18");
    assert_eq!(world.get(), 18);
}

#[test]
fn a_candidate_is_refused_in_the_words_define_has_and_the_model_writes_again() {
    let world = Shared::new(21);
    let repl = heater(&world);
    let good = format!("{COOL}\n\nSUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')");
    let (lm, _) = writes(
        [
            // No step defines it.
            "SUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')",
            // A name the map has.
            "def finish():\n    return 1\n\nSUBMIT('finish', 'Finish.')",
            // It leans on a variable a step left in the scratch, where it ran without fault.
            "limit = 18\ndef cool():\n    while heat() > limit:\n        cooler()\n    return heat()\n\nprint(cool())\nSUBMIT('cool', 'Cool to the limit.')",
            // It fails when it is tried.
            "def broken():\n    return heat() / 0\n\nSUBMIT('broken', 'Divide the heat.')",
            // No description.
            "def cool_to_18():\n    return 1\n\nSUBMIT('cool_to_18', '')",
            good.as_str(),
        ],
        json!({}),
    );
    let writer = writer(lm).max_iters(6).checkpoint(put_back(&world)).writer(&repl);
    let wrote = writer.write(inputs! { "task" => "Bring the heat down to 18 degrees." }).unwrap();
    let outputs: Vec<&str> = wrote.ran.steps.iter().map(|step| step.output.as_str()).collect();
    assert!(
        outputs[0].starts_with("[Not accepted] No step defines a function `cool_to_18` at its top level."),
        "{}",
        outputs[0]
    );
    assert_eq!(
        outputs[1],
        "[Not accepted] `finish` is already the name of a tool. Give the function another name."
    );
    // It printed 18 in the scratch, and alone it has no `limit`.
    assert!(
        outputs[2].starts_with("18\n[Not accepted] `cool` failed when it was tried on its own. NameError"),
        "{}",
        outputs[2]
    );
    assert!(
        outputs[3].starts_with("[Not accepted] `broken` failed when it was tried on its own. ZeroDivisionError"),
        "{}",
        outputs[3]
    );
    assert!(outputs[4].starts_with("[Not accepted] Give a description"), "{}", outputs[4]);
    // Only what was accepted is in the REPL: nothing of the five that were not.
    assert_eq!((wrote.name.as_str(), wrote.code.as_str()), ("cool_to_18", COOL));
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["cool_to_18"]);
    assert_eq!(world.get(), 21);
}

#[test]
fn the_callers_checks_are_given_the_function_and_the_scratch_it_was_tried_in() {
    let world = Shared::new(21);
    let repl = heater(&world);
    let lazy = "def cool_to_18():\n    cooler()\n    return heat()\n\nSUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')";
    let good = format!("{COOL}\n\nSUBMIT('cool_to_18', 'Bring the heat down to 18 degrees.')");
    let (lm, _) = writes([lazy, good.as_str()], json!({}));
    let writer = writer(lm).checkpoint(put_back(&world)).writer(&repl).check("comes to 18", |tried| {
        // What it printed when it was tried, and the scratch, to see what trying left.
        let now = tried.run("print(heat())")?;
        if tried.printed == Some("18") && now == "18" {
            Ok(())
        } else {
            Err(format!("it left the heat at {now}"))
        }
    });
    let wrote = writer.write(inputs! { "task" => "Bring the heat down to 18 degrees." }).unwrap();
    assert_eq!(wrote.ran.steps[0].output, "[Not accepted] comes to 18: it left the heat at 20");
    assert_eq!((wrote.code.as_str(), world.get()), (COOL, 21));
}

#[test]
fn stand_ins_are_what_a_writer_tries_against_where_an_action_cannot_be_put_back() {
    let world = Shared::new(21);
    let repl = heater(&world);
    // The same names, and nothing changed: each call is written down.
    let record = Shared::new(Vec::<String>::new());
    let (called, starts) = (record.clone(), record.clone());
    let rehearsal = repl.spawn("rehearsal").function("cooler", "Make it one degree cooler.", move |_| {
        called.lock().push("cooler".to_owned());
        Ok("heat is one lower")
    });
    let cell = "def cool_two():\n    cooler()\n    cooler()\n    return 'cooled'\n\nprint(cool_two())";
    let (lm, _) = writes([cell, "SUBMIT('cool_two', 'Make it two degrees cooler.')"], json!({}));
    let seen = record.clone();
    let writer = writer(lm)
        .repl(&rehearsal)
        .writer(&repl)
        // The record starts again where the candidate is tried: what the model did while it explored is not the candidate.
        .trying(move |_| starts.lock().clear())
        .check("two calls", move |_| {
            if seen.lock().len() == 2 {
                Ok(())
            } else {
                Err(format!("it called {:?}", seen.lock()))
            }
        });
    let wrote = writer.write(inputs! { "task" => "Two degrees cooler." }).unwrap();
    assert_eq!(wrote.name, "cool_two");
    // Tried twice over, by the model and then by the writer, and the house was never touched.
    assert_eq!(world.get(), 21);
    // It is kept in the real REPL, where `cooler` is the real one.
    assert_eq!(repl.execute("print(cool_two())").unwrap(), "cooled");
    assert_eq!(world.get(), 19);
}

#[test]
fn a_function_that_takes_what_nothing_names_is_not_tried_and_the_check_is_told() {
    let repl = heater(&Shared::new(21));
    let cell = "def cool_by(degrees):\n    for _ in range(degrees):\n        cooler()\n    return heat()\n\nSUBMIT('cool_by', 'Make it cooler by some degrees.')";
    let (lm, _) = writes([cell], json!({}));
    let tried_with = Shared::new(None);
    let keeps = tried_with.clone();
    let writer = writer(lm).writer(&repl).check("seen", move |tried| {
        keeps.set(Some(tried.printed.map(str::to_owned)));
        Ok(())
    });
    writer.write(inputs! { "task" => "Cooler by some degrees." }).unwrap();
    assert_eq!(tried_with.get(), Some(None));

    // With an input of that name, it is tried with it.
    let (lm, _) = writes([cell], json!({}));
    let keeps = tried_with.clone();
    let writer = writer_with_degrees(lm).writer(&repl).check("seen", move |tried| {
        keeps.set(Some(tried.printed.map(str::to_owned)));
        Ok(())
    });
    writer
        .write(inputs! { "task" => "Cooler by some degrees.", "degrees" => 2 })
        .unwrap();
    assert_eq!(tried_with.get(), Some(Some("19".to_owned())));
}

fn writer_with_degrees(lm: Lm) -> Rlm {
    writer(lm).input_typed("degrees", "int")
}

#[test]
fn a_writer_that_never_comes_to_a_function_puts_nothing_on_the_list() {
    let repl = heater(&Shared::new(21));
    let (lm, _) = writes(
        ["print('thinking')"],
        json!({"name": "heat", "description": "Read it.", "code": "def heat():\n    return 1"}),
    );
    let write = writer(lm).max_iters(2).writer(&repl).tool("write", WRITE);
    let model = scripted(["write", "finish"]);
    let rdm = Rdm::new(operate()).tool(write).source(repl.clone()).model(model.clone());
    rdm.call(inputs! { "task" => "Bring the heat down." }).unwrap();
    let observed = shown(&model)[1]["observation"].as_str().unwrap().to_owned();
    assert!(
        observed.starts_with("Nothing was put on the list. The language model did not submit outputs that were accepted in 2 steps"),
        "{observed}"
    );
    assert!(
        observed.ends_with("`heat` is already the name of a tool. Give the function another name."),
        "{observed}"
    );
    assert!(repl.offers().is_empty());
}

#[test]
fn a_name_on_the_map_of_the_run_cannot_be_taken() {
    let repl = heater(&Shared::new(21));
    let (lm, _) = writes(
        [
            "def write():\n    return 1\n\nSUBMIT('write', 'Write.')",
            "def task():\n    return 1\n\nSUBMIT('task', 'Task.')",
            "def warm():\n    return heat()\n\nSUBMIT('warm', 'Read the heat.')",
        ],
        json!({}),
    );
    let steps = Shared::new(Vec::new());
    let sees = steps.clone();
    let write = writer(lm)
        .watch(move |step| sees.lock().push(step.output.clone()))
        .writer(&repl)
        .tool("write", WRITE);
    let rdm = Rdm::new(operate())
        .tool(write)
        .source(repl.clone())
        .model(scripted(["write", "finish"]));
    rdm.call(inputs! { "task" => "Read the heat." }).unwrap();
    let steps = steps.get();
    assert_eq!(
        steps[0],
        "[Not accepted] `write` is already the name of a tool. Give the function another name."
    );
    assert_eq!(
        steps[1],
        "[Not accepted] `task` is already the name of an input of the request. Give the function another name."
    );
    assert_eq!(repl.written().keys().collect::<Vec<_>>(), ["warm"]);
}

// Decisions from code

fn about_work() -> Arc<rdm::Scripted> {
    client(|inputs, _, question| {
        let about = inputs["about"].as_str().unwrap_or_default();
        match question.labels().is_empty() {
            true => common::Say::Yes(about.contains("work")),
            false => common::Say::Pick(
                question
                    .labels()
                    .into_iter()
                    .find(|label| about.contains(label.as_str()))
                    .unwrap_or_default(),
            ),
        }
    })
}

#[test]
fn code_asks_the_decision_model_and_loops_over_what_it_says() {
    let model = about_work();
    let repl = deciding(Repl::new(), model.clone());
    let code =
        "apps = ['work mail', 'music', 'work chat']\nprint([app for app in apps if decide('Is `inputs.about` used for work?', app) > 0.5])";
    assert_eq!(repl.execute(code).unwrap(), "['work mail', 'work chat']");
    assert_eq!(
        repl.execute("print(choose('Which room is `inputs.about` in?', ['kitchen', 'hall'], 'the hall light'))")
            .unwrap(),
        "hall"
    );
    // Three yes-or-no questions and one choice, each a request of its own.
    assert_eq!(model.requests().len(), 4);
    assert_eq!(
        model.requests()[0].questions["answer"].instructions,
        "Is `inputs.about` used for work?"
    );
    assert!(
        repl.execute("choose('Which?', [])")
            .unwrap_err()
            .to_string()
            .contains("choose() needs at least one option")
    );
}

#[test]
fn a_written_function_that_decides_asks_as_part_of_the_run_that_pointed_at_it() {
    // The model given to `deciding` is for code that no request runs: here a request does.
    let unused = client(|_, _, _| -> bool { panic!("the run's own model is the one to ask") });
    let repl = deciding(Repl::new(), unused);
    repl.define(
        "work_apps",
        "Which apps are for work.",
        "def work_apps():\n    return [app for app in ['work mail', 'music'] if decide('Is `inputs.about` used for work?', app) > 0.5]",
    )
    .unwrap();
    let model = client(|inputs, name, question| match name {
        "next" if inputs["observation"] == "" => common::Say::Pick("work_apps".to_owned()),
        "next" => common::Say::Pick("finish".to_owned()),
        _ => {
            assert!(question.labels().is_empty());
            common::Say::Yes(inputs["about"].as_str().unwrap_or_default().contains("work"))
        }
    });
    let rdm = Rdm::new(operate()).source(repl).model(model.clone());
    let result = rdm.call(inputs! { "task" => "Which apps are for work?" }).unwrap();
    assert_eq!(result.steps("next")[0].1, "['work mail']");
    // Two requests of the pointer, and the two the function asked, one level down in the same run.
    assert_eq!(result.trajectory.len(), 4);
    assert_eq!(result.trajectory.iter().filter(|entry| entry.depth == 1).count(), 2);
}

#[test]
fn written_code_is_held_to_the_limits_of_the_repl_it_runs_in() {
    let world = Shared::new(1000);
    let repl = heater(&world).limits(Limits {
        calls: 5,
        ..Limits::default()
    });
    let (lm, _) = writes(["while True:\n    cooler()", "SUBMIT(heat())"], json!({}));
    let ran = Rlm::parse("task -> heat: int")
        .unwrap()
        .lm(lm)
        .repl(&repl)
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert!(ran.steps[0].output.contains("more than 5 times"), "{}", ran.steps[0].output);
}
