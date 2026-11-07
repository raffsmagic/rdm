//! What ends a run and what does not: a tool's failure is its observation, whatever the tool
//! failed with and whatever the model answered, and a module used wrongly is an error before
//! anything is sent.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use common::{Say, client, scripted};
use rdm::systemone::{Answer, Kind, Question, Scripted};
use rdm::{Ask, Error, Json, Opt, Rdm, Reason, Shared, Signature, Tool, inputs};
use serde_json::json;

/// Says yes, chooses the first option, and scores the lowest level.
fn agreeable() -> Arc<Scripted> {
    client(|_: &Json, _: &str, question: &Question| match question.kind {
        Kind::Noul => Say::Yes(true),
        Kind::Choice => Say::Pick(question.labels().remove(0)),
        Kind::Score => Say::Level(0),
    })
}

/// What a panic said, for a test that expects one.
fn said(panic: Box<dyn std::any::Any + Send>) -> String {
    let text = panic.downcast_ref::<&str>().map(|text| text.to_string());
    text.or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default()
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn a_tool_that_panics_has_failed_and_that_is_its_observation() {
    // In the Python module any exception a tool raises is its observation. A Rust tool that
    // indexes past the end, or reads a key that is not there, panics: it is the same failure.
    let wait = Signature::new("Wait for a door.")
        .input("door")
        .noul("knock", "Is `inputs.door` shut?");
    let knock = Tool::new("knock", "Knock.", |_| {
        let devices: std::collections::HashMap<&str, &str> = [("tv", "off")].into();
        Ok(devices["door"].to_owned())
    });
    let rdm = Rdm::new(wait).tool(knock).model(agreeable());
    let result = rdm.call(inputs! { "door" => "front" }).unwrap();
    let seen = result.observations["knock"].as_str().unwrap();
    assert!(seen.starts_with("Execution error in knock: "), "{seen}");
    assert_eq!(result.termination_reason, Reason::Final);
    assert_eq!(result.trajectory[0].observations.as_ref().unwrap()["knock"], seen);
    // What the panic said is the error, as what a returned error says is.
    let rdm = Rdm::new(Signature::new("Wait.").input("door").noul("knock", "Knock?"))
        .tool(Tool::new("knock", "", |_| -> rdm::Result<()> {
            panic!("the door is {}", "jammed")
        }))
        .model(agreeable());
    assert_eq!(
        rdm.call(inputs! { "door" => "front" }).unwrap().observations["knock"],
        "Execution error in knock: the door is jammed"
    );
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn one_of_several_tools_that_panics_does_not_take_the_others_with_it() {
    let two = Signature::new("Two.").input("x").noul("a", "A?").noul("b", "B?").noul("c", "C?");
    let rdm = Rdm::new(two)
        .tool(Tool::new("a", "", |_| -> rdm::Result<()> { panic!("the ledger did not answer") }))
        .tool(Tool::new("b", "", |_| Ok("paged")))
        .tool(Tool::new("c", "", |_| -> rdm::Result<()> { Err("the queue is full".into()) }))
        .model(agreeable());
    let result = rdm.call(inputs! { "x" => 1 }).unwrap();
    assert_eq!(
        json!(result.observations),
        json!({
            "a": "Execution error in a: the ledger did not answer",
            "b": "paged",
            "c": "Execution error in c: the queue is full",
        })
    );
    assert_eq!(result.trajectory.len(), 1);
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn a_panic_under_a_tool_is_that_tools_failure_and_the_loop_goes_on() {
    // A pointer's loop is shown the failure, and can point at something else.
    let steps = Signature::new("Take steps.").input("observation").tool("next", "What next?");
    let model = scripted(["slip", "step", "finish"]);
    let rdm = Rdm::new(steps)
        .tool(Tool::new("slip", "", |_| -> rdm::Result<()> { panic!("slipped") }))
        .tool(Tool::new("step", "", |_| Ok("stepped")))
        .model(model.clone());
    let result = rdm.call(inputs! {}).unwrap();
    assert_eq!(
        result.observations["next"],
        json!([["slip", "Execution error in slip: slipped"], ["step", "stepped"], ["finish", null]])
    );
    assert_eq!(
        model.requests()[1].state["inputs"]["observation"],
        "Execution error in slip: slipped"
    );
    // A panic in a sub-call, two levels down, is the failure of the tool nearest it.
    let count = Signature::new("Count.").input_typed("n", "int").noul("again", "Again?");
    let again = Tool::new("again", "", |call| {
        let n: i64 = call.arg("n")?;
        assert!(n < 2, "counted too far");
        Ok(call.sub(inputs! { "n" => n + 1 })?.observations.get("again").cloned())
    });
    let result = Rdm::new(count)
        .tool(again)
        .max_depth(5)
        .model(agreeable())
        .call(inputs! { "n" => 0 })
        .unwrap();
    assert_eq!(result.trajectory.len(), 3);
    assert_eq!(result.observations["again"], "Execution error in again: counted too far");
}

#[test]
#[cfg_attr(panic = "abort", ignore = "a panic cannot be caught here, and ends the process")]
fn a_panic_that_is_no_tools_goes_on_to_the_caller_as_it_was_raised() {
    // A model that panics at the root is nobody's observation. Several runs at once are on
    // threads of their own, and the caller is still given what the panic said.
    let model = Scripted::new(|request| {
        assert!(request.state["inputs"]["n"] != 2, "the model cannot count to two");
        [("again", Answer::noul(0.1))]
    });
    let rdm = Rdm::new(Signature::new("Count.").input_typed("n", "int").noul("again", "Again?")).model(model);
    let one = catch_unwind(AssertUnwindSafe(|| rdm.call(inputs! { "n" => 2 })));
    assert_eq!(said(one.unwrap_err()), "the model cannot count to two");
    let several = catch_unwind(AssertUnwindSafe(|| {
        rdm.batch([inputs! { "n" => 1 }, inputs! { "n" => 2 }, inputs! { "n" => 3 }])
    }));
    assert_eq!(said(several.unwrap_err()), "the model cannot count to two");
    // And so does a fit's, which is not a tool either.
    let fit = |_: Signature, _: rdm::Inputs, _: &rdm::Call| -> rdm::Result<(Signature, rdm::Inputs)> { panic!("the fit lost its place") };
    let rdm = rdm.copy().fit_fn(fit).model(agreeable());
    assert_eq!(
        said(catch_unwind(AssertUnwindSafe(|| rdm.call(inputs! { "n" => 1 }))).unwrap_err()),
        "the fit lost its place"
    );
}

/// A switch, a device that may be none of them, and a tool that takes the device.
fn act() -> Signature {
    Signature::new("Operate a smart home.")
        .input("task")
        .noul("switch", "Does `inputs.task` ask for a device to be switched?")
        .choice(
            "device",
            "Which device does `inputs.task` mean?",
            [Opt::new("tv", ""), Opt::new("lights", ""), Opt::none("The task names no device.")],
        )
}

#[test]
fn an_answer_read_as_a_type_it_is_not_is_the_tools_failure_and_not_the_end_of_the_run() {
    // Whether a run survives depends on how the module was used, and never on what the model
    // answered. Here it says yes to `switch` and chooses none of the devices.
    let model = || client(|_, name, _| if name == "switch" { Say::Yes(true) } else { "None".into() });
    let careless = Tool::new("switch", "", |call| {
        let device: String = call.arg("device")?;
        Ok(format!("{device}: on"))
    });
    let result = Rdm::new(act())
        .tool(careless)
        .model(model())
        .call(inputs! { "task" => "Switch it on." })
        .unwrap();
    let seen = result.observations["switch"].as_str().unwrap();
    assert!(
        seen.starts_with("Execution error in switch: The answer 'device' is read as a type it is not"),
        "{seen}"
    );
    assert!(seen.contains("read it as an `Option`"), "{seen}");
    assert_eq!((result.termination_reason, result.trajectory.len()), (Reason::Final, 1));
    // Read as what it can be, choosing none is a None.
    let careful = Tool::new("switch", "", |call| {
        let device: Option<String> = call.arg("device")?;
        Ok(device.map_or("nothing to switch".to_owned(), |device| format!("{device}: on")))
    });
    let result = Rdm::new(act())
        .tool(careful.clone())
        .model(model())
        .call(inputs! { "task" => "Switch it on." })
        .unwrap();
    assert_eq!(result.observations["switch"], "nothing to switch");
    let chose = client(|_, name, _| if name == "switch" { Say::Yes(true) } else { "lights".into() });
    let result = Rdm::new(act())
        .tool(careful)
        .model(chose)
        .call(inputs! { "task" => "Lights." })
        .unwrap();
    assert_eq!(result.observations["switch"], "lights: on");
    // An answer of another kind than it is read as is the tool's failure too: a yes is not a number.
    let miscast = Tool::new("switch", "", |call| call.arg::<u64>("switch"));
    let result = Rdm::new(act())
        .tool(miscast)
        .model(model())
        .call(inputs! { "task" => "t" })
        .unwrap();
    assert!(
        result.observations["switch"]
            .as_str()
            .unwrap()
            .starts_with("Execution error in switch: ")
    );
    // What stays a usage error is what the code got wrong: a name that is no answer and no
    // input, and an input, which the caller gave, read as a type it is not.
    for wrong in [
        Tool::new("switch", "", |call| call.arg::<String>("devise")),
        Tool::new("switch", "", |call| call.arg::<u64>("task")),
    ] {
        let error = Rdm::new(act())
            .tool(wrong)
            .model(model())
            .call(inputs! { "task" => "t" })
            .unwrap_err();
        assert!(error.is_usage(), "{error:?}");
    }
}

#[test]
fn a_question_that_cannot_be_asked_is_refused_before_anything_is_sent() {
    let nothing: [&str; 0] = [];
    // A choice with no options: the Python module refuses to make the type.
    let empty = Signature::new("Classify.")
        .input("listing")
        .choice("category", "Which category?", nothing);
    let model = agreeable();
    let rdm = Rdm::new(empty.clone()).model(model.clone());
    let error = rdm.call(inputs! { "listing" => "x" }).unwrap_err();
    assert_eq!(
        error,
        Error::Usage("Output 'category' is a choice with no options: a choice has at least one.".to_owned())
    );
    assert_eq!(rdm.check().unwrap_err(), error);
    assert!(model.requests().is_empty());
    // A score has two to ten levels.
    for levels in [1, 11] {
        let rated =
            Signature::new("Rate.")
                .input("listing")
                .score("rating", "How good?", (0..levels).map(|level| format!("level {level}")));
        let error = Rdm::new(rated).model(model.clone()).call(inputs! { "listing" => "x" }).unwrap_err();
        assert!(
            error.is_usage() && error.to_string().contains("a score has two to ten levels"),
            "{error:?}"
        );
    }
    assert!(model.requests().is_empty());
    // Two and ten are scores, and one option is a choice: it is answered without being asked.
    let fine = Signature::new("Rate.")
        .input("listing")
        .score("rating", "How good?", (0..10).map(|level| format!("level {level}")))
        .score("worth", "Worth it?", ["no", "yes"])
        .choice("category", "Which category?", ["Kitchen"]);
    let result = Rdm::new(fine).model(model.clone()).call(inputs! { "listing" => "x" }).unwrap();
    assert_eq!((result.chosen("category"), model.requests().len()), (Some("Kitchen"), 1));

    // The natural way to meet it: a walk down a taxonomy that reaches a node with nothing under
    // it. The request is not recorded, not counted and not sent: it is the caller's error, as
    // a pointer with nothing to point at is.
    let model = agreeable();
    let descend = Tool::new("category", "", move |call| {
        call.sub_with(Ask::new(inputs! { "listing" => "x" }).signature(empty.clone()))
    });
    let top = Signature::new("Classify.")
        .input("listing")
        .choice("category", "Which category?", ["Kitchen", "Garden"]);
    let seen = Shared::new(0);
    let counted = seen.clone();
    let watch = Tool::new("watch", "", move |call| Ok(counted.set(call.requests())));
    let rdm = Rdm::new(top.noul("watch", "Watch?"))
        .tool(descend)
        .tool(watch)
        .threads(1)
        .model(model.clone());
    let error = rdm.call(inputs! { "listing" => "x" }).unwrap_err();
    assert!(
        error.is_usage() && error.to_string().contains("'category' is a choice with no options"),
        "{error:?}"
    );
    assert_eq!((model.requests().len(), seen.get()), (1, 1));
}

#[test]
fn an_edit_that_names_an_output_the_signature_does_not_have_is_not_let_pass() {
    // A misspelt name changes nothing. DSPy raises where it is written; here the signature
    // says so when it is checked, and every call checks.
    let act =
        Signature::new("Operate a smart home.")
            .input("task")
            .choice("device", "Which device does `inputs.task` mean?", ["tv", "lights"]);
    let misspelt = act.clone().with_options("devcie", ["door"]);
    assert_eq!(misspelt.outputs(), act.outputs());
    let error = misspelt.check().unwrap_err();
    assert_eq!(
        error,
        Error::Usage("`with_options` was given 'devcie', which is no output of the signature: it has ['device'].".to_owned())
    );
    let model = agreeable();
    assert_eq!(
        Rdm::new(misspelt.clone())
            .model(model.clone())
            .call(inputs! { "task" => "t" })
            .unwrap_err(),
        error
    );
    // Given to one call, it is that call's error.
    let rdm = Rdm::new(act.clone()).model(model.clone());
    assert_eq!(
        rdm.call_with(Ask::new(inputs! { "task" => "t" }).signature(misspelt)).unwrap_err(),
        error
    );
    assert!(model.requests().is_empty());
    // An output it has is edited, and the signature is as good as it was.
    assert!(act.clone().with_options("device", ["door"]).check().is_ok());
    // Taking away a field it does not have leaves it as it is, as DSPy's `delete` does.
    assert_eq!(act.clone().without("devcie"), act);
}
