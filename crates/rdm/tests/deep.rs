//! Recursion as deep as the limits allow: a run that goes deep reaches a limit, which is
//! recorded, and never the end of a stack, which is the end of the process.

mod common;

use std::sync::{Arc, Mutex};

use common::client;
use rdm::{Rdm, Reason, Signature, Tool, inputs};

/// The stack a thread is given unless it asks for another: what a tool run beside others has, and most callers.
const SMALL: usize = 2 * 1024 * 1024;

/// Run on a thread with a small stack, as a caller that is not the main thread would.
fn on_a_small_stack<T: Send + 'static>(run: impl FnOnce() -> T + Send + 'static) -> T {
    common::on_a_stack(SMALL, run)
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn a_module_calls_itself_hundreds_of_levels_deep_on_a_small_stack() {
    // One sub-call a level: each is asked on the stack of the tool that made it, which is on the
    // stack of the call above. Three hundred of them once needed more stack than a thread has.
    let levels = 300;
    let count = Signature::new("Count down.")
        .input_typed("n", "int")
        .noul("again", "Is `inputs.n` more than zero?");
    let again = Tool::new("again", "", |call| {
        let n: i64 = call.arg("n")?;
        Ok(call.sub(inputs! { "n" => n - 1 })?.observed::<i64>("again").unwrap_or(0) + 1)
    });
    let model = client(|inputs, _, _| inputs["n"].as_i64() > Some(0));
    let rdm = Rdm::new(count)
        .tool(again)
        .max_depth(levels + 1)
        .max_predictor_calls(levels + 10)
        .model(model);
    let result = on_a_small_stack(move || rdm.call(inputs! { "n" => levels }).unwrap());
    assert_eq!(result.observations["again"], levels);
    assert_eq!((result.trajectory.len(), result.termination_reason), (levels + 1, Reason::Final));
    assert_eq!(result.trajectory.last().map(|entry| entry.depth), Some(levels));
}

#[test]
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
fn a_run_too_deep_for_the_one_stack_of_a_module_is_stopped_and_says_so() {
    // No thread can be started here, so there is no other stack to go on with. The run goes as
    // deep as the one stack has room for, and the tool that asked for one call more is told why
    // it was not made: its failure is an observation, and the module is still there.
    let levels = 5_000;
    let count = Signature::new("Count down.")
        .input_typed("n", "int")
        .noul("again", "Is `inputs.n` more than zero?");
    let again = Tool::new("again", "", |call| {
        let n: i64 = call.arg("n")?;
        Ok(call.sub(inputs! { "n" => n - 1 })?.observed::<i64>("again").unwrap_or(0) + 1)
    });
    let model = client(|inputs, _, _| inputs["n"].as_i64() > Some(0));
    let rdm = Rdm::new(count)
        .tool(again)
        .max_depth(levels + 1)
        .max_predictor_calls(levels + 10)
        .model(model);
    let result = rdm.call(inputs! { "n" => levels }).unwrap();
    let deepest = result.trajectory.last().map(|entry| entry.depth).unwrap();
    assert!((16..rdm::DEEPEST).contains(&deepest), "{deepest} deep");
    let met = &result.trajectory[result.trajectory.len() - 2];
    let seen = met.observations.as_ref().unwrap()["again"].as_str().unwrap();
    assert!(seen.starts_with("Execution error in again: The run is too deep to go on"), "{seen}");
    // And it runs again, from the top of the stack.
    assert_eq!(rdm.call(inputs! { "n" => 3 }).unwrap().observations["again"], 3);
}

/// Two modules whose tools call each other: each call is another module's, so neither's
/// `max_depth` is ever reached. Returns the first, and the tools hold both.
fn ping_pong(configure: impl Fn(Rdm) -> Rdm) -> Rdm {
    let modules: Arc<Mutex<Vec<Rdm>>> = Arc::default();
    for index in 0..2usize {
        let others = modules.clone();
        let pong = Tool::new("pong", "", move |call| {
            let n: i64 = call.arg("n")?;
            let other = others.lock().unwrap()[1 - index].clone();
            Ok(call.on(&other).sub(inputs! { "n" => n - 1 })?.termination_reason)
        });
        let signature = Signature::new(format!("Ping {index}."))
            .input_typed("n", "int")
            .noul("pong", "Is `inputs.n` more than zero?");
        let model = client(|inputs, _, _| inputs["n"].as_i64() > Some(0));
        modules.lock().unwrap().push(configure(Rdm::new(signature).tool(pong).model(model)));
    }
    modules.lock().unwrap()[0].clone()
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn two_modules_that_call_each_other_go_as_deep_as_the_runs_requests_allow() {
    // With the limits a module has unless told otherwise: a depth of 3 each, counted from where
    // each was called, and 256 requests for the run. The Python module stops such a run near
    // depth 70, at the interpreter's own limit, as the observation of the tool that met it.
    let result = on_a_small_stack(|| ping_pong(|module| module).call(inputs! { "n" => 250 }).unwrap());
    assert_eq!((result.trajectory.len(), result.termination_reason), (251, Reason::Final));
    assert_eq!(result.trajectory.last().map(|entry| entry.depth), Some(250));
    // Further than the run may ask, it is the run's limit on requests that stops it, and says so.
    let result = on_a_small_stack(|| ping_pong(|module| module).call(inputs! { "n" => 400 }).unwrap());
    assert_eq!(result.termination_reason, Reason::MaxPredictorCalls);
    let asked = result.trajectory.iter().filter(|entry| entry.asked()).count();
    assert_eq!((asked, result.trajectory.len()), (256, 257));
    assert_eq!(result.trajectory[256].skipped, Some(Reason::MaxPredictorCalls));
}

#[test]
#[cfg_attr(
    all(target_family = "wasm", not(target_feature = "atomics")),
    ignore = "no thread can be started here"
)]
fn no_run_goes_deeper_than_the_deepest_a_run_may() {
    // Let the modules ask as many requests as they like, and what is left to stop two that call
    // each other is the depth no run goes below: the call there is answered, the tool it
    // selected is not run, and the reason is `max_depth` all the way up.
    let deepest = 1024;
    let result = on_a_small_stack(|| {
        let first = ping_pong(|module| module.max_predictor_calls(100_000));
        first.call(inputs! { "n" => 5_000 }).unwrap()
    });
    assert_eq!(result.termination_reason, Reason::MaxDepth);
    assert_eq!(result.observations["pong"], "max_depth");
    assert_eq!(result.trajectory.len(), deepest + 1);
    let last = result.trajectory.last().unwrap();
    assert_eq!((last.depth, last.value("pong"), last.skipped), (deepest, true.into(), None));
    assert_eq!(last.observations.as_ref().map(|seen| seen.len()), Some(0));
}
