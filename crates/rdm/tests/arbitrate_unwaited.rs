//! A branch let finish unwaited keeps going when its run ends well, and is stopped when the run, or anything enclosing it, fails or panics.
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Ledger, Ruling, State};
use rdm::{Arbiter, Call, Rdm, Signature, Tool, inputs};
use std::sync::{Mutex, mpsc::channel};
use std::time::Duration;
use systemone::{Answer, Scripted};
const LIMIT: Duration = Duration::from_secs(5);

struct ShadowThenWait;
impl Arbiter for ShadowThenWait {
    fn arbitrate(&self, l: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        assert_eq!(l.branch("quick").unwrap().state, State::Returned);
        Ok(Ruling {
            used: vec!["quick".into()],
            wait: vec!["fail".into()],
            shadow: vec!["shadow".into()],
            ..Default::default()
        })
    }
}
struct FinishWithShadow;
impl Arbiter for FinishWithShadow {
    fn arbitrate(&self, _: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling {
            used: vec!["quick".into()],
            shadow: vec!["shadow".into()],
            done: true,
            ..Default::default()
        })
    }
}
fn dm() -> Scripted {
    Scripted::new(|r| r.questions.keys().map(|k| (k.clone(), Answer::noul(0.9))).collect::<Vec<_>>())
}

#[test]
fn fatal_error_after_an_earlier_shadow_ruling_stops_the_shadow_branch() {
    let (ready, heard) = channel();
    let ready = Mutex::new(ready);
    let heard = Mutex::new(heard);
    let (shadowed, was_shadowed) = channel();
    let shadowed = Mutex::new(shadowed);
    let was_shadowed = Mutex::new(was_shadowed);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let root = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Quick?")
            .noul("shadow", "Verify?")
            .noul("fail", "Investigate?"),
    )
    .model(dm())
    .arbiter(ShadowThenWait)
    .tool(Tool::new("quick", "Quick", move |_| {
        heard.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("shadow", "Model-only verifier", move |c: &Call| {
        ready.lock().unwrap().send(()).unwrap();
        let end = std::time::Instant::now() + LIMIT;
        while !c.shadowed() {
            assert!(std::time::Instant::now() < end, "shadow ruling never applied");
            std::thread::yield_now();
        }
        shadowed.lock().unwrap().send(()).unwrap();
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        observed
            .lock()
            .unwrap()
            .send((c.shadowed(), c.stopped(), c.go_on().is_err()))
            .unwrap();
        Ok("late evidence")
    }))
    .tool(Tool::new(
        "fail",
        "Fatal failure after first ruling",
        move |_| -> rdm::Result<&str> {
            was_shadowed.lock().unwrap().recv_timeout(LIMIT).unwrap();
            Err(rdm::Error::Usage("fatal after shadow".into()))
        },
    ));
    let result = root.call(inputs! {"q"=>"q"});
    release.send(()).unwrap();
    let state = came.recv_timeout(LIMIT).unwrap();
    assert!(
        matches!(result, Err(rdm::Error::Usage(_))),
        "expected original fatal error: {result:?}"
    );
    assert_eq!(state, (true, true, true), "shadow permission survived a fatal parent exit");
}

#[test]
fn successful_completion_does_not_cancel_a_permitted_shadow() {
    let (ready, heard) = channel();
    let ready = Mutex::new(ready);
    let heard = Mutex::new(heard);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let root = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Quick?")
            .noul("shadow", "Verify?"),
    )
    .model(dm())
    .arbiter(FinishWithShadow)
    .tool(Tool::new("quick", "Quick", move |_| {
        heard.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("shadow", "Model-only verifier", move |c: &Call| {
        ready.lock().unwrap().send(()).unwrap();
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        observed
            .lock()
            .unwrap()
            .send((c.shadowed(), c.stopped(), c.go_on().is_ok()))
            .unwrap();
        Ok("late evidence")
    }));
    let result = root.call(inputs! {"q"=>"q"});
    release.send(()).unwrap();
    let state = came.recv_timeout(LIMIT).unwrap();
    assert!(result.is_ok(), "success case failed: {result:?}");
    assert_eq!(state, (true, false, true), "successful detached shadow was cancelled");
}

#[test]
fn a_failure_recording_the_final_event_revokes_shadow_continuation() {
    let (ready, heard) = channel();
    let ready = Mutex::new(ready);
    let heard = Mutex::new(heard);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let root = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Quick?")
            .noul("shadow", "Verify?"),
    )
    .model(dm())
    .arbiter(FinishWithShadow)
    .tool(Tool::new("quick", "Quick", move |_| {
        heard.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("shadow", "Model-only verifier", move |c: &Call| {
        ready.lock().unwrap().send(()).unwrap();
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        observed
            .lock()
            .unwrap()
            .send((c.shadowed(), c.stopped(), c.go_on().is_err()))
            .unwrap();
        Ok("late evidence")
    }))
    .record(|event: &rdm::record::Event| {
        if matches!(event, rdm::record::Event::Ended { .. }) {
            Err(rdm::Error::Tool("final record failed".into()))
        } else {
            Ok(())
        }
    });
    let result = root.call(inputs! {"q"=>"q"});
    release.send(()).unwrap();
    let state = came.recv_timeout(LIMIT).unwrap();
    assert!(result.is_err(), "a refused final event must fail the call");
    assert_eq!(
        state,
        (true, true, true),
        "shadow permission survived a fatal error outside the local settling scope"
    );
}

#[test]
fn a_later_parent_failure_revokes_a_completed_childs_shadow() {
    let (ready, heard) = channel();
    let ready = Mutex::new(ready);
    let heard = Mutex::new(heard);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let child = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Quick?")
            .noul("shadow", "Verify?"),
    )
    .model(dm())
    .arbiter(FinishWithShadow)
    .tool(Tool::new("quick", "Quick", move |_| {
        heard.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("shadow", "Model-only verifier", move |c: &Call| {
        ready.lock().unwrap().send(()).unwrap();
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        observed
            .lock()
            .unwrap()
            .send((c.shadowed(), c.stopped(), c.go_on().is_err()))
            .unwrap();
        Ok("late evidence")
    }));
    let root = Rdm::new(Signature::new("Run task").input("q").noul("work", "Work?"))
        .model(dm())
        .tool(Tool::new("work", "Use child then fail", move |c: &Call| -> rdm::Result<&str> {
            c.on(&child).sub(inputs! {"q"=>"q"})?;
            Err(rdm::Error::Usage("parent failed after child finished".into()))
        }));
    let result = root.call(inputs! {"q"=>"q"});
    release.send(()).unwrap();
    let state = came.recv_timeout(LIMIT).unwrap();
    assert!(result.is_err(), "parent error must propagate");
    assert_eq!(
        state,
        (true, true, true),
        "detached child shadow survived the enclosing task failure"
    );
}

#[test]
fn a_panic_recording_the_final_event_revokes_shadow_continuation() {
    let (ready, heard) = channel();
    let ready = Mutex::new(ready);
    let heard = Mutex::new(heard);
    let (release, wait) = channel();
    let wait = Mutex::new(wait);
    let (observed, came) = channel();
    let observed = Mutex::new(observed);
    let root = Rdm::new(
        Signature::new("Answer")
            .input("q")
            .noul("quick", "Quick?")
            .noul("shadow", "Verify?"),
    )
    .model(dm())
    .arbiter(FinishWithShadow)
    .tool(Tool::new("quick", "Quick", move |_| {
        heard.lock().unwrap().recv_timeout(LIMIT).unwrap();
        Ok("answer")
    }))
    .tool(Tool::new("shadow", "Model-only verifier", move |c: &Call| {
        ready.lock().unwrap().send(()).unwrap();
        wait.lock().unwrap().recv_timeout(LIMIT).unwrap();
        observed
            .lock()
            .unwrap()
            .send((c.shadowed(), c.stopped(), c.go_on().is_err()))
            .unwrap();
        Ok("late evidence")
    }))
    .record(|event: &rdm::record::Event| {
        if matches!(event, rdm::record::Event::Ended { .. }) {
            panic!("final record panicked")
        } else {
            Ok(())
        }
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| root.call(inputs! {"q"=>"q"})));
    release.send(()).unwrap();
    let state = came.recv_timeout(LIMIT).unwrap();
    assert!(result.is_err(), "a refused final event must fail the call");
    assert_eq!(
        state,
        (true, true, true),
        "shadow permission survived a fatal error outside the local settling scope"
    );
}
