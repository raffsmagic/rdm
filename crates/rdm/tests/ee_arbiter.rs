//! The arbiter as a circuit: a speculative core's back end, checked as hardware is.
//!
//! - A branch that says what it has so far faster than a ruling is made must not keep every
//!   ruling from being acted on (the arbiter's metastability: a decision that never settles).
//! - An arbiter that uses everything and waits for everything comes to what no arbiter does (a
//!   reference model: the speculative core against the in-order one).
//! - Under many timings, every outside action is in the receipts, no stopped branch acts after
//!   it is stopped, and the request ends (safety and liveness, as assertions over random runs).
// Its cases hold branches at once, on threads: not in a WebAssembly module without them.
#![cfg(not(target_family = "wasm"))]

use rdm::arbitrate::{Arbiter, Ledger, Ruling, State};
use rdm::{Call, Rdm, Signature, Tool, inputs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use systemone::{Answer, Scripted};

fn model() -> Arc<Scripted> {
    Arc::new(Scripted::new(|r| {
        r.questions.keys().map(|n| (n.clone(), Answer::noul(0.9))).collect::<Vec<_>>()
    }))
}

/// Uses `quick` once it is back, and stops the rest: it takes `think` to rule, as a model would.
struct UseQuick {
    think: Duration,
    acted: AtomicUsize,
    set_aside: AtomicUsize,
}

impl Arbiter for UseQuick {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        std::thread::sleep(self.think);
        let back = ledger.branch("quick").is_some_and(|b| b.state == State::Returned);
        Ok(match back {
            true => Ruling {
                used: vec!["quick".into()],
                done: true,
                ..Ruling::default()
            },
            false => Ruling::all(ledger),
        })
    }

    fn acted(&self, _: &Ledger, _: &Ruling, acted: bool) {
        match acted {
            true => self.acted.fetch_add(1, Ordering::SeqCst),
            false => self.set_aside.fetch_add(1, Ordering::SeqCst),
        };
    }
}

/// A branch that says more every `every`, `times` times, unless it is stopped first.
fn chatty(every: Duration, times: usize, said: Arc<AtomicUsize>) -> Tool {
    Tool::new("chatty", "Say more and more.", move |call: &Call| {
        for n in 0..times {
            if call.stopped() {
                return Ok(format!("stopped after {n}"));
            }
            call.partial(n);
            said.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(every);
        }
        Ok("said it all".to_owned())
    })
}

/// A ruling that needs only `quick` is set aside by every new partial of a branch it stops. It
/// is not acted on until `chatty` falls quiet by itself: here, after all it had to say.
#[test]
fn a_chatty_branch_the_ruling_stops_does_not_keep_the_ruling_from_being_acted_on() {
    let arbiter = Arc::new(UseQuick {
        think: Duration::from_millis(20),
        acted: AtomicUsize::new(0),
        set_aside: AtomicUsize::new(0),
    });
    let said = Arc::new(AtomicUsize::new(0));
    let ask = Signature::new("Answer.")
        .input("q")
        .noul("quick", "Quick?")
        .noul("chatty", "Chatty?");
    let rdm = Rdm::new(ask)
        .model(model())
        .arbiter(ArcArbiter(arbiter.clone()))
        .tool(Tool::new("quick", "Answer at once.", |_| Ok("answer")))
        .tool(chatty(Duration::from_millis(2), 400, said.clone()));
    let begun = Instant::now();
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    let took = begun.elapsed();
    let set_aside = arbiter.set_aside.load(Ordering::SeqCst);
    let arbitration = result.arbitration.unwrap();
    eprintln!(
        "took {took:?}, rounds {}, acted {}, set aside {set_aside}, chatty said {} of 400",
        arbitration.rounds,
        arbiter.acted.load(Ordering::SeqCst),
        said.load(Ordering::SeqCst)
    );
    assert_eq!(arbitration.used, ["quick"]);
    // One ruling is enough to see `quick` and stop `chatty`; one more may be set aside by a
    // partial in flight. More than that is a ruling that does not settle.
    assert!(
        set_aside <= 2,
        "{set_aside} rulings set aside by partials of a branch each of them stopped; took {took:?}"
    );
}

/// An arbiter given as an `Arc`, so a test can read what it counted.
struct ArcArbiter<A>(Arc<A>);

impl<A: Arbiter> Arbiter for ArcArbiter<A> {
    fn arbitrate(&self, ledger: &Ledger, call: &Call) -> rdm::Result<Ruling> {
        self.0.arbitrate(ledger, call)
    }
    fn first(&self) -> bool {
        self.0.first()
    }
    fn beside(&self) -> Vec<String> {
        self.0.beside()
    }
    fn acted(&self, ledger: &Ledger, ruling: &Ruling, acted: bool) {
        self.0.acted(ledger, ruling, acted)
    }
}

/// The reference model: every result used, every branch waited for.
struct All;

impl Arbiter for All {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        Ok(Ruling::all(ledger))
    }
}

/// A small generator, so a failing timing can be run again from its seed.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn routes(delays: [u64; 3], partials: bool) -> Rdm {
    let ask = Signature::new("Answer.").input("q").noul("a", "A?").noul("b", "B?").noul("c", "C?");
    let mut rdm = Rdm::new(ask).model(model());
    for (name, delay) in ["a", "b", "c"].into_iter().zip(delays) {
        rdm = rdm.tool(Tool::new(name, "A route.", move |call: &Call| {
            if partials {
                call.partial(format!("{name} so far"));
            }
            std::thread::sleep(Duration::from_micros(delay));
            Ok(format!("{name} found it"))
        }));
    }
    rdm
}

/// Under many timings, arbitration that uses all and waits for all comes to the same
/// observations as no arbitration.
#[test]
fn an_arbiter_that_uses_all_comes_to_what_none_does() {
    let mut rng = Lcg(0x5eed);
    for round in 0..100 {
        let delays = [rng.below(3000), rng.below(3000), rng.below(3000)];
        let partials = rng.below(2) == 1;
        let plain = routes(delays, partials).call(inputs! {"q" => "q"}).unwrap();
        let arbitrated = routes(delays, partials).arbiter(All).call(inputs! {"q" => "q"}).unwrap();
        assert_eq!(
            plain.observations, arbitrated.observations,
            "round {round}, delays {delays:?}, partials {partials}"
        );
        let arbitration = arbitrated.arbitration.unwrap();
        assert!(arbitration.stopped.is_empty(), "round {round}: {arbitration:?}");
        assert_eq!(arbitration.fallbacks, 0, "round {round}");
    }
}

/// Rules at random: uses, waits, lets finish, starts and finishes as its seed says.
struct Random(Mutex<Lcg>);

impl Arbiter for Random {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        let mut rng = self.0.lock().unwrap();
        let mut ruling = Ruling::default();
        for branch in &ledger.branches {
            let name = branch.output.clone();
            match (branch.state, rng.below(4)) {
                (State::Returned, 0..=2) => ruling.used.push(name),
                (State::Running, 0 | 1) => ruling.wait.push(name),
                (State::Running, 2) => ruling.shadow.push(name),
                (State::Idle, 0) => ruling.start.push(name),
                _ => {}
            }
        }
        ruling.done = rng.below(2) == 1;
        Ok(ruling)
    }

    fn first(&self) -> bool {
        true
    }
}

/// Under many timings and random rulings: everything done outside is in the receipts, a branch
/// does nothing outside once it has seen itself stopped, and the request ends.
#[test]
fn under_random_rulings_and_timings_what_is_done_outside_is_said_and_the_request_ends() {
    let mut rng = Lcg(0xa7b17e7);
    for round in 0..200 {
        let seed = rng.next();
        let delays: Vec<u64> = (0..4).map(|_| rng.below(2000)).collect();
        let done_outside = Arc::new(AtomicUsize::new(0));
        let acted_after_stop = Arc::new(AtomicBool::new(false));
        let ask = Signature::new("Answer and do.")
            .input("q")
            .noul("r0", "R0?")
            .noul("r1", "R1?")
            .noul("r2", "R2?")
            .noul("r3", "R3?");
        let mut rdm = Rdm::new(ask).model(model()).arbiter(Random(Mutex::new(Lcg(seed))));
        for (at, delay) in delays.iter().copied().enumerate() {
            let (done_outside, acted_after_stop) = (done_outside.clone(), acted_after_stop.clone());
            rdm = rdm.tool(Tool::new(format!("r{at}"), "A route that may do something.", move |call: &Call| {
                call.partial(format!("r{at} so far"));
                std::thread::sleep(Duration::from_micros(delay));
                // Odd routes do something outside; a stopped one is refused and ends.
                if at % 2 == 1 {
                    let seen_stopped = call.stopped();
                    call.acting(|| {
                        if seen_stopped {
                            acted_after_stop.store(true, Ordering::SeqCst);
                        }
                        done_outside.fetch_add(1, Ordering::SeqCst);
                        format!("r{at} did it")
                    })?;
                }
                std::thread::sleep(Duration::from_micros(delay));
                Ok(format!("r{at} found it"))
            }));
        }
        let begun = Instant::now();
        let result = rdm.call(inputs! {"q" => "q"}).unwrap();
        assert!(begun.elapsed() < Duration::from_secs(5), "round {round}, seed {seed}: did not end");
        // Let branches that were let finish come to an end before counting what they did.
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !acted_after_stop.load(Ordering::SeqCst),
            "round {round}, seed {seed}: acted when stopped"
        );
        assert_eq!(
            result.receipts.len(),
            done_outside.load(Ordering::SeqCst),
            "round {round}, seed {seed}: what was done outside is not all in the receipts: {:?}",
            result.arbitration
        );
        let arbitration = result.arbitration.unwrap();
        for used in &arbitration.used {
            assert!(
                !arbitration.stopped.contains(used) || arbitration.partial.contains(used),
                "round {round}, seed {seed}: {arbitration:?}"
            );
        }
    }
}

/// One run of `chatty` alone, its partials used, under the model's arbiter and a budget of
/// `budget` requests: how many rulings were put to the model, and what arbitration came to.
fn used_chatty(budget: Option<usize>) -> (usize, Duration, rdm::arbitrate::Arbitration, usize) {
    let rulings = Arc::new(AtomicUsize::new(0));
    let asked = rulings.clone();
    let model = Arc::new(Scripted::new(move |request| {
        if !request.questions.contains_key("next") {
            return vec![("chatty", Answer::noul(0.9))];
        }
        asked.fetch_add(1, Ordering::SeqCst);
        // A ruling takes as long as a hosted decision might.
        std::thread::sleep(Duration::from_millis(20));
        let options = request.questions["next"].labels();
        let mut answers = vec![("next", Answer::chose("finish", options.iter().map(String::as_str)))];
        for name in ["use_chatty", "wait_chatty", "conflict"] {
            if request.questions.contains_key(name) {
                answers.push((name, Answer::noul(if name == "use_chatty" { 0.95 } else { 0.05 })));
            }
        }
        answers
    }));
    let said = Arc::new(AtomicUsize::new(0));
    let ask = Signature::new("Answer.").input("q").noul("chatty", "Chatty?");
    let mut rdm = Rdm::new(ask)
        .model(model)
        .arbiter(rdm::Arbitrate::new())
        .tool(chatty(Duration::from_millis(2), 400, said.clone()));
    if let Some(budget) = budget {
        rdm = rdm.max_predictor_calls(budget);
    }
    let begun = Instant::now();
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    let took = begun.elapsed();
    (
        rulings.load(Ordering::SeqCst),
        took,
        result.arbitration.unwrap(),
        said.load(Ordering::SeqCst),
    )
}

/// A ruling that uses a branch is set aside by each new partial of it, and each is a request to
/// the model. Two in a row that come to the same are acted on: the request does not wait for the
/// branch to fall quiet, and pays for two rulings, not one for each partial.
#[test]
fn rulings_on_a_chatty_branch_they_use_settle_when_two_agree() {
    let (rulings, took, arbitration, said) = used_chatty(None);
    eprintln!(
        "{rulings} rulings asked, rounds {}, fallbacks {}, used {:?}, answered {}, chatty said {said} of 400, took {took:?}",
        arbitration.rounds, arbitration.fallbacks, arbitration.used, arbitration.answered
    );
    assert!(rulings <= 3, "{rulings} rulings put to the model");
    assert!(arbitration.answered && arbitration.used == ["chatty"]);
    assert!(said < 400, "the request waited for the branch to fall quiet");
}

const DEADLINE: Duration = Duration::from_secs(5);

/// Tells a branch in step to say its next version, and waits until it has.
struct Talk {
    tell: Mutex<mpsc::Sender<()>>,
    told: Mutex<mpsc::Receiver<()>>,
}

impl Talk {
    fn more(&self) {
        self.tell.lock().unwrap().send(()).unwrap();
        self.told.lock().unwrap().recv_timeout(DEADLINE).unwrap();
    }
}

/// A branch `chatty` that says the first of `versions` at once, and each next one only when
/// told ([`Talk::more`]), so every interleaving of its partials with rulings is fixed. Once told
/// with none left it says nothing more; it ends when stopped, or after `quiet` untold.
fn in_step(versions: &[&str], quiet: Duration) -> (Tool, Talk) {
    let (tell, hear) = mpsc::channel::<()>();
    let (said, told) = mpsc::channel::<()>();
    let (hear, said) = (Mutex::new(hear), Mutex::new(said));
    let versions: Vec<String> = versions.iter().map(|v| v.to_string()).collect();
    let tool = Tool::new("chatty", "Say more when told.", move |call: &Call| {
        let (hear, said) = (hear.lock().unwrap(), said.lock().unwrap());
        call.partial(versions[0].clone());
        let mut next = 1;
        let mut untold = Instant::now();
        loop {
            match hear.recv_timeout(Duration::from_millis(2)) {
                Ok(()) => {
                    if let Some(version) = versions.get(next) {
                        call.partial(version.clone());
                        next += 1;
                    }
                    untold = Instant::now();
                    said.send(()).unwrap();
                }
                Err(_) if call.stopped() => return Ok("stopped".to_owned()),
                Err(_) if untold.elapsed() > quiet => return Ok("fell quiet".to_owned()),
                Err(_) => {}
            }
        }
    });
    let talk = Talk {
        tell: Mutex::new(tell),
        told: Mutex::new(told),
    };
    (tool, talk)
}

/// Rules on `chatty` by what it has so far, as `rule` says, having first told it to say more.
struct Stepping {
    talk: Talk,
    rule: fn(&str) -> Ruling,
    rulings: AtomicUsize,
}

impl Arbiter for Stepping {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        self.rulings.fetch_add(1, Ordering::SeqCst);
        let so_far = ledger.branch("chatty").and_then(|b| b.partial.clone()).unwrap_or_default();
        self.talk.more();
        Ok((self.rule)(so_far.as_str().unwrap_or_default()))
    }
}

fn finish() -> Ruling {
    Ruling {
        used: vec!["chatty".into()],
        done: true,
        ..Ruling::default()
    }
}

fn hold() -> Ruling {
    Ruling {
        used: vec!["chatty".into()],
        wait: vec!["chatty".into()],
        ..Ruling::default()
    }
}

fn stepping(versions: &[&str], rule: fn(&str) -> Ruling) -> (Rdm, Arc<Stepping>) {
    let (tool, talk) = in_step(versions, DEADLINE);
    let arbiter = Arc::new(Stepping {
        talk,
        rule,
        rulings: AtomicUsize::new(0),
    });
    let ask = Signature::new("Answer.").input("q").noul("chatty", "Chatty?");
    let rdm = Rdm::new(ask).model(model()).arbiter(ArcArbiter(arbiter.clone())).tool(tool);
    (rdm, arbiter)
}

/// Each ruling sees one version and the next is said while it is made. The first is set aside;
/// the second comes to the same, and is acted on with the version it saw.
#[test]
fn a_chatty_branch_that_is_used_settles_in_two_rulings_that_agree() {
    let versions: Vec<String> = (1..=100).map(|n| format!("v{n}")).collect();
    let versions: Vec<&str> = versions.iter().map(String::as_str).collect();
    let (rdm, arbiter) = stepping(&versions, |_| finish());
    let begun = Instant::now();
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbiter.rulings.load(Ordering::SeqCst), 2);
    assert!(begun.elapsed() < DEADLINE / 2, "the request waited for the branch to fall quiet");
    assert!(arbitration.answered);
    assert_eq!(result.observations["chatty"], "v2", "not what the ruling acted on saw");
    assert_eq!(arbitration.versions, [("chatty".to_owned(), 2)]);
}

/// A correction that changes the ruling still sets it aside: the second ruling, on the
/// correction, holds where the first finished, so neither stands, and the third, on what is
/// said last, is acted on.
#[test]
fn a_correction_that_changes_the_ruling_still_sets_it_aside() {
    let (rdm, arbiter) = stepping(&["draft", "corrected", "settled"], |so_far| match so_far {
        "corrected" => hold(),
        _ => finish(),
    });
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbiter.rulings.load(Ordering::SeqCst), 3);
    assert_eq!(result.observations["chatty"], "settled");
    assert_eq!(arbitration.versions, [("chatty".to_owned(), 3)]);
    assert!(arbitration.answered);
}

/// The model's arbiter over a branch in step, under a budget of `budget` requests: its rulings
/// as `next` says for each, by its place, from 0.
fn budgeted(budget: usize, quiet: Duration, next: fn(usize) -> &'static str) -> (rdm::Prediction, usize, Duration) {
    let (tool, talk) = in_step(&["v1", "v2", "v3", "v4"], quiet);
    let rulings = Arc::new(AtomicUsize::new(0));
    let asked = rulings.clone();
    let model = Arc::new(Scripted::new(move |request| {
        if !request.questions.contains_key("next") {
            return vec![("chatty", Answer::noul(0.9))];
        }
        let at = asked.fetch_add(1, Ordering::SeqCst);
        talk.more();
        let options = request.questions["next"].labels();
        let mut answers = vec![("next", Answer::chose(next(at), options.iter().map(String::as_str)))];
        for (name, p) in [("use_chatty", 0.95), ("wait_chatty", 0.9), ("conflict", 0.05)] {
            if request.questions.contains_key(name) {
                answers.push((name, Answer::noul(p)));
            }
        }
        answers
    }));
    let ask = Signature::new("Answer.").input("q").noul("chatty", "Chatty?");
    let rdm = Rdm::new(ask)
        .model(model)
        .arbiter(rdm::Arbitrate::new())
        .tool(tool)
        .max_predictor_calls(budget);
    let begun = Instant::now();
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    (result, rulings.load(Ordering::SeqCst), begun.elapsed())
}

/// The budget runs out with one ruling made, set aside only by what the branch it used said
/// since. Nothing disagrees with it, so it is acted on: the request is answered, not left to
/// wait for the branch to fall quiet.
#[test]
fn a_budget_spent_after_rulings_that_agree_answers_with_the_last_of_them() {
    let (result, rulings, took) = budgeted(2, DEADLINE, |_| "finish");
    let arbitration = result.arbitration.unwrap();
    assert_eq!(rulings, 1);
    assert!(took < DEADLINE / 2, "the request waited for the branch to fall quiet");
    assert!(arbitration.answered, "{arbitration:?}");
    assert_eq!(arbitration.fallbacks, 0);
    assert_eq!(result.observations["chatty"], "v1");
}

/// The budget runs out after two rulings that did not agree: one finished, the next waited.
/// Nothing was decided, so the request goes on as no arbiter would, and waits for the branch.
#[test]
fn a_budget_spent_after_rulings_that_disagree_falls_back() {
    let quiet = Duration::from_millis(300);
    let (result, rulings, took) = budgeted(3, quiet, |at| if at == 0 { "finish" } else { "wait" });
    let arbitration = result.arbitration.unwrap();
    assert_eq!(rulings, 2);
    assert!(took >= quiet, "{took:?}: did not wait for the branch");
    assert!(arbitration.fallbacks > 0);
    assert!(!arbitration.answered);
    assert_eq!(result.observations["chatty"], "fell quiet");
}

/// Stops `early` once it has said what it has, waiting for `chatty`; then uses what `early` had
/// when it was stopped. `chatty` says more while every ruling is made.
struct UseStopped {
    talk: Talk,
    set_aside: AtomicUsize,
}

impl Arbiter for UseStopped {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        let early = ledger.branch("early").unwrap();
        Ok(match (early.state, &early.partial) {
            (State::Running, None) => Ruling::all(ledger),
            (State::Running, Some(_)) => {
                self.talk.more();
                Ruling {
                    wait: vec!["chatty".into()],
                    ..Ruling::default()
                }
            }
            _ => {
                self.talk.more();
                Ruling {
                    used: vec!["early".into()],
                    done: true,
                    ..Ruling::default()
                }
            }
        })
    }

    fn acted(&self, _: &Ledger, _: &Ruling, acted: bool) {
        if !acted {
            self.set_aside.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// What a stopped branch had said is shown to later rulings at the version it was stopped at,
/// and it can say no more: a ruling that uses it is never set aside for it.
#[test]
fn a_stopped_branch_used_for_what_it_had_never_sets_a_ruling_aside() {
    let versions: Vec<String> = (1..=100).map(|n| format!("c{n}")).collect();
    let versions: Vec<&str> = versions.iter().map(String::as_str).collect();
    let (chatty, talk) = in_step(&versions, DEADLINE);
    let early = Tool::new("early", "Say what it has, then work on.", |call: &Call| {
        call.partial("what early had");
        while !call.stopped() {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok("early, after it was stopped")
    });
    let arbiter = Arc::new(UseStopped {
        talk,
        set_aside: AtomicUsize::new(0),
    });
    let ask = Signature::new("Answer.")
        .input("q")
        .noul("early", "Early?")
        .noul("chatty", "Chatty?");
    let rdm = Rdm::new(ask)
        .model(model())
        .arbiter(ArcArbiter(arbiter.clone()))
        .tool(early)
        .tool(chatty);
    let begun = Instant::now();
    let result = rdm.call(inputs! {"q" => "q"}).unwrap();
    let arbitration = result.arbitration.unwrap();
    assert!(begun.elapsed() < DEADLINE / 2, "the request waited for chatty to fall quiet");
    assert_eq!(arbiter.set_aside.load(Ordering::SeqCst), 0, "{arbitration:?}");
    assert_eq!(arbitration.used, ["early"]);
    assert!(arbitration.stopped.contains(&"early".to_owned()), "{arbitration:?}");
    assert_eq!(result.observations["early"], "what early had");
    assert!(arbitration.answered);
}

/// Rules `finish` in round 0, telling `chatty` to say more while it does, and cannot rule after.
/// Says what it is told of each ruling, in order: `acted` and `revived` apart.
struct Revival {
    talk: Talk,
    told: Arc<Mutex<Vec<String>>>,
}

impl Arbiter for Revival {
    fn arbitrate(&self, ledger: &Ledger, _: &Call) -> rdm::Result<Ruling> {
        match ledger.round {
            0 => {
                self.talk.more();
                Ok(finish())
            }
            _ => Err(rdm::Error::Tool("no ruling".into())),
        }
    }

    fn acted(&self, ledger: &Ledger, _: &Ruling, acted: bool) {
        self.told.lock().unwrap().push(format!("acted {} {acted}", ledger.round));
    }

    fn revived(&self, ledger: &Ledger, ruling: &Ruling, from: usize) {
        let version = ledger.branch("chatty").map(|b| b.version);
        self.told.lock().unwrap().push(format!(
            "revived {} from {from}: done {}, chatty at {version:?}",
            ledger.round, ruling.done
        ));
    }
}

/// Round 0 is set aside by what chatty said since, and round 1 cannot be made: round 0's ruling
/// stands as round 1's. Each round is told once, in order: round 0 that it was not acted on
/// (and it stays so), round 1 that round 0's ruling was acted on as its own, on what round 0
/// saw. The record says round 1's ruling is round 0's, and what it saw then.
#[test]
fn a_ruling_that_stands_in_for_one_not_made_is_told_as_that_round_from_its_own() {
    let (tool, talk) = in_step(&["v1", "v2"], DEADLINE);
    let told = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded = events.clone();
    let result = Rdm::new(Signature::new("Answer.").input("q").noul("chatty", "Chatty?"))
        .model(model())
        .arbiter(Revival { talk, told: told.clone() })
        .tool(tool)
        .record(move |event: &rdm::record::Event| {
            match event {
                rdm::record::Event::Arbitrated {
                    round, ruling, saw, error, ..
                } => recorded.lock().unwrap().push(format!(
                    "arbitrated {round}: done {}, saw {saw:?}, failed {}",
                    ruling.done,
                    error.is_some()
                )),
                rdm::record::Event::Revived { round, from, .. } => recorded.lock().unwrap().push(format!("revived {round} from {from}")),
                _ => {}
            }
            Ok(())
        })
        .call(inputs! {"q" => "q"})
        .unwrap();
    assert_eq!(
        *told.lock().unwrap(),
        ["acted 0 false", "revived 1 from 0: done true, chatty at Some(1)"]
    );
    assert_eq!(
        *events.lock().unwrap(),
        [
            r#"arbitrated 0: done true, saw [("chatty", 1)], failed false"#,
            r#"arbitrated 1: done true, saw [("chatty", 1)], failed true"#,
            "revived 1 from 0",
        ]
    );
    let arbitration = result.arbitration.unwrap();
    assert_eq!(arbitration.reused, Some((0, 1)));
    assert_eq!(arbitration.fallbacks, 0);
    assert!(arbitration.answered);
    assert_eq!(arbitration.versions, [("chatty".to_owned(), 1)]);
    assert_eq!(result.observations["chatty"], "v1");
}

// A fixture kept byte for byte as it was written.
include!("ee/revival.rs");
