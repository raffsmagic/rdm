//! Calls run at once, and calls run one inside another.
//!
//! The tools one request selects run at once, and so do the sub-calls of one batch. A tool that
//! asks the module again is recursion, which is as deep as the limits allow and not as deep as
//! the caller's stack happens to be: past a few calls on one stack, the next has a stack of its own.
//!
//! That is where a thread can be started. A WebAssembly module has the one stack it was called
//! on, unless it was built with threads: there the calls run one after another, in the order
//! they would have been started, and a run is as deep as that stack lets it be.

use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};

use crate::error::{Error, Result};

/// The stack of a thread the module starts: for the tools one request selects, for the
/// sub-calls of one batch, and for a run that has gone deep.
const STACK: usize = 8 * 1024 * 1024;

/// How many calls, each inside the one before, run on one stack before the next is given a
/// stack of its own. A sub-call is two: the call, and the tool that made it. Recursion is as
/// deep as the limits allow and not as deep as the caller's stack happens to be.
const NESTED: usize = 16;

/// Whether a thread can be started here. In WebAssembly one can only where the module was built
/// with threads (`wasm32-wasip1-threads`); anywhere else there, starting one is an error.
pub(crate) const THREADS: bool = cfg!(not(all(target_family = "wasm", not(target_feature = "atomics"))));

/// How much of its one stack a WebAssembly module must have left for a call to run inside
/// another: room for the tool the call runs and the model it asks.
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
const RESERVE: usize = 128 * 1024;

/// How much is left of the one stack a WebAssembly module has: from where this is called down
/// to the stack's lowest address, which the linker names. A megabyte in all unless the module
/// was linked with another size (`-C link-arg=-zstack-size=...`).
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
fn stack_left() -> usize {
    unsafe extern "C" {
        static __stack_low: u8;
    }
    let here = 0u8;
    (std::hint::black_box(&raw const here) as usize).saturating_sub(&raw const __stack_low as usize)
}

thread_local! {
    /// How many calls of the module are on this thread's stack, each inside the one before.
    static NESTING: Cell<usize> = const { Cell::new(0) };
}

/// Run calls at once, at most `width` of them at a time, and return their results in order.
///
/// One call runs where it is called. Several run on threads of their own, each with a stack
/// of a set size, or one after another where no thread can be started. A call that panics does
/// not take the others with it: they finish, and the panic goes on to the caller as it was raised.
pub(crate) fn in_parallel<T: Send, F: FnOnce() -> Result<T> + Send>(width: usize, mut calls: Vec<F>) -> Vec<Result<T>> {
    if calls.len() <= 1 {
        return calls.pop().map(nested).into_iter().collect();
    }
    let count = calls.len();
    let waiting: Vec<Mutex<Option<F>>> = calls.into_iter().map(|call| Mutex::new(Some(call))).collect();
    let results: Vec<Mutex<Option<Result<T>>>> = (0..count).map(|_| Mutex::new(None)).collect();
    let panicked: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let position = next.fetch_add(1, Ordering::Relaxed);
            let Some(call) = waiting
                .get(position)
                .and_then(|call| call.lock().unwrap_or_else(PoisonError::into_inner).take())
            else {
                break;
            };
            match catch_unwind(AssertUnwindSafe(|| nested(call))) {
                Ok(result) => *results[position].lock().unwrap_or_else(PoisonError::into_inner) = Some(result),
                Err(panic) => {
                    panicked.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(panic);
                }
            }
        }
    };
    std::thread::scope(|scope| {
        let threads = (0..count.min(width.max(1))).map(|_| std::thread::Builder::new().stack_size(STACK).spawn_scoped(scope, work));
        // A thread that could not be started leaves its share to the others, or to this one.
        if threads.filter(|thread| thread.is_ok()).count() == 0 {
            work();
        }
    });
    if let Some(panic) = panicked.into_inner().unwrap_or_else(PoisonError::into_inner) {
        resume_unwind(panic);
    }
    // A call that panicked has gone on to the caller, so every result is here.
    results
        .into_iter()
        .filter_map(|result| result.into_inner().unwrap_or_else(PoisonError::into_inner))
        .collect()
}

/// Run one call inside another: on this stack while it has few enough on it, and then on a
/// stack of its own. A tool that asks the module again is recursion, and this is what keeps
/// it off the caller's stack: a run that goes deep reaches a limit, and not the end of a stack.
/// Where no thread can be started there is no other stack, and every call is on this one: a
/// run that goes deep reaches the end of what that stack has room for, which is an error for
/// the tool that asked, and not the end of the stack, which is the end of the module.
fn nested<T: Send>(call: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    /// Puts the count back when the call is over, however it ended.
    struct Back(usize);
    impl Drop for Back {
        fn drop(&mut self) {
            NESTING.set(self.0);
        }
    }
    #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
    if stack_left() < RESERVE {
        return Err(Error::Tool(format!(
            "The run is too deep to go on: a WebAssembly module has one stack, and {} KiB of it are left.",
            stack_left() / 1024
        )));
    }
    let here = NESTING.get();
    if here < NESTED || !THREADS {
        let _back = Back(here);
        NESTING.set(here + 1);
        return call();
    }
    std::thread::scope(|scope| {
        let thread = std::thread::Builder::new()
            .stack_size(STACK)
            .spawn_scoped(scope, call)
            .map_err(|error| {
                Error::Tool(format!(
                    "The run is too deep to go on: no thread could be started for it ({error})."
                ))
            })?;
        thread.join().unwrap_or_else(|panic| resume_unwind(panic))
    })
}

/// A call not yet run.
type Pending<T> = Box<dyn FnOnce() -> Result<T> + Send>;

/// One call that may be started on a thread of its own and handed back if it could not be.
type Started<T> = Arc<Mutex<Option<Pending<T>>>>;

/// What [`Arrivals::next`] hands back: a call that came back, or a word from one still running.
pub(crate) enum Came<T> {
    /// A call came back, with its position, or with its panic.
    Back(usize, std::thread::Result<Result<T>>),
    /// A call still running had something to say ([`Arrivals::waker`]).
    Woken,
}

/// Calls started without being waited for, whose results are taken as they come.
///
/// What [`in_parallel`] does with every call at once and then waits for all of them, this does
/// one call at a time and waits for none: a call is put to start, at most `width` run at once,
/// the rest wait their turn, and [`Arrivals::next`] starts what it can and hands back whichever
/// has come back first. One that is waiting its turn can still be taken back
/// ([`Arrivals::withdraw`]) and is then never run. Nothing waits on a clock: `next` waits for a result.
///
/// A call that is still running when the arrivals are dropped runs to its end on its own
/// thread, and what it comes to goes nowhere. It holds what it was given, and so a call that
/// should end sooner is told so through what it was given (a call's stop, [`crate::Call::stopped`]).
///
/// Where no thread can be started every call runs on this one when `next` asks for a result,
/// in the order they were started: one that was withdrawn first is never run.
pub(crate) struct Arrivals<T> {
    width: usize,
    waiting: VecDeque<(usize, Pending<T>)>,
    running: usize,
    sent: Sender<Came<T>>,
    came: Receiver<Came<T>>,
    /// Whether a call alone runs where it is asked for ([`Arrivals::alone_here`]).
    alone_here: bool,
}

impl<T: Send + 'static> Arrivals<T> {
    /// No calls yet, at most `width` of them to run at once.
    pub(crate) fn new(width: usize) -> Self {
        let (sent, came) = channel();
        Arrivals {
            width: width.max(1),
            waiting: VecDeque::new(),
            running: 0,
            sent,
            came,
            alone_here: false,
        }
    }

    /// The same, where a call that is the only one, with none running beside it, runs where it
    /// is asked for and not on a thread of its own: as calls run at once and joined always ran
    /// one alone. For whoever has nothing to do while it runs but wait for it.
    pub(crate) fn alone_here(mut self) -> Self {
        self.alone_here = true;
        self
    }

    /// Put a call, known by `position`, to start when [`Arrivals::next`] is next asked for a
    /// result and fewer than `width` run: until then it can be withdrawn.
    pub(crate) fn start(&mut self, position: usize, call: impl FnOnce() -> Result<T> + Send + 'static) {
        self.waiting.push_back((position, Box::new(call)));
    }

    /// Take back a call that is waiting its turn: it is never run. False where it has started.
    pub(crate) fn withdraw(&mut self, position: usize) -> bool {
        let before = self.waiting.len();
        self.waiting.retain(|(at, _)| *at != position);
        self.waiting.len() < before
    }

    /// What wakes [`Arrivals::next`] with [`Came::Woken`] while calls run: for a call to say
    /// it has something before it comes back. Once nothing is running, it wakes no one.
    pub(crate) fn waker(&self) -> impl Fn() + Send + Sync + 'static {
        let sent = Mutex::new(self.sent.clone());
        move || {
            let _ = sent.lock().unwrap_or_else(PoisonError::into_inner).send(Came::Woken);
        }
    }

    /// The next result, as it comes, with the position of its call, or a word from a call
    /// still running. None when no call is running or waiting. A call that panicked is handed
    /// back as its panic, for the caller to raise.
    pub(crate) fn next(&mut self) -> Option<Came<T>> {
        // One alone, with none running, runs here: no thread is started to be waited for.
        if self.alone_here && self.running == 0 && self.waiting.len() == 1 {
            while self.came.try_recv().is_ok() {}
            let (position, call) = self.waiting.pop_front()?;
            return Some(Came::Back(position, catch_unwind(AssertUnwindSafe(|| nested(call)))));
        }
        // What waits starts only now: a call that came back makes room for one, but whoever
        // reads what came back may withdraw it first.
        self.fill();
        if self.running == 0 {
            // Nothing could be started on a thread of its own: the next runs here.
            // What a call that ran here said while it ran is past: only what it came to is handed back.
            while self.came.try_recv().is_ok() {}
            let (position, call) = self.waiting.pop_front()?;
            return Some(Came::Back(position, catch_unwind(AssertUnwindSafe(|| nested(call)))));
        }
        let came = self.came.recv().ok()?;
        if let Came::Back(..) = came {
            self.running -= 1;
        }
        Some(came)
    }

    /// A result that has come already, if one has, without waiting for one.
    pub(crate) fn ready(&mut self) -> Option<Came<T>> {
        let came = self.came.try_recv().ok()?;
        if let Came::Back(..) = came {
            self.running -= 1;
        }
        Some(came)
    }

    /// Start what is waiting, while fewer than `width` run.
    fn fill(&mut self) {
        while THREADS && self.running < self.width {
            let Some((position, call)) = self.waiting.pop_front() else { return };
            let held: Started<T> = Arc::new(Mutex::new(Some(call)));
            let (taken, sent) = (held.clone(), self.sent.clone());
            let thread = std::thread::Builder::new().stack_size(STACK).spawn(move || {
                let call = taken.lock().unwrap_or_else(PoisonError::into_inner).take();
                if let Some(call) = call {
                    let _ = sent.send(Came::Back(position, catch_unwind(AssertUnwindSafe(|| nested(call)))));
                }
            });
            match thread {
                Ok(_) => self.running += 1,
                Err(_) => {
                    // No thread could be started: it waits again, first, and runs here when asked for.
                    if let Some(call) = held.lock().unwrap_or_else(PoisonError::into_inner).take() {
                        self.waiting.push_front((position, call));
                    }
                    return;
                }
            }
        }
    }
}
