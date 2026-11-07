#![cfg(feature = "repl")]
//! The same code, given the same answers, prints the same and ends the same way: in two REPLs
//! of one process, and in two processes. It is what a replay of a run rests on. What the code
//! is answered from outside is its functions, its randomness and its clock, and so the
//! randomness and the clock here are the test's own.

use rdm::Json;
use rdm::repl::{Cell, Ending, Limits, Repl};
use serde_json::Map;

/// Code that would print something else in another process if anything in it were drawn from
/// the process: hashes, the order of sets and dicts, randomness, the clocks, ids, floats.
const CELLS: [&str; 16] = [
    "print(hash('abc'), hash(b'abc'), hash((1, 'a')), hash(frozenset({1, 2})), hash(3.5), hash(None))",
    "print({'pear', 'apple', 'fig', 'kiwi', 'plum'}, set(range(20, 0, -3)), {('a', 1), ('b', 2), 'c'})",
    "s = {'x', 'y', 'z'} | {'w', 'v'}\nprint(s, s - {'y'}, frozenset(s), sorted(s))",
    "d = {k: len(k) for k in ['bb', 'a', 'ccc']}\nd.pop('a')\nd['z'] = 0\nprint(d, list(d.items()))",
    "import random\nprint(random.random(), random.randint(1, 10 ** 6), random.choice('abcdef'))",
    "import random\nr = random.Random()\nmixed = list(range(10))\nrandom.shuffle(mixed)\nprint(r.random(), mixed)",
    "import os\nprint(list(os.urandom(6)))",
    "import random\nrandom.seed(42)\nprint(random.random())",
    "import time\nprint(time.time())",
    "import time\nprint(time.monotonic())",
    "import datetime\neast = datetime.timezone(datetime.timedelta(hours=-5), 'EST')\nprint(datetime.datetime.now(), datetime.datetime.now(east), datetime.date.today())",
    "import time\nprint(time.process_time())",
    "a, b = [1], [1]\nprint(id(a) == id(b), id(a) - id(b))",
    "import re\nprint(0.1 + 0.2, 2 ** 100, 1e300 * 10, re.findall(r'\\w+', 'a bb ccc'))",
    "print('half')\nx = {}['missing']",
    "x = 'x' * 10 ** 9",
];

/// Randomness that is the same wherever it is asked for.
fn given(count: usize) -> rdm::Result<Vec<u8>> {
    Ok((0..count).map(|at| (at.wrapping_mul(31).wrapping_add(7) % 251) as u8).collect())
}

/// A clock that is the same wherever it is read: half a second into the leap day of 2000.
const NOW: f64 = 951_782_400.5;

/// A REPL whose randomness and clock are the test's own.
fn fixed() -> Repl {
    Repl::new().randomness(given).clock(|| Ok(NOW))
}

/// Each cell, run in order in one REPL.
fn ran(repl: &Repl) -> Vec<Cell> {
    CELLS.iter().map(|code| repl.execute_cell(code, &Map::new()).unwrap()).collect()
}

#[test]
fn two_repls_given_the_same_randomness_print_the_same() {
    let (one, other) = (ran(&fixed()), ran(&fixed()));
    assert_eq!(one, other);
    // Seeded as CPython seeds it, it draws what CPython draws.
    assert_eq!(one[7].printed, "0.6394267984578837");
    // Every clock reads the REPL's. The process's own is no clock: it counts the code's time only, and is nought.
    assert_eq!([one[8].printed.as_str(), one[9].printed.as_str()], ["951782400.5"; 2]);
    assert_eq!(
        one[10].printed,
        "2000-02-29 00:00:00.500000 2000-02-28 19:00:00.500000-05:00 2000-02-29"
    );
    assert_eq!(one[11].printed, "0.0");
    assert!(matches!(one[14].ending, Ending::Raised { .. }) && one[14].printed == "half");
    assert!(matches!(one[15].ending, Ending::OutOfMemory { .. }));
}

#[test]
fn what_is_drawn_at_random_is_the_sources() {
    // Two REPLs with the operating system's randomness differ in what they drew, and in nothing else.
    let (one, other) = (ran(&Repl::new().clock(|| Ok(NOW))), ran(&Repl::new().clock(|| Ok(NOW))));
    let drawn = [4, 5, 6];
    for (at, (one, other)) in one.iter().zip(&other).enumerate() {
        assert_eq!(drawn.contains(&at), one != other, "cell {at}: {one:?} and {other:?}");
    }
    // A source that fails is an error the code can catch, and one that gives the wrong count ends the run.
    let failing = Repl::new().randomness(|_| Err("no randomness today".into()));
    let caught = "import os\ntry:\n    os.urandom(4)\nexcept RuntimeError as error:\n    print(error)";
    assert_eq!(failing.execute(caught).unwrap(), "no randomness today");
    let short = Repl::new().randomness(|count| Ok(vec![0; count / 2]));
    assert!(short.execute_cell("import os\nos.urandom(4)", &Map::new()).unwrap_err().is_usage());
    // More than the REPL may hold is refused before the source is asked, as any value too large is.
    let held = Repl::new().randomness(|_| panic!("asked for too much")).limits(Limits {
        memory: 1024,
        ..Limits::default()
    });
    let refused = held.execute_cell("import os\nos.urandom(10 ** 6)", &Map::new()).unwrap();
    assert!(
        matches!(&refused.ending, Ending::OutOfMemory { error } if error.starts_with("MemoryError")),
        "{refused:?}"
    );
}

#[test]
fn the_clock_is_the_repls_to_read() {
    let at = |seconds: f64| Repl::new().clock(move || Ok(seconds));
    let now = "import time, datetime\nprint(time.time(), datetime.datetime.now(), datetime.date.today())";
    // Before 1970, and in the last second of a year.
    assert_eq!(at(-1.5).execute(now).unwrap(), "-1.5 1969-12-31 23:59:58.500000 1969-12-31");
    assert_eq!(
        at(1_704_067_199.0).execute(now).unwrap(),
        "1704067199.0 2023-12-31 23:59:59 2023-12-31"
    );
    // An instant no date can hold is a date the code cannot have, and a time it can.
    let far = at(1e12);
    assert_eq!(far.execute("import time\nprint(time.time())").unwrap(), "1000000000000.0");
    let error = far.execute("import datetime\ndatetime.date.today()").unwrap_err().to_string();
    assert!(error.starts_with("OverflowError"), "{error}");
    // A clock that gives no instant, or an error, is refused, and the code can tell.
    let error = at(f64::NAN).execute("import time\ntime.time()").unwrap_err().to_string();
    assert_eq!(error, "PermissionError: the clock gave no instant");
    // The system's clock, unless the REPL is given another: it moves.
    let system = Repl::new();
    let read = |repl: &Repl| repl.execute("import time\nprint(time.time())").unwrap().parse::<f64>().unwrap();
    assert!(read(&system) > 1_700_000_000.0);
}

#[test]
fn how_each_cell_ended_is_data() {
    let repl = Repl::new()
        .ending("done(total)", "Give the total and stop.", |_| Ok(()))
        .function("broken()", "A function that ends the run.", |_| -> rdm::Result<()> {
            Err(rdm::Error::usage("the module was used wrongly"))
        })
        .limits(Limits {
            time: std::time::Duration::from_millis(50),
            ..Limits::default()
        });
    let cell = |code: &str| repl.execute_cell(code, &Map::new()).unwrap();
    assert_eq!(
        cell("print('fine')"),
        Cell {
            printed: "fine".to_owned(),
            ending: Ending::Ran
        }
    );
    assert_eq!(cell("print('before')\ndone(3)\nprint('after')").ending, Ending::Finished);
    // Out of time, the interpreter is made again, and what the code had made is gone.
    repl.execute("kept = 1").unwrap();
    assert_eq!(cell("while True:\n    pass").ending, Ending::TimedOut);
    assert!(matches!(cell("print(kept)").ending, Ending::Raised { error } if error.starts_with("NameError")));
    // A function that ends the run is the caller's error, as it is for `execute`.
    assert!(repl.execute_cell("broken()", &Map::new()).unwrap_err().is_usage());
    // As data, an ending says which it is, beside what was printed.
    let written = serde_json::to_value(cell("print('half')\n1 / 0")).unwrap();
    assert_eq!(
        written,
        serde_json::json!({"printed": "half", "ended": "raised", "error": "ZeroDivisionError: division by zero"})
    );
    assert_eq!(
        serde_json::from_value::<Cell>(written.clone()).unwrap(),
        cell("print('half')\n1 / 0")
    );
    let _: Json = written;
}

/// Where this test prints its cells for the test that starts it in another process.
const SHOW: &str = "RDM_DETERMINISM_SHOW";
const BEGIN: &str = "<<<cells";
const END: &str = "cells>>>";

#[test]
fn the_cells_as_this_process_runs_them() {
    let cells = serde_json::to_string(&ran(&fixed())).unwrap();
    if std::env::var_os(SHOW).is_some() {
        println!("{BEGIN}{cells}{END}");
    }
}

// Another process is started from this one, and a WebAssembly module can start none. Two
// modules can be set side by side by running this suite twice: the first test checks each.
#[test]
#[cfg(not(target_family = "wasm"))]
fn two_processes_print_the_same() {
    let here = ran(&fixed());
    let test = std::env::current_exe().unwrap();
    let other = std::process::Command::new(test)
        .args(["--exact", "the_cells_as_this_process_runs_them", "--nocapture", "--test-threads=1"])
        .env(SHOW, "1")
        .output()
        .unwrap();
    assert!(other.status.success(), "{}", String::from_utf8_lossy(&other.stderr));
    let printed = String::from_utf8_lossy(&other.stdout);
    let shown = printed
        .split_once(BEGIN)
        .and_then(|(_, rest)| rest.split_once(END))
        .map(|(cells, _)| cells);
    let there: Vec<Cell> = serde_json::from_str(shown.expect("the other process printed its cells")).unwrap();
    assert_eq!(here, there);
}
