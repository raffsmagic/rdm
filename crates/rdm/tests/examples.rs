#![cfg(all(feature = "filter", feature = "fit", feature = "near", feature = "lm", feature = "repl"))]
// This test starts cargo and the examples, and a WebAssembly module can start no program. It
// is run on this machine, and runs the examples as WebAssembly when it is asked to (below).
#![cfg(not(target_family = "wasm"))]
//! The examples, run: each program of `examples/` is built, run with the stand-in models and no
//! network, and what it prints is set beside what it is recorded to print.
//!
//! A record is a transcript in `tests/examples/`: `<example>.txt` for the cell as it is written,
//! and `<example>.<case>.txt` for the same program with other inputs. Its first line is the
//! command, as it would be typed, and the rest is what the program then prints:
//!
//! ```text
//! $ cargo run -p rdm --example act -- --task 3
//! {'switch': 'door: now closed'}
//! ```
//!
//! - One example alone, while others are being written: `cargo test -p rdm --test examples -- --exact act`.
//!   Only that example is built.
//! - To write down what an example prints now: `RDM_EXAMPLES_RECORD=1` with the same command.
//!   A transcript is meant to be what the Python cell prints, so read what was recorded:
//!   `tests/examples/python/cells.py` records one from the Python cell itself, and says whether
//!   the Rust example prints and sends the same.
//! - An example of the table that is still a placeholder passes its own test here and fails
//!   `tests/yet_to_come.rs`, which is how a missing port fails the suite.
//! - The examples as WebAssembly: `RDM_EXAMPLES_TARGET=wasm32-wasip1` with the same command.
//!   Each is built for that target and run by `wasmtime`, given a directory of its own, and
//!   is to print what it prints here. `RDM_EXAMPLES_RUNNER` names another runtime, as the
//!   words that go before the program, with `{dir}` where that directory is named: one that
//!   gives the program the directory as the one it is run in, and its own environment, and,
//!   for `wasm32-wasip2`, the `wasi:http` its programs import.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[path = "../examples/common/py.rs"]
mod py;
#[path = "examples/table.rs"]
mod table;

use table::{PLACEHOLDER, TABLE};

/// How long one run of an example may take. On stand-ins each is a moment.
const PATIENCE: Duration = Duration::from_secs(120);

/// The target the examples are built for and run on, where it is not this machine: `RDM_EXAMPLES_TARGET`.
fn target() -> Option<String> {
    std::env::var("RDM_EXAMPLES_TARGET").ok().filter(|target| !target.is_empty())
}

/// What runs a program built for another target, as the words that go before it:
/// `RDM_EXAMPLES_RUNNER`, or wasmtime. The program is given `scratch` as the directory it is
/// run in, and the environment the runtime is run with. A program built for WASI 0.2 imports
/// the host's HTTP whether it comes to ask or not, so it is given that too; a stand-in never asks.
fn runner(target: &str, scratch: &Path) -> Result<Vec<String>, String> {
    let named = std::env::var("RDM_EXAMPLES_RUNNER").ok().filter(|runner| !runner.is_empty());
    let http = if target.ends_with("wasip2") { "-S http " } else { "" };
    let runner = named.unwrap_or_else(|| format!("wasmtime run {http}-S inherit-env --dir={{dir}}::."));
    let words = words(&runner)?.into_iter();
    Ok(words.map(|word| word.replace("{dir}", &scratch.display().to_string())).collect())
}

macro_rules! examples {
    ($($name:ident),* $(,)?) => {$(
        #[test]
        fn $name() {
            check(stringify!($name));
        }
    )*};
}

// One test for each example of the table, by its name, and one for the post's own program
// if it comes to have one.
examples!(
    triage, act, zen, outline, handbook, choose, filter, hierarchy, fit, map, near, dimensions, route, repl, write_map, starter, home,
    post,
);

/// An example beyond these is found by its file and its transcripts, and run here.
#[test]
fn any_other_example() {
    let known = |name: &String| TABLE.contains(&name.as_str()) || name == "post";
    let mut others: BTreeSet<String> = transcripts().into_iter().map(|(name, _)| name).collect();
    others.extend(sources());
    for name in others.into_iter().filter(|name| !known(name)) {
        check(&name);
    }
}

#[test]
fn values_are_printed_as_python_prints_them() {
    use serde_json::json;
    // Each expected text is what python3 prints for the same value.
    assert_eq!(py::repr(&json!({"switch": "tv: now on"})), "{'switch': 'tv: now on'}");
    assert_eq!(py::repr(&json!({})), "{}");
    assert_eq!(py::repr(&json!(["Flat is better than nested."])), "['Flat is better than nested.']");
    assert_eq!(py::repr(&json!("let's do more of those!")), r#""let's do more of those!""#);
    assert_eq!(py::repr(&json!(r#"both ' and ""#)), r#"'both \' and "'"#);
    assert_eq!(
        py::repr(&json!("a\nb\t\\ \u{7f}\u{a0}é\u{200b}")),
        concat!(r"'a\nb\t\\ \x7f\xa0é\", "u200b'")
    );
    assert_eq!(
        py::repr(&json!([true, false, null, 21, -3, 0.5, 1.0, 1e-5, 1e16, 0.1 + 0.2])),
        "[True, False, None, 21, -3, 0.5, 1.0, 1e-05, 1e+16, 0.30000000000000004]"
    );
    assert_eq!(
        py::repr(&json!({"tv": "off", "heat": 21, "rooms": {"entry": ["door", "lock"]}})),
        "{'tv': 'off', 'heat': 21, 'rooms': {'entry': ['door', 'lock']}}"
    );
    // `str` is `repr` but for a string, which is shown as it is: what `print` does with each value.
    assert_eq!(py::str(&"upset: "), "upset: ");
    assert_eq!(py::str(&json!(true)), "True");
    assert_eq!(py::str(&9usize), "9");
    assert_eq!(py::str(&vec!["lock locked", "finish"]), "['lock locked', 'finish']");
    assert_eq!(py::line(&[py::str(&"->"), py::str(&json!(["a"]))]), "-> ['a']");
}

#[test]
fn a_transcript_begins_with_its_command() {
    let read = |line: &str| Run::read("act", line).map(|run| (run.environment, run.arguments));
    assert_eq!(read("$ cargo run -p rdm --example act"), Ok((vec![], vec![])));
    assert_eq!(
        read(r#"$ RDM_TRACE=1 cargo run -p rdm --example act -- --task "Let's \"go\"." --house '{"tv": "on"}' 3"#),
        Ok((
            vec![("RDM_TRACE".into(), "1".into())],
            vec![
                "--task".into(),
                r#"Let's "go"."#.into(),
                "--house".into(),
                r#"{"tv": "on"}"#.into(),
                "3".into()
            ]
        ))
    );
    assert!(read("$ cargo run -p rdm --example zen").is_err(), "a transcript of another example");
    assert!(read("cargo run -p rdm --example act").is_err(), "no $");
    assert!(read("$ cargo run -p rdm --example act -- 'open").is_err(), "a quote left open");
    assert_eq!(
        differences("a\nb\nc\n", "a\nx\nc\nd\n"),
        ["    a", "  - b", "  + x", "    c", "  + d"].map(str::to_owned)
    );
}

/// The ways to a model that is not a stand-in, each asked on this machine and no further: a
/// System One server by its URL, the same server through a gateway's registry file, and a
/// language model answering decision questions. Each is asked for the `act` cell as it is
/// written, and the example prints what it prints on its stand-in.
#[test]
fn an_example_asks_the_models_the_environment_names() {
    use one_system::{Backend, Gateway, Server};
    use rdm::systemone::{Answer, Scripted};

    // The servers asked here are on this machine, and so are the files that name them: an
    // example run as WebAssembly is given neither, and what it prints on its stand-ins is checked above.
    if let Some(target) = target() {
        return eprintln!("not run for {target}: the models asked are servers on this machine.");
    }
    let program = built("act").unwrap_or_else(|why| panic!("\nThe example `act` does not build.\n\n{why}\n"));
    let transcript = fs::read_to_string(recorded().join("act.txt")).expect("the transcript of act");
    let expected = transcript.split_once('\n').expect("a command, and what it prints").1;
    let ask = |environment: &[(&str, &str)]| {
        let run = Run {
            name: "act".into(),
            environment: environment
                .iter()
                .map(|(variable, value)| (variable.to_string(), value.to_string()))
                .collect(),
            arguments: Vec::new(),
        };
        run.run(&program).expect("a run of act")
    };
    let asked = |environment: &[(&str, &str)]| {
        let ran = ask(environment);
        assert!(ran.succeeded, "{environment:?}\n{}", ran.complained);
        assert_eq!(ran.printed, expected, "{environment:?}\n{}", ran.complained);
        ran.complained
    };
    let refused = |environment: &[(&str, &str)]| {
        let ran = ask(environment);
        assert!(!ran.succeeded && ran.printed.is_empty(), "{environment:?}\n{}", ran.printed);
        ran.complained
    };

    // With nothing asked for and no key, the stand-in, and a line that says so.
    let said = asked(&[("RDM_DECISION_MODEL", "")]);
    assert!(
        said.starts_with("decision model: the stand-in") && said.lines().count() == 1,
        "{said}"
    );
    assert_eq!(asked(&[]), "", "a model that was asked for is not announced");
    // A model that needs a key, with none: one line, before anything is asked.
    assert_eq!(
        refused(&[("RDM_DECISION_MODEL", "typesafe")]),
        "Error: RDM_DECISION_MODEL=typesafe needs a key in TYPESAFE_API_KEY.\n"
    );
    assert_eq!(
        refused(&[("RDM_DECISION_MODEL", "openrouter")]),
        "Error: RDM_DECISION_MODEL=openrouter needs a key in OPENROUTER_API_KEY.\n"
    );
    assert!(refused(&[("RDM_DECISION_MODEL", "jev")]).starts_with("Error: RDM_DECISION_MODEL=jev is not one of: standin, typesafe,"));

    // A System One server on this machine, which asks for a key.
    let jev = Scripted::new(|request| {
        let answer = |name: &str, question: &rdm::systemone::Question| match name {
            "switch" => Answer::noul(0.97),
            "device" => Answer::chose("tv", question.labels().iter().map(String::as_str)),
            _ => Answer::chose("on", question.labels().iter().map(String::as_str)),
        };
        let each = request
            .questions
            .iter()
            .map(|(name, question)| (name.clone(), answer(name, question)));
        each.collect::<Vec<_>>()
    });
    let gateway = Gateway::builder("served")
        .backend(Backend::new("only", jev))
        .selector("only")
        .build()
        .expect("a gateway");
    let server = Server::bind("127.0.0.1:0", gateway, "the-key-of-this-test")
        .expect("a server")
        .spawn();
    let url = server.url();
    let key = ("SERVER_KEY", "the-key-of-this-test");
    let served = ("RDM_DECISION_MODEL_NAME", "served");
    asked(&[("RDM_DECISION_MODEL", &url), ("RDM_DECISION_MODEL_KEY", "SERVER_KEY"), key, served]);
    let said = refused(&[("RDM_DECISION_MODEL", &url), served]);
    assert!(said.starts_with("Error: ") && said.lines().count() == 1, "{said}");
    // A server of several models that is asked for none refuses the request: what is missing is said.
    let said = refused(&[("RDM_DECISION_MODEL", &url), ("RDM_DECISION_MODEL_KEY", "SERVER_KEY"), key]);
    assert!(said.contains("422 schema_validation") && said.lines().count() == 1, "{said}");
    assert!(
        said.ends_with("No model was named: a server of several, as a one-system gateway is, wants one in RDM_DECISION_MODEL_NAME.\n"),
        "{said}"
    );

    // The same server, behind a gateway a registry file describes.
    let registry = std::env::temp_dir().join(format!("rdm-example-registry-{}.json", std::process::id()));
    let backend = serde_json::json!({"id": "served", "base_url": url, "model": "served", "api_key_env": "SERVER_KEY", "description": "The server of this test."});
    let written = serde_json::json!({"name": "examples", "selector": "served", "backends": [backend]});
    fs::write(&registry, written.to_string()).expect("a registry file");
    asked(&[("RDM_DECISION_MODEL", &format!("gateway:{}", registry.display())), key]);
    let said = refused(&[("RDM_DECISION_MODEL", "gateway:/there/is/no/such/registry.json")]);
    assert!(
        said.starts_with("Error: The registry /there/is/no/such/registry.json does not load"),
        "{said}"
    );
    let _ = fs::remove_file(&registry);
    server.stop();

    // A language model that speaks Chat Completions, answering the decision questions as evidence.
    let evidence = serde_json::json!({
        "switch": {"noul": 0.97},
        "device": {"probabilities": {"tv": 1.0, "lights": 0.0, "door": 0.0, "lock": 0.0}, "confidence": 1.0},
        "to": {"probabilities": {"on": 1.0, "off": 0.0, "open": 0.0, "closed": 0.0, "locked": 0.0, "unlocked": 0.0}, "confidence": 1.0},
    });
    let writer = chat_server(evidence.to_string());
    asked(&[
        ("RDM_DECISION_MODEL", "lm"),
        ("RDM_LANGUAGE_MODEL", &writer),
        ("RDM_LANGUAGE_MODEL_NAME", "a-model"),
    ]);
    // These runs name the stand-in as their language model unless told otherwise: here none is named.
    let said = refused(&[("RDM_DECISION_MODEL", "lm"), ("RDM_LANGUAGE_MODEL", "")]);
    assert!(
        said.starts_with("Error: RDM_DECISION_MODEL=lm asks a language model to decide, and none is named"),
        "{said}"
    );
    // A stand-in for a language model was named, and cannot decide: that is said, and not that none was named.
    for stand_in in ["standin", "stand-in"] {
        assert_eq!(
            refused(&[("RDM_DECISION_MODEL", "lm"), ("RDM_LANGUAGE_MODEL", stand_in)]),
            format!(
                "Error: RDM_DECISION_MODEL=lm asks a language model to decide, and RDM_LANGUAGE_MODEL={stand_in} is a stand-in, \
                 which cannot: name a real language model, or set RDM_DECISION_MODEL=standin.\n"
            )
        );
    }
}

/// A stand-in for a server that speaks Chat Completions, on this machine: whatever it is
/// asked, the assistant says `reply`. Returns its address, `/v1` included.
fn chat_server(reply: String) -> String {
    use std::io::{BufRead, BufReader, Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a port on this machine");
    let address = listener.local_addr().expect("its address");
    std::thread::spawn(move || {
        for connection in listener.incoming().flatten() {
            let mut reader = BufReader::new(connection);
            let mut length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; length];
            if reader.read_exact(&mut body).is_err() {
                continue;
            }
            let completion = serde_json::json!({
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "a-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            .to_string();
            let answer = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{completion}",
                completion.len()
            );
            let _ = reader.get_mut().write_all(answer.as_bytes());
        }
    });
    format!("http://{address}/v1")
}

// ---- Where things are.

fn package() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn source(name: &str) -> PathBuf {
    package().join("examples").join(format!("{name}.rs"))
}

fn recorded() -> PathBuf {
    package().join("tests").join("examples")
}

/// The examples there are files for.
fn sources() -> Vec<String> {
    let files = fs::read_dir(package().join("examples")).expect("the examples");
    let names = files
        .flatten()
        .map(|file| file.path())
        .filter(|path| path.extension().is_some_and(|ends| ends == "rs"));
    names.filter_map(|path| Some(path.file_stem()?.to_str()?.to_owned())).collect()
}

/// Every transcript, with the example it is of: the name of the file up to its first dot.
fn transcripts() -> Vec<(String, PathBuf)> {
    let Ok(files) = fs::read_dir(recorded()) else {
        return Vec::new();
    };
    let mut found: Vec<(String, PathBuf)> = files
        .flatten()
        .map(|file| file.path())
        .filter(|path| path.extension().is_some_and(|ends| ends == "txt"))
        .filter_map(|path| Some((path.file_name()?.to_str()?.split('.').next()?.to_owned(), path)))
        .collect();
    found.sort();
    found
}

// ---- One example.

/// Builds an example, runs each of its transcripts, and fails with what differs.
fn check(name: &str) {
    let recording = std::env::var("RDM_EXAMPLES_RECORD").is_ok_and(|value| !value.is_empty() && value != "0");
    let Ok(written) = fs::read_to_string(source(name)) else {
        return eprintln!("{name}: there is no examples/{name}.rs.");
    };
    if written.contains(PLACEHOLDER) {
        return eprintln!("{name}: to come. examples/{name}.rs is still the placeholder.");
    }
    let mut cases: Vec<PathBuf> = transcripts()
        .into_iter()
        .filter(|(of, _)| of == name)
        .map(|(_, path)| path)
        .collect();
    let first = recorded().join(format!("{name}.txt"));
    if !first.exists() {
        assert!(
            recording,
            "\nThe example `{name}` has no tests/examples/{name}.txt: what it prints as the cell is written.\n\
             To write down what it prints now: RDM_EXAMPLES_RECORD=1 cargo test -p rdm --test examples -- --exact {name}\n"
        );
        fs::create_dir_all(recorded()).expect("the directory of transcripts");
        fs::write(&first, format!("$ cargo run -p rdm --example {name}\n")).expect("a new transcript");
        cases.insert(0, first);
    }
    let program = built(name).unwrap_or_else(|why| panic!("\nThe example `{name}` does not build.\n\n{why}\n"));
    let mut failures = Vec::new();
    for case in cases {
        let shown = case.strip_prefix(package()).unwrap_or(&case).display().to_string();
        let transcript = fs::read_to_string(&case).expect("a transcript");
        let (command, expected) = transcript.split_once('\n').unwrap_or((&transcript, ""));
        let run = match Run::read(name, command) {
            Ok(run) => run,
            Err(why) => {
                failures.push(format!("{shown}: {why}"));
                continue;
            }
        };
        let ran = match run.run(&program) {
            Ok(ran) => ran,
            Err(why) => {
                failures.push(format!("{shown}: {why}"));
                continue;
            }
        };
        if !ran.succeeded {
            failures.push(format!(
                "{shown}: the example failed.\n\n  {command}\n\nWhat it printed:\n{}\nWhat it wrote to stderr:\n{}",
                indented(&ran.printed),
                indented(&ran.complained)
            ));
        } else if recording {
            fs::write(&case, format!("{command}\n{}", ran.printed)).expect("a transcript");
            eprintln!("{shown}: recorded.");
        } else if ran.printed != expected {
            failures.push(format!(
                "{shown}: the example did not print what is recorded.\n\n  {command}\n\n  - recorded, + printed\n{}\n\nWhat it wrote to stderr:\n{}\n\
                 To write down what it prints now: RDM_EXAMPLES_RECORD=1 cargo test -p rdm --test examples -- --exact {name}",
                differences(expected, &ran.printed).join("\n"),
                indented(&ran.complained)
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n\n"));
}

fn indented(text: &str) -> String {
    text.lines().map(|line| format!("    {line}\n")).collect()
}

/// The example's program, built now, so that it is the source as it is. It is built by the
/// cargo that runs this test, into the directory this test was built into: after `cargo test`
/// that is a check that nothing changed, and after `cargo test --test examples` it is the
/// build of this one example, whatever state the others are in.
fn built(name: &str) -> Result<PathBuf, String> {
    let test = std::env::current_exe().map_err(|error| error.to_string())?;
    // target/[<triple>/]<profile>/deps/examples-<hash>
    let profile = test
        .parent()
        .and_then(Path::parent)
        .ok_or("This test is not where cargo builds one.")?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut build = Command::new(cargo);
    build
        .current_dir(package())
        .args(["build", "--offline", "-p", "rdm", "--example", name]);
    // What cargo told this test about its own package is not for the cargo it starts: a build
    // script that reads one of these would be run again for every crate that has one, and
    // again by the next `cargo test`.
    const OF_THIS_TEST: [&str; 7] = [
        "CARGO_MANIFEST_DIR",
        "CARGO_MANIFEST_PATH",
        "CARGO_CRATE_NAME",
        "CARGO_BIN_NAME",
        "CARGO_PRIMARY_PACKAGE",
        "CARGO_TARGET_TMPDIR",
        "CARGO_SBOM_PATH",
    ];
    for (variable, _) in std::env::vars_os() {
        let named = variable.to_string_lossy();
        if named.starts_with("CARGO_PKG_") || OF_THIS_TEST.contains(&named.as_ref()) {
            build.env_remove(&variable);
        }
    }
    match profile.file_name().and_then(|named| named.to_str()) {
        Some("debug") => {}
        Some("release") => drop(build.arg("--release")),
        Some(other) => drop(build.args(["--profile", other])),
        None => {}
    }
    // The directory cargo builds into is the one with its tag in it: the profile's own, or
    // the one above that when the build is for a named target.
    let above = |directory: &Path| directory.parent().map(Path::to_owned);
    let is_target = |directory: &PathBuf| directory.join("CACHEDIR.TAG").exists();
    if let Some(over) = above(profile) {
        if let Some(target) = target() {
            build.arg("--target-dir").arg(over);
            build.arg("--target").arg(target);
        } else if is_target(&over) {
            build.arg("--target-dir").arg(over);
        } else if let Some(target) = above(&over).filter(is_target) {
            build.arg("--target-dir").arg(target);
            build.arg("--target").arg(over.file_name().unwrap_or_default());
        }
    }
    let done = build
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cargo could not be run: {error}"))?;
    if !done.status.success() {
        return Err(String::from_utf8_lossy(&done.stderr).into_owned());
    }
    let program = match target() {
        // target/<triple>/<profile>/examples, beside this machine's own target/<profile>.
        Some(target) => {
            let over = above(profile).ok_or("This test is not where cargo builds one.")?;
            let suffix = if target.starts_with("wasm") { ".wasm" } else { "" };
            let built = over.join(&target).join(profile.file_name().unwrap_or_default());
            built.join("examples").join(format!("{name}{suffix}"))
        }
        None => profile.join("examples").join(format!("{name}{}", std::env::consts::EXE_SUFFIX)),
    };
    match program.exists() {
        true => Ok(program),
        false => Err(format!("cargo built it, but not at {}.", program.display())),
    }
}

/// One run of an example: what its transcript's first line says to set and to give it.
#[derive(Debug, PartialEq)]
struct Run {
    name: String,
    environment: Vec<(String, String)>,
    arguments: Vec<String>,
}

/// What a run came to.
struct Ran {
    succeeded: bool,
    printed: String,
    complained: String,
}

impl Run {
    /// Reads `$ [NAME=value ...] cargo run -p rdm --example <name> [-- arguments]`.
    fn read(name: &str, line: &str) -> Result<Run, String> {
        let form =
            format!("its first line is to be `$ cargo run -p rdm --example {name}`, with variables before it and arguments after `--`");
        let command = line.strip_prefix("$ ").ok_or(&form)?;
        let mut words = words(command)?.into_iter().peekable();
        let mut environment = Vec::new();
        while let Some((variable, value)) = words.peek().and_then(|word| word.split_once('=')) {
            if variable.is_empty()
                || !variable
                    .chars()
                    .all(|each| each.is_ascii_uppercase() || each.is_ascii_digit() || each == '_')
            {
                break;
            }
            environment.push((variable.to_owned(), value.to_owned()));
            words.next();
        }
        let rest: Vec<String> = words.collect();
        let cargo = ["cargo", "run", "-p", "rdm", "--example", name];
        if rest.len() < cargo.len() || rest[..cargo.len()] != cargo {
            return Err(form);
        }
        let arguments = match rest.get(cargo.len()).map(String::as_str) {
            None => Vec::new(),
            Some("--") => rest[cargo.len() + 1..].to_vec(),
            Some(_) => return Err(form),
        };
        Ok(Run {
            name: name.to_owned(),
            environment,
            arguments,
        })
    }

    /// Runs the program with the stand-in models, in a directory of its own, with nothing of
    /// this environment that could reach a real model: no key, and no choice of model.
    fn run(&self, program: &Path) -> Result<Ran, String> {
        static RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let run = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let scratch = std::env::temp_dir().join(format!("rdm-example-{}-{}-{run}", self.name, std::process::id()));
        fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
        let file = |named: &str| fs::File::create(scratch.join(named)).map_err(|error| error.to_string());
        let mut command = match target() {
            // The runtime is started where this test is, which is where a tool that keeps a
            // version of it for this project looks for one. The program is not: it is in `scratch`.
            Some(target) => {
                let runner = runner(&target, &scratch)?;
                let (runtime, given) = runner.split_first().ok_or("RDM_EXAMPLES_RUNNER names no program")?;
                let mut command = Command::new(runtime);
                command.args(given).arg(program).current_dir(package());
                command
            }
            None => {
                let mut command = Command::new(program);
                command.current_dir(&scratch);
                command
            }
        };
        command.args(&self.arguments).env_clear();
        for kept in ["PATH", "SystemRoot", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(kept) {
                command.env(kept, value);
            }
        }
        command.env("RDM_DECISION_MODEL", "standin").env("RDM_LANGUAGE_MODEL", "standin");
        command.envs(self.environment.iter().map(|(variable, value)| (variable, value)));
        command.stdin(Stdio::null()).stdout(file("stdout")?).stderr(file("stderr")?);
        let mut child = command.spawn().map_err(|error| format!("the program could not be run: {error}"))?;
        let started = Instant::now();
        let status = loop {
            match child.try_wait().map_err(|error| error.to_string())? {
                Some(status) => break status,
                None if started.elapsed() > PATIENCE => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "it was still running after {} seconds, and was stopped.",
                        PATIENCE.as_secs()
                    ));
                }
                None => std::thread::sleep(Duration::from_millis(5)),
            }
        };
        let read = |named: &str| String::from_utf8_lossy(&fs::read(scratch.join(named)).unwrap_or_default()).into_owned();
        let ran = Ran {
            succeeded: status.success(),
            printed: read("stdout"),
            complained: read("stderr"),
        };
        let _ = fs::remove_dir_all(&scratch);
        Ok(ran)
    }
}

/// The words of a command, as a shell reads them: split at spaces, with `'...'` kept as it is,
/// and `"..."` kept but for `\"` and `\\`.
fn words(command: &str) -> Result<Vec<String>, String> {
    let (mut words, mut word, mut within) = (Vec::new(), String::new(), false);
    let mut characters = command.chars();
    while let Some(character) = characters.next() {
        match character {
            ' ' | '\t' if within => {
                words.push(std::mem::take(&mut word));
                within = false;
            }
            ' ' | '\t' => {}
            '\'' => {
                within = true;
                loop {
                    match characters.next().ok_or("a ' is left open")? {
                        '\'' => break,
                        other => word.push(other),
                    }
                }
            }
            '"' => {
                within = true;
                loop {
                    match characters.next().ok_or("a \" is left open")? {
                        '"' => break,
                        '\\' => match characters.next().ok_or("a \" is left open")? {
                            escaped @ ('"' | '\\' | '$' | '`') => word.push(escaped),
                            other => word.extend(['\\', other]),
                        },
                        other => word.push(other),
                    }
                }
            }
            '\\' => {
                within = true;
                word.push(characters.next().ok_or("a \\ ends the line")?);
            }
            other => {
                within = true;
                word.push(other);
            }
        }
    }
    if within {
        words.push(word);
    }
    Ok(words)
}

/// What differs between what is recorded and what was printed, line by line: a line only
/// recorded after `-`, a line only printed after `+`, and the lines they share between them.
fn differences(recorded: &str, printed: &str) -> Vec<String> {
    let (old, new): (Vec<&str>, Vec<&str>) = (recorded.lines().collect(), printed.lines().collect());
    // The longest run of lines the two share, from each place on, found from the end.
    let mut shared = vec![vec![0usize; new.len() + 1]; old.len() + 1];
    for at in (0..old.len()).rev() {
        for to in (0..new.len()).rev() {
            shared[at][to] = if old[at] == new[to] {
                shared[at + 1][to + 1] + 1
            } else {
                shared[at + 1][to].max(shared[at][to + 1])
            };
        }
    }
    let (mut at, mut to, mut lines) = (0, 0, Vec::new());
    while at < old.len() || to < new.len() {
        if at < old.len() && to < new.len() && old[at] == new[to] {
            lines.push(format!("    {}", old[at]));
            (at, to) = (at + 1, to + 1);
        } else if at < old.len() && (to == new.len() || shared[at + 1][to] >= shared[at][to + 1]) {
            lines.push(format!("  - {}", old[at]));
            at += 1;
        } else {
            lines.push(format!("  + {}", new[to]));
            to += 1;
        }
    }
    if recorded.ends_with('\n') != printed.ends_with('\n') {
        lines.push("  (and one ends with a new line where the other does not)".to_owned());
    }
    lines
}
