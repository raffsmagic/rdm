#![cfg(all(feature = "filter", feature = "fit", feature = "near", feature = "lm", feature = "repl"))]
// What is read here is the source of the examples, where this package is on this machine. A
// test run as WebAssembly is given the directory it is run in, by a name of its own.
#![cfg(not(target_family = "wasm"))]
//! The examples that are yet to come. Every notebook cell of the site is to be a program in
//! `examples/`, with what it prints recorded in `tests/examples/`: that is what this project is
//! finished by. This fails for as long as one of them is missing or is still its placeholder,
//! and says which.
//!
//! It is a test target of its own, and the last by name, so that while it fails `cargo test`
//! has already run every other: only the doc tests come after it, and `--no-fail-fast` runs those.

use std::fs;
use std::path::Path;

#[path = "examples/table.rs"]
mod table;

use table::{PLACEHOLDER, TABLE};

#[test]
fn no_example_of_the_table_is_yet_to_come() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR"));
    let state = |name: &&str| match fs::read_to_string(package.join("examples").join(format!("{name}.rs"))) {
        Err(_) => Some(format!("{name}: there is no examples/{name}.rs")),
        Ok(written) if written.contains(PLACEHOLDER) => Some(format!("{name}: examples/{name}.rs is still the placeholder")),
        Ok(_) if !package.join("tests").join("examples").join(format!("{name}.txt")).exists() => Some(format!(
            "{name}: tests/examples/{name}.txt, what the cell prints as it is written, is not recorded"
        )),
        Ok(_) => None,
    };
    let missing: Vec<String> = TABLE.iter().filter_map(state).collect();
    assert!(
        missing.is_empty(),
        "\n{} of the {} examples are yet to come:\n  {}\n",
        missing.len(),
        TABLE.len(),
        missing.join("\n  ")
    );
}
