//! What the examples share: the models the environment asks for, the form above a cell, the
//! house the smart home cells run, printing as Python prints, and a view of a run.
//!
//! An example begins `mod common;`. Each uses a part of this, so nothing here is unused code
//! to complain about. `examples/README.md` says how an example is laid out.
#![allow(dead_code, unused_imports, unused_macros)]

pub mod form;
pub mod house;
pub mod models;
pub mod py;
pub mod show;

pub use form::Form;
pub use house::{House, cooler, heat, house, switch, warmer};
pub use models::{Decider, Plans, Say, decision_model, language_model};
pub use show::{show_branch, show_house, show_leaf, show_text, show_tree, trace, tracing};

/// The handbook the notebook's cells search: 71 lines, 2,555 characters, four sections.
/// `open("handbook.md").read()` in Python.
pub const HANDBOOK: &str = include_str!("../assets/handbook.md");

/// The map `write_map` saved and `starter` loads, as the site ships it.
pub const STARTER_MAP: &str = include_str!("../assets/starter_map.json");

/// Shopify's product taxonomy, version 2026-02: a line for each of 12,378 categories, after
/// three lines of heading. The file the `hierarchy` cell fetches, kept here so that no run
/// needs the network for it.
pub const CATEGORIES: &str = include_str!("../assets/categories.txt");

/// `starter_map.json` as `open("starter_map.json")` finds it: the one in the directory the
/// program runs in, if `write_map --to starter_map.json` saved one there, or else the one the
/// site ships.
pub fn starter_map() -> String {
    std::fs::read_to_string("starter_map.json").unwrap_or_else(|_| STARTER_MAP.to_owned())
}

/// Why an example stopped: the message of whatever failed, printed as it reads.
///
/// `main` returns it, so any error can be passed up with `?`, and a key that is missing or a
/// server that is down is one plain line and not a debug dump.
pub struct Failure(String);

impl Failure {
    pub fn new(message: impl Into<String>) -> Self {
        Failure(message.into())
    }
}

impl<E: std::fmt::Display> From<E> for Failure {
    fn from(error: E) -> Self {
        Failure(error.to_string())
    }
}

impl std::fmt::Debug for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an example's `main` returns.
pub type Outcome = Result<(), Failure>;
