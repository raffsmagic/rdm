//! What the page draws beside a cell, as plain lines: the calls of a run as a tree, the house,
//! the text. A cell's own `print`s are the whole of what an example prints unless it is asked
//! for more with `RDM_TRACE=1` or `--trace`; then these lines are printed too, each after `| `.

use rdm::{Entry, Json, Prediction, Shared};
use serde::Serialize;

use super::py;

/// Whether the run is to be shown: `RDM_TRACE` is set to something other than `0`, or
/// `--trace` is among the arguments.
pub fn tracing() -> bool {
    let asked = std::env::var("RDM_TRACE").is_ok_and(|value| !value.is_empty() && value != "0");
    asked || std::env::args().skip(1).any(|argument| argument == "--trace")
}

fn say(line: impl AsRef<str>) {
    if tracing() {
        println!("| {}", line.as_ref());
    }
}

/// A value as Python writes it, cut to a line's worth, with its length where it was cut.
fn short(value: &Json) -> String {
    const MOST: usize = 72;
    let written = py::repr(value);
    if written.chars().count() <= MOST {
        return written;
    }
    let start: String = written.chars().take(MOST).collect();
    match value {
        Json::String(text) => format!("{start}… ({} characters)", text.chars().count()),
        Json::Array(items) => format!("{start}… ({} items)", items.len()),
        _ => format!("{start}…"),
    }
}

/// The calls of a run, as the site shows them: every request in the order it was made, under
/// the request whose tool made it. Each is what it showed, then each answer with its
/// probability and, where the answer ran a tool, what the tool returned. Printed only when
/// the run is to be shown (see [`tracing`]).
pub fn trace(result: &Prediction) {
    if tracing() {
        for line in calls(result) {
            println!("| {line}");
        }
    }
}

/// The lines [`trace`] prints.
pub fn calls(result: &Prediction) -> Vec<String> {
    let entries = &result.trajectory;
    let count = entries.len();
    let mut lines = vec![format!(
        "{count} request{}, ended by {}",
        if count == 1 { "" } else { "s" },
        result.termination_reason
    )];
    // The requests each one led to, in the order they were made: its tool's sub-calls, and
    // the next turn of its loop.
    let mut under: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut first = Vec::new();
    for (at, entry) in entries.iter().enumerate() {
        match entry.parent.filter(|parent| *parent < at) {
            Some(parent) => under[parent].push(at),
            None => first.push(at),
        }
    }
    // Walked without recursion: a loop of many turns is a chain as long.
    let mut next: Vec<usize> = first.into_iter().rev().collect();
    while let Some(at) = next.pop() {
        request(at, &entries[at], &mut lines);
        next.extend(under[at].iter().rev());
    }
    lines
}

fn request(at: usize, entry: &Entry, lines: &mut Vec<String>) {
    let indent = "  ".repeat(entry.depth);
    let number = at + 1;
    let shown = match &entry.inputs {
        Json::Object(inputs) => {
            let each = inputs.iter().map(|(name, value)| format!("{name}: {}", short(value)));
            each.collect::<Vec<_>>().join(" | ")
        }
        other => short(other),
    };
    let mut notes = String::new();
    if let Some(tool) = entry.tool_name.as_deref().filter(|_| entry.iteration == 0) {
        notes.push_str(&format!(" (asked by {tool})"));
    }
    if entry.iteration > 0 {
        notes.push_str(&format!(" (turn {})", entry.iteration + 1));
    }
    if entry.repeat {
        notes.push_str(" (a repeat: answered as before, and nothing run)");
    }
    if let Some(reason) = entry.skipped {
        notes.push_str(&format!(" (not asked: {reason})"));
    }
    if let Some(tokens) = entry.tokens.filter(|tokens| *tokens > 0) {
        notes.push_str(&format!(" [{tokens} tokens]"));
    }
    lines.push(format!("{indent}{number}. {shown}{notes}"));
    let gap = " ".repeat(number.to_string().len() + 2);
    for (name, answer) in entry.outputs.iter().flatten() {
        let value = entry.value(name);
        let odds = match answer {
            rdm::Decision::Noul(noul) => format!("p = {:.2}", noul.probability),
            rdm::Decision::Choice(choice) => format!("p = {:.2}", choice.probabilities.get(&choice.value).copied().unwrap_or_default()),
            rdm::Decision::Score(score) => format!("level {}", score.level),
        };
        let unasked = entry.unasked.iter().flatten().any(|unasked| unasked == name);
        let odds = if unasked { "its one option, not asked".to_owned() } else { odds };
        let ran = entry.observations.as_ref().and_then(|observations| observations.get(name));
        let ran = ran.map(|returned| format!(" -> {}", short(returned))).unwrap_or_default();
        lines.push(format!("{indent}{gap}{name}: {} ({odds}){ran}", py::repr(&value)));
    }
}

/// Show a text the run is about. The page marks each request on it; here it is a line saying
/// how long it is.
pub fn show_text(text: &str) {
    say(format!("text: {} characters, {} lines", text.chars().count(), text.lines().count()));
}

/// Redraw the house from its state, with a line about what just happened.
pub fn show_house<T: Serialize>(house: &Shared<T>, note: &str) {
    if tracing() {
        let house = py::repr(&*house.lock());
        say(format!("house: {house} ({note})"));
    }
}

/// Draw everything a search could reach: `under(piece)` gives what is under a piece. The page
/// draws it in grey; here it is a line saying how many pieces there are.
pub fn show_tree<P: Into<String>>(root: &str, under: impl Fn(&str) -> Vec<P>) {
    if !tracing() {
        return;
    }
    let (mut pieces, mut next) = (0usize, vec![root.to_owned()]);
    while let Some(piece) = next.pop() {
        pieces += 1;
        if pieces == 100_000 {
            return say("laid out: more than 100,000 pieces the search could reach");
        }
        next.extend(under(&piece).into_iter().map(Into::into));
    }
    say(format!("laid out: {pieces} pieces the search could reach"));
}

/// Draw one step of a walk down a taxonomy: how likely each option was, and which were followed.
pub fn show_branch<L: AsRef<str>, F: AsRef<str>>(odds: &[(L, f64)], likely: &[F]) {
    if !tracing() {
        return;
    }
    let followed = |option: &str| likely.iter().any(|each| each.as_ref() == option);
    let each = odds.iter().map(|(option, probability)| {
        let option = option.as_ref();
        format!("{}{option} {probability:.2}", if followed(option) { "-> " } else { "" })
    });
    say(format!("branch: {}", each.collect::<Vec<_>>().join(", ")));
}

/// Mark where a walk down a taxonomy ended.
pub fn show_leaf(path: &str) {
    say(format!("leaf: {path}"));
}
