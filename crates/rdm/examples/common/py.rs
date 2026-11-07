//! Printing as Python prints. The cells print values as Python writes them (`True`, `['tv on']`,
//! `{'switch': 'tv: now on'}`), and a port prints the same lines, so here is `str`, `repr` and
//! `print` for anything that can be written as JSON.
//!
//! A list is written as a list and an object as a dict, in the order its names were given. JSON
//! has no tuple, so a tuple is written as a list would be.

use serde::Serialize;
use serde_json::Value as Json;

/// A value as Python's `repr` writes it: a string in quotes, `True`, `None`, a list, a dict.
pub fn repr(value: &impl Serialize) -> String {
    let mut written = String::new();
    write(&json(value), &mut written);
    written
}

/// A value as Python's `str` writes it, which is how `print` shows it: a string as it is, and
/// anything else as its `repr`.
pub fn str(value: &impl Serialize) -> String {
    match json(value) {
        Json::String(text) => text,
        other => repr(&other),
    }
}

/// What `print` writes for these values: each as its `str`, with a space between them.
pub fn line(values: &[String]) -> String {
    values.join(" ")
}

/// Python's `print`: each value as its `str`, a space between them, and the end of the line.
/// With nothing, an empty line.
///
/// ```text
/// print("refund:", result.refund.value)          py::print!("refund:", result.value("refund"));
/// ```
#[allow(unused_macros)]
macro_rules! print {
    () => {
        ::std::println!()
    };
    ($($value:expr),+ $(,)?) => {
        ::std::println!("{}", $crate::common::py::line(&[$($crate::common::py::str(&$value)),+]))
    };
}
#[allow(unused_imports)]
pub(crate) use print;

fn json(value: &impl Serialize) -> Json {
    serde_json::to_value(value).unwrap_or(Json::Null)
}

fn write(value: &Json, written: &mut String) {
    match value {
        Json::Null => written.push_str("None"),
        Json::Bool(true) => written.push_str("True"),
        Json::Bool(false) => written.push_str("False"),
        Json::Number(number) if number.is_f64() => written.push_str(&rdm::as_observation(value)),
        Json::Number(number) => written.push_str(&number.to_string()),
        Json::String(text) => quoted(text, written),
        Json::Array(items) => {
            written.push('[');
            for (at, item) in items.iter().enumerate() {
                if at > 0 {
                    written.push_str(", ");
                }
                write(item, written);
            }
            written.push(']');
        }
        Json::Object(entries) => {
            written.push('{');
            for (at, (name, entry)) in entries.iter().enumerate() {
                if at > 0 {
                    written.push_str(", ");
                }
                quoted(name, written);
                written.push_str(": ");
                write(entry, written);
            }
            written.push('}');
        }
    }
}

/// A string in quotes, as Python writes one: single quotes unless it has one and no double
/// quote, and what cannot be printed escaped.
fn quoted(text: &str, written: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    written.push(quote);
    for character in text.chars() {
        match character {
            '\\' => written.push_str("\\\\"),
            '\n' => written.push_str("\\n"),
            '\r' => written.push_str("\\r"),
            '\t' => written.push_str("\\t"),
            character if character == quote => {
                written.push('\\');
                written.push(character);
            }
            character if printable(character) => written.push(character),
            character => match character as u32 {
                code @ 0..=0xff => written.push_str(&format!("\\x{code:02x}")),
                code @ 0x100..=0xffff => written.push_str(&format!("\\u{code:04x}")),
                code => written.push_str(&format!("\\U{code:08x}")),
            },
        }
    }
    written.push(quote);
}

/// Whether Python would print a character as it is. It escapes what controls, what formats,
/// and every space but the plain one. This knows the ones the standard library can tell, and
/// the formatting characters met in text: enough for what the examples print.
fn printable(character: char) -> bool {
    if character == ' ' {
        return true;
    }
    let formats = matches!(
        character as u32,
        0xad | 0x600..=0x605
            | 0x61c
            | 0x6dd
            | 0x70f
            | 0x180e
            | 0x200b..=0x200f
            | 0x202a..=0x202e
            | 0x2060..=0x2064
            | 0x2066..=0x206f
            | 0xfeff
            | 0xfff9..=0xfffb
            | 0xe000..=0xf8ff
            | 0xe0001
            | 0xe0020..=0xe007f
            | 0xf0000..=0x10ffff
    );
    !(character.is_control() || character.is_whitespace() || formats)
}
