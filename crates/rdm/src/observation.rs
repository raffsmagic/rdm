//! What a tool returned, as the next request shows it.
//!
//! The Python module shows an observation with `json.dumps`, and a model is to be shown the
//! same text from either: so this writes JSON as Python writes it, down to how a float is spelled.

use systemone::Json;

/// What a tool returned, as an input shows it: a text as it is, anything else as JSON, as inputs are shown.
///
/// The JSON is written as Python's `json.dumps` writes it, so a model is shown the same text
/// from either: `, ` and `: ` between things, and a number that is not whole as Python
/// writes a float.
///
/// ```
/// use serde_json::json;
///
/// assert_eq!(rdm::as_observation(&json!("tv: now on")), "tv: now on");
/// assert_eq!(rdm::as_observation(&json!({"found": ["a", "b"], "share": 0.00001})), r#"{"found": ["a", "b"], "share": 1e-05}"#);
/// ```
pub fn as_observation(result: &Json) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
        fn write_f64<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()> {
            writer.write_all(python_float(value).as_bytes())
        }

        fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { writer.write_all(b", ") }
        }

        fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { writer.write_all(b", ") }
        }

        fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
    }
    match result {
        Json::Null => String::new(),
        Json::String(text) => text.clone(),
        other => {
            let mut written = Vec::new();
            let mut serializer = serde_json::Serializer::with_formatter(&mut written, Spaced);
            match serde::Serialize::serialize(other, &mut serializer) {
                Ok(()) => String::from_utf8(written).unwrap_or_else(|_| other.to_string()),
                Err(_) => other.to_string(),
            }
        }
    }
}

/// A number that is not whole as Python writes one: the fewest digits that read back as the
/// same number, with an exponent of at least two digits below `0.0001` and from `1e+16`, and
/// `.0` after a whole one.
fn python_float(value: f64) -> String {
    // The fewest digits, as `d.ddde[-]x`: the same digits Python finds, laid out its way below.
    let written = format!("{value:e}");
    let Some((digits, exponent)) = written.split_once('e') else {
        return written;
    };
    let Ok(exponent) = exponent.parse::<i32>() else { return written };
    let (sign, digits) = match digits.strip_prefix('-') {
        Some(digits) => ("-", digits.replace('.', "")),
        None => ("", digits.replace('.', "")),
    };
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let direction = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{first}{point}{rest}e{direction}{:02}", exponent.unsigned_abs());
    }
    if exponent < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat(exponent.unsigned_abs() as usize - 1));
    }
    let whole = exponent as usize + 1;
    if digits.len() <= whole {
        return format!("{sign}{digits}{}.0", "0".repeat(whole - digits.len()));
    }
    let (before, after) = digits.split_at(whole);
    format!("{sign}{before}.{after}")
}
