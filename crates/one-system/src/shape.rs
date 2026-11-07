//! What a backend can take, and what a request is the size of.
//!
//! A backend declares two things. Its [`Capabilities`] are hard: a request outside them is
//! never sent to it. Its [`Limits`] are soft: they keep a request away from it while another
//! backend will have it, and are forgotten when none will.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use systemone::{Json, Kind, Question};

use crate::letters::is_letter;

/// How many options a choice has, or levels a score: what `criteria` counts to. A noul has none.
pub(crate) fn criteria_count(question: &Question) -> u64 {
    match &question.criteria {
        Some(Json::Object(options)) => options.len() as u64,
        Some(Json::Array(levels)) => levels.len() as u64,
        _ => 0,
    }
}

/// What a backend is able to answer. A request outside this is never sent to it.
///
/// A backend that declares none is taken to answer anything.
///
/// ```
/// use one_system::Capabilities;
/// use systemone::{Kind, Question, Request};
///
/// let laya = Capabilities::all().max_questions(16).min_criteria(2);
/// let request = Request::new("17 + 25 = 42").question("correct", Question::noul("Is the sum right?"));
/// assert!(laya.supports(&request.state, &request.questions));
///
/// let no_nouls = Capabilities::new([Kind::Choice, Kind::Score]);
/// assert!(!no_nouls.supports(&request.state, &request.questions));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// The kinds of question it answers.
    pub question_types: Vec<Kind>,
    /// The most questions one request may ask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_questions: Option<u64>,
    /// The fewest options a choice, or levels a score, may have.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_criteria: Option<u64>,
    /// The most options a choice, or levels a score, may have.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_criteria: Option<u64>,
    /// `false` for a backend whose state must be text. Such a backend is never sent a state
    /// with pictures in it, which is an array: written out as text, it would be base64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_state: Option<bool>,
    /// What it can be shown, by OpenRouter's names: `["text", "image"]` for a backend that
    /// reads pictures ([`systemone::image`]). None is text only, as every backend declared
    /// before pictures is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<String>>,
}

impl Capabilities {
    /// A backend that answers these kinds of question, and has no other bound.
    pub fn new(question_types: impl IntoIterator<Item = Kind>) -> Self {
        Self {
            question_types: question_types.into_iter().collect(),
            max_questions: None,
            min_criteria: None,
            max_criteria: None,
            structured_state: None,
            input_modalities: None,
        }
    }

    /// A backend that answers all three kinds.
    pub fn all() -> Self {
        Self::new([Kind::Choice, Kind::Score, Kind::Noul])
    }

    /// The most questions one request may ask it.
    pub fn max_questions(mut self, count: u64) -> Self {
        self.max_questions = Some(count);
        self
    }

    /// The fewest options a choice, or levels a score, may have.
    pub fn min_criteria(mut self, count: u64) -> Self {
        self.min_criteria = Some(count);
        self
    }

    /// The most options a choice, or levels a score, may have.
    pub fn max_criteria(mut self, count: u64) -> Self {
        self.max_criteria = Some(count);
        self
    }

    /// Whether the state may be an object or a list. `false` is text only.
    pub fn structured_state(mut self, allowed: bool) -> Self {
        self.structured_state = Some(allowed);
        self
    }

    /// What it can be shown, by OpenRouter's names: `text`, `image`.
    ///
    /// ```
    /// use one_system::Capabilities;
    /// use systemone::{Question, Request};
    ///
    /// let picture = Request::new(serde_json::json!([{"inputs": {"window": "<image: window>"}}, systemone::image("image/png", b"png")]))
    ///     .question("editor", Question::noul("Is `inputs.window` an editor?"));
    /// assert!(!Capabilities::all().supports(&picture.state, &picture.questions));
    /// assert!(Capabilities::all().input_modalities(["text", "image"]).supports(&picture.state, &picture.questions));
    /// ```
    pub fn input_modalities<S: Into<String>>(mut self, modalities: impl IntoIterator<Item = S>) -> Self {
        self.input_modalities = Some(modalities.into_iter().map(Into::into).collect());
        self
    }

    /// Whether it can be shown pictures: `image` is among its input modalities.
    pub fn takes_images(&self) -> bool {
        self.input_modalities
            .as_ref()
            .is_some_and(|modalities| modalities.iter().any(|modality| modality == "image"))
    }

    /// Whether a state and these questions are inside what was declared.
    ///
    /// The bounds on criteria are for choices and scores. A noul has no criteria to count. A
    /// state with pictures in it is for a backend that takes images, and that takes a state
    /// that is not text.
    pub fn supports(&self, state: &Json, questions: &IndexMap<String, Question>) -> bool {
        if self.max_questions.is_some_and(|most| questions.len() as u64 > most) {
            return false;
        }
        if self.structured_state == Some(false) && !state.is_string() {
            return false;
        }
        if systemone::images(state) > 0 && !self.takes_images() {
            return false;
        }
        questions.values().all(|question| {
            if !self.question_types.contains(&question.kind) {
                return false;
            }
            if question.kind == Kind::Noul {
                return true;
            }
            let count = criteria_count(question);
            self.min_criteria.is_none_or(|least| count >= least) && self.max_criteria.is_none_or(|most| count <= most)
        })
    }

    /// The rule a declaration breaks, if it breaks one.
    pub(crate) fn fault(&self) -> Option<&'static str> {
        let mut seen = Vec::new();
        for kind in &self.question_types {
            if seen.contains(kind) {
                return Some("capabilities require unique native question types");
            }
            seen.push(*kind);
        }
        if seen.is_empty() {
            return Some("capabilities require nonempty question_types");
        }
        if [self.max_questions, self.min_criteria, self.max_criteria].contains(&Some(0)) {
            return Some("capability limits must be positive integers");
        }
        if let (Some(least), Some(most)) = (self.min_criteria, self.max_criteria)
            && least > most
        {
            return Some("min_criteria must not exceed max_criteria");
        }
        if let Some(modalities) = &self.input_modalities {
            let unique = modalities
                .iter()
                .enumerate()
                .all(|(at, modality)| !modalities[..at].contains(modality));
            if modalities.is_empty() || !unique || modalities.iter().any(String::is_empty) {
                return Some("input_modalities must be unique names, such as text and image");
            }
        }
        None
    }
}

/// What a backend would rather not be sent. Routing hints, and no more: they are not a
/// boundary for privacy or for safety.
///
/// A request over a limit goes to another backend while there is one that will have it.
/// When the limits would leave no backend at all, they are set aside, and every backend
/// that is able to answer is a candidate again.
///
/// ```
/// use one_system::{Limits, Summary};
///
/// let short_english = Limits::default().max_characters(1200).max_non_ascii_letter_fraction(0.02);
/// assert_eq!(short_english.max_characters, Some(1200));
/// assert_eq!(Summary::of(&serde_json::json!("Grüße")).non_ascii_letter_fraction, 0.4);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// The longest state, in characters of its compact JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_characters: Option<u64>,
    /// The largest share of the state's letters that may be outside ASCII, from 0 to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_non_ascii_letter_fraction: Option<f64>,
    /// The most questions in one request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_questions: Option<u64>,
    /// The most options or levels in the widest question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_criteria: Option<u64>,
}

impl Limits {
    /// The longest state it would rather have, in characters of the state's compact JSON.
    pub fn max_characters(mut self, count: u64) -> Self {
        self.max_characters = Some(count);
        self
    }

    /// The largest share of letters outside ASCII it would rather have, from 0 to 1.
    pub fn max_non_ascii_letter_fraction(mut self, fraction: f64) -> Self {
        self.max_non_ascii_letter_fraction = Some(fraction);
        self
    }

    /// The most questions it would rather be asked at once.
    pub fn max_questions(mut self, count: u64) -> Self {
        self.max_questions = Some(count);
        self
    }

    /// The most options or levels it would rather the widest question had. At least 2.
    pub fn max_criteria(mut self, count: u64) -> Self {
        self.max_criteria = Some(count);
        self
    }

    /// Whether a request of this size is one the backend would rather have. Being at a limit
    /// is inside it.
    pub fn allow(&self, summary: &Summary, questions: &IndexMap<String, Question>) -> bool {
        let widest = questions.values().map(criteria_count).max().unwrap_or(0);
        self.max_characters.is_none_or(|most| summary.characters <= most)
            && self
                .max_non_ascii_letter_fraction
                .is_none_or(|most| summary.non_ascii_letter_fraction <= most)
            && self.max_questions.is_none_or(|most| questions.len() as u64 <= most)
            && self.max_criteria.is_none_or(|most| widest <= most)
    }

    pub(crate) fn fault(&self) -> Option<&'static str> {
        if self.max_characters == Some(0) {
            return Some("a backend max_characters must be a positive integer");
        }
        if self
            .max_non_ascii_letter_fraction
            .is_some_and(|fraction| !(0.0..=1.0).contains(&fraction))
        {
            return Some("a backend max_non_ascii_letter_fraction must fall between 0 and 1");
        }
        if self.max_questions == Some(0) {
            return Some("a backend max_questions must be a positive integer");
        }
        if self.max_criteria.is_some_and(|most| most < 2) {
            return Some("a backend max_criteria must be at least 2");
        }
        None
    }
}

/// The size of a state and the script it is written in: what a selector is shown in place
/// of a state it is not trusted with.
///
/// It is measured on the state's compact JSON, so the quotes round a text and the braces of
/// an object are counted, and how a caller spaced its JSON changes nothing. A picture in the
/// state is counted as a picture and not as characters: its base64 is no text to read.
///
/// ```
/// use one_system::Summary;
///
/// let summary = Summary::of(&serde_json::json!({"m": "éa界😀 9"}));
/// assert_eq!((summary.characters, summary.non_ascii_letter_fraction), (14, 0.5));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// How many characters (Unicode code points) the compact JSON of the state has.
    pub characters: u64,
    /// The share of its letters that are outside ASCII, to two places. Zero with no letters.
    pub non_ascii_letter_fraction: f64,
    /// How many pictures it shows ([`systemone::images`]). Not written when there are none.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub images: u64,
}

fn is_zero(count: &u64) -> bool {
    *count == 0
}

impl Summary {
    /// Measure a state.
    ///
    /// A letter is what Unicode puts in its general category L, which is what the Go gateway
    /// counts. A mark that writes a vowel (in Devanagari, Thai, Arabic, Hebrew) is not one, nor
    /// is a combining accent, a digit, or a Roman numeral.
    ///
    /// The state is measured as the value it is. One written with escapes (`"\u4e2d"`) is
    /// measured as the characters they stand for, where the Go gateway measures the escapes
    /// as they were typed.
    ///
    /// ```
    /// use one_system::Summary;
    ///
    /// let state = serde_json::json!([{"inputs": {"window": "<image: window>"}}, systemone::image("image/png", &[0; 3000])]);
    /// let summary = Summary::of(&state);
    /// assert_eq!((summary.characters, summary.images), (41, 1)); // `[{"inputs":{"window":"<image: window>"}}]`
    /// ```
    pub fn of(state: &Json) -> Self {
        let images = systemone::images(state) as u64;
        let text = match state {
            // The state as it reads without its pictures.
            Json::Array(items) if images > 0 => {
                let words: Vec<&Json> = items.iter().filter(|item| !systemone::is_image(item)).collect();
                serde_json::to_string(&words).unwrap_or_default()
            }
            _ => state.to_string(),
        };
        let (mut characters, mut letters, mut foreign) = (0u64, 0u64, 0u64);
        for character in text.chars() {
            characters += 1;
            if is_letter(character) {
                letters += 1;
                if !character.is_ascii() {
                    foreign += 1;
                }
            }
        }
        let fraction = if letters == 0 {
            0.0
        } else {
            (foreign as f64 / letters as f64 * 100.0).round() / 100.0
        };
        Self {
            characters,
            non_ascii_letter_fraction: fraction,
            images,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use systemone::Request;

    #[test]
    fn the_summaries_the_go_gateway_pins() {
        let of = |state: Json| {
            let summary = Summary::of(&state);
            (summary.characters, summary.non_ascii_letter_fraction)
        };
        assert_eq!(of(json!("PRIVATE_STATE_CANARY")), (22, 0.0));
        assert_eq!(of(json!({"m": "éa界😀 9"})), (14, 0.5));
        // An e and a combining accent: the accent and the emoji are not letters.
        assert_eq!(of(json!("e\u{301}😀界")), (6, 0.5));
        assert_eq!(of(json!(format!("é{}", "a".repeat(30)))), (33, 0.03));
        assert_eq!(of(json!([123, "😀"])), (9, 0.0));
    }

    #[test]
    fn a_letter_is_a_letter_and_a_mark_is_not() {
        let of = |state: &str| Summary::of(&json!(state)).non_ascii_letter_fraction;
        // Sixty letters and one Hindi syllable, which is one letter and the sign of its vowel.
        assert_eq!(of(&format!("{} \u{915}\u{93e}", "a".repeat(60))), 0.02);
        // One letter in ASCII, and with it: a Thai consonant and vowel sign, a Hebrew letter
        // and point, an Arabic letter and its fatha, a Korean syllable, a circled letter (a
        // symbol), a Roman numeral (a number). Four letters of the five are outside ASCII.
        assert_eq!(of("a \u{e01}\u{e34} \u{5d1}\u{5b0} \u{628}\u{64e} \u{d55c} \u{24b6} \u{2163}"), 0.8);
        // Marks alone are no letters at all.
        assert_eq!(of("\u{93e}\u{e34}\u{5b0}\u{64e}\u{345}"), 0.0);
    }

    #[test]
    fn hard_capabilities() {
        let request = Request::new(json!({"a": 1}))
            .question("yes", Question::noul("?"))
            .question("pick", Question::choice("?", [("a", None), ("b", None), ("c", None)]));
        let supports = |capabilities: Capabilities| capabilities.supports(&request.state, &request.questions);
        assert!(supports(Capabilities::all()));
        assert!(supports(Capabilities::all().max_questions(2).min_criteria(3).max_criteria(3)));
        assert!(!supports(Capabilities::all().max_questions(1)));
        assert!(!supports(Capabilities::all().min_criteria(4)));
        assert!(!supports(Capabilities::all().max_criteria(2)));
        assert!(!supports(Capabilities::all().structured_state(false)));
        assert!(supports(Capabilities::all().structured_state(true)));
        assert!(!supports(Capabilities::new([Kind::Choice])));
        // A noul has no criteria, and the bounds on criteria pass it by.
        let noul = Request::new("text").question("yes", Question::noul("?"));
        assert!(
            Capabilities::all()
                .min_criteria(2)
                .structured_state(false)
                .supports(&noul.state, &noul.questions)
        );
    }

    #[test]
    fn soft_limits_are_inclusive() {
        let request = Request::new("abcd")
            .question("pick", Question::choice("?", [("a", None), ("b", None)]))
            .question("rate", Question::score("?", ["0", "1", "2"]));
        let summary = Summary::of(&request.state);
        assert_eq!(summary.characters, 6);
        let allow = |limits: Limits| limits.allow(&summary, &request.questions);
        assert!(allow(Limits::default()));
        assert!(allow(Limits::default().max_characters(6).max_questions(2).max_criteria(3)));
        assert!(!allow(Limits::default().max_characters(5)));
        assert!(!allow(Limits::default().max_questions(1)));
        assert!(!allow(Limits::default().max_criteria(2)));
        let foreign = Summary::of(&json!("ñandú"));
        assert!(
            !Limits::default()
                .max_non_ascii_letter_fraction(0.02)
                .allow(&foreign, &request.questions)
        );
    }

    #[test]
    fn declarations_that_break_a_rule() {
        assert!(Capabilities::all().fault().is_none());
        assert!(Capabilities::new([]).fault().is_some());
        assert!(Capabilities::new([Kind::Noul, Kind::Noul]).fault().is_some());
        assert!(Capabilities::all().max_questions(0).fault().is_some());
        assert!(Capabilities::all().min_criteria(3).max_criteria(2).fault().is_some());
        assert!(Limits::default().max_criteria(1).fault().is_some());
        assert!(Limits::default().max_non_ascii_letter_fraction(1.5).fault().is_some());
        assert!(Limits::default().max_characters(0).fault().is_some());
    }
}
