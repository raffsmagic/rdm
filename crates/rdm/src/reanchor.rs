//! ReAnchor: the numbers each output is decided with, fitted to what it should have decided.
//!
//! A model answers with probabilities, and a decision is made from them by a few numbers that
//! are never sent to it ([`Params`]): a noul's threshold, a score's cuts, a choice's weights.
//! Their defaults (a threshold of 0.5, cuts halfway between levels, every weight 1) suit a
//! model whose probabilities mean what they say. A smaller model's often do not: its yeses can
//! sit at 0.35 and its noes at 0.2, so that every one of them is a no at 0.5. ReAnchor finds the
//! numbers that decide best for one model, from what it answered and what it should have
//! decided, and asks it nothing.
//!
//! What it is fitted from is [`Evidence`]: one output asked once, with the question, what the
//! model answered, and what should have been decided ([`Want`]). [`gather`] reads it from a
//! run's [`record`](crate::record) events, joined to labels the host keeps by the tag it gave
//! the call ([`Ask::tag`](crate::Ask::tag)). Fitting is arithmetic over those events
//! ([`systemone::decide`]), so the numbers a host holds are a memo of its log: the same events
//! and labels fit the same numbers again, anywhere, with no model at hand.
//!
//! The search is the one of `dspy.experimental.ReAnchor`:
//!
//! - **The candidates come from the evidence.** Any setting between two neighbouring observed
//!   values decides the same, so the candidates are the midpoints of those gaps: between the
//!   probabilities of a yes, between the expected levels of a score, between the points where
//!   an option's pick flips for a weight. Past [`CANDIDATES`] gaps, the values are thinned to
//!   evenly spaced ones.
//! - **A setting stays unless another decides strictly better, and the gain holds across
//!   folds.** The evidence is split into up to [`FOLDS`] parts. For each part a setting is
//!   picked on the others and scored on that part, and a new setting is kept only where those
//!   held-out scores beat the current setting's. A gain that rests on one example is refused.
//! - **Among settings that decide as well, the one in the widest gap wins**, since it leaves
//!   the most room on either side.
//! - **One output at a time, and one number of it at a time.** A choice's weights move one
//!   option at a time while the others hold, and a score's cuts one cut at a time, each kept
//!   between its neighbours.
//!
//! The folds are made from the examples themselves (a hash of each one's identity), not drawn
//! at random, so a fit is the same every time it is made from the same evidence.
//!
//! ```
//! use std::sync::{Arc, Mutex};
//!
//! use rdm::reanchor::{ReAnchor, Want, gather};
//! use rdm::record::Event;
//! use rdm::{Ask, Rdm, Signature, inputs};
//! use systemone::{Answer, Scripted};
//!
//! // A model whose yes for "was the computer asked?" sits at about 0.35, and its no at about 0.2.
//! let shy = Scripted::new(|request| {
//!     let said = request.state["inputs"]["said"].as_str().unwrap_or_default().to_owned();
//!     let p = if said.starts_with("open") { 0.33 + said.len() as f64 / 1000.0 } else { 0.2 };
//!     [("asked", Answer::noul(p))]
//! })
//! .named("edge");
//!
//! let events = Arc::new(Mutex::new(Vec::new()));
//! let kept = events.clone();
//! let ask = Rdm::new(Signature::new("Hear a sentence.").input("said").noul("asked", "Is `inputs.said` put to the computer?"))
//!     .model(shy)
//!     .record(move |event: &Event| {
//!         kept.lock().unwrap().push(event.clone());
//!         Ok(())
//!     });
//!
//! // Ten sentences, each called with a tag the host keeps its label under.
//! let said = ["open files", "open the editor", "open firefox", "open mail", "open the terminal",
//!             "thanks", "see you then", "it rained", "lunch was good", "that was long"];
//! for (i, sentence) in said.iter().enumerate() {
//!     ask.call_with(Ask::new(inputs! { "said" => *sentence }).tag(i)).unwrap();
//!     assert!(!ask.call(inputs! { "said" => *sentence }).unwrap().yes("asked")); // every one a no at 0.5
//! }
//!
//! let gathered = gather(events.lock().unwrap().iter(), |tag, output| {
//!     let i = tag.as_u64()? as usize;
//!     (output == "asked").then(|| Want::Noul(i < 5))
//! });
//! let fitted = ReAnchor::new().fit(&gathered.evidence);
//! assert_eq!(fitted.models, ["edge"]);
//!
//! for (output, params) in &fitted.params {
//!     ask.set_params(output.clone(), params.clone());
//! }
//! assert!(ask.call(inputs! { "said" => "open files" }).unwrap().yes("asked"));
//! assert!(!ask.call(inputs! { "said" => "thanks" }).unwrap().yes("asked"));
//! ```

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use systemone::{Answer, Decision, Json, Kind, Params, Question, decide};

use crate::record::Event;

/// The most candidate settings tried for one number. More distinct values than that are thinned
/// to evenly spaced ones.
pub const CANDIDATES: usize = 40;

/// The smallest and the largest weight a choice's option is tried at.
pub const WEIGHTS: (f64, f64) = (1e-3, 1e3);

/// The most parts the evidence is split into for the fold check.
pub const FOLDS: usize = 5;

/// What one output should have decided.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Want {
    /// A yes (`true`) or a no.
    Noul(bool),
    /// This option, by its label.
    Choice(String),
    /// This level of a score, by its index: the first level is 0.
    Level(usize),
}

/// One output asked once: the question, what the model answered, and what should have been decided.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// What the example is known by: the host's own identity for it. Evidence of one example
    /// is held out together in the fold check.
    pub example: String,
    /// The output's name: the numbers fitted are the numbers of that name.
    pub output: String,
    /// The question as it was asked.
    pub question: Question,
    /// What the model answered it.
    pub answer: Answer,
    /// What should have been decided.
    pub want: Want,
    /// The model it was put to, by the name it was asked as.
    #[serde(default)]
    pub model: String,
    /// The model the server says answered, where it said (a version, often).
    #[serde(default)]
    pub served: String,
}

/// The evidence read from a run's events, and how many outputs could not be read, by why.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Gathered {
    /// One for each output asked that has a label.
    pub evidence: Vec<Evidence>,
    /// What was passed over, counted by the reason.
    pub skipped: IndexMap<String, usize>,
}

/// Read the evidence from the events of runs, with what each output should have decided.
///
/// `want` is given the tag of the run an output was asked in (`began`'s, as the call was
/// given it) and the output's name, and says what should have been decided, or nothing where
/// there is no label. The tag, written as JSON, is the example's identity. The events are of
/// one process: a run's number is unique in it, not past it, so a host that keeps the events
/// of several processes reads each one's apart.
///
/// An `asked` whose request was not kept, that failed, or that left an output unanswered is
/// passed over and counted, as is an output with no label.
pub fn gather<'a>(events: impl IntoIterator<Item = &'a Event>, mut want: impl FnMut(&Json, &str) -> Option<Want>) -> Gathered {
    let mut tags: HashMap<u64, Json> = HashMap::new();
    let mut gathered = Gathered::default();
    let skip = |gathered: &mut Gathered, why: &str, count: usize| {
        *gathered.skipped.entry(why.to_owned()).or_default() += count;
    };
    for event in events {
        match event {
            Event::Began { run, tag, .. } => {
                tags.insert(*run, tag.clone());
            }
            Event::Asked {
                run,
                model,
                request,
                response,
                ..
            } => {
                let Some(request) = request else {
                    skip(&mut gathered, "the request was not kept", 1);
                    continue;
                };
                let Some(response) = response else {
                    skip(&mut gathered, "the request failed", request.questions.len());
                    continue;
                };
                let Some(tag) = tags.get(run) else {
                    skip(&mut gathered, "no `began` for its run", request.questions.len());
                    continue;
                };
                for (output, question) in &request.questions {
                    let Some(answer) = response.answers.get(output) else {
                        skip(&mut gathered, "the output was not answered", 1);
                        continue;
                    };
                    let Some(wanted) = want(tag, output) else {
                        skip(&mut gathered, "no label", 1);
                        continue;
                    };
                    gathered.evidence.push(Evidence {
                        example: tag.to_string(),
                        output: output.clone(),
                        question: question.clone(),
                        answer: answer.clone(),
                        want: wanted,
                        model: model.clone(),
                        served: response.model.clone(),
                    });
                }
            }
            _ => {}
        }
    }
    gathered
}

/// The fit: the numbers to decide each output with, and how each was found.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Fitted {
    /// The models the evidence was asked of, by the name they were asked as. The numbers are
    /// for them: another model's probabilities mean something else.
    pub models: Vec<String>,
    /// The models the servers said answered (versions, often). A change here under the same
    /// name is a model that changed under the numbers.
    pub served: Vec<String>,
    /// The numbers, by output: the ones it started from, with each that was fitted in its place.
    /// An output it started with none for and fitted none for is not here.
    pub params: IndexMap<String, Params>,
    /// How each output was fitted, in the order the evidence first names them.
    pub outputs: Vec<Fitting>,
    /// Fitted together ([`ReAnchor::fit_joint`]): each setting kept, in the order it was kept,
    /// with what it gained. A run of small gains here is a fit leaning on its folds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
    /// Fitted together: the examples left out, counted by why.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub skipped: IndexMap<String, usize>,
}

/// One setting a joint fit kept.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// The pass it was kept in, from 0.
    pub pass: usize,
    /// The output whose numbers changed.
    pub output: String,
    /// Its numbers from then on.
    pub params: Params,
    /// The metric's mean over every example scored, before and after.
    pub before: f64,
    /// See `before`.
    pub after: f64,
    /// How much more the held-out folds scored with it than without, in the metric's units.
    pub held_out_gain: f64,
}

/// How one output was fitted.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Fitting {
    /// The output.
    pub output: String,
    /// The kind of question it was asked as, where it was one kind.
    pub kind: Option<Kind>,
    /// How many times it was asked, with a label, and could be decided.
    pub calls: usize,
    /// How many examples those were in.
    pub examples: usize,
    /// How many differently worded questions were asked under its name. More than one is
    /// numbers fitted to several questions at once: one name, pooled.
    pub questions: usize,
    /// The share decided right with the numbers it started from.
    pub before: f64,
    /// The share decided right with the numbers it ends with.
    pub after: f64,
    /// The numbers it ends with, where they are not the ones it started from.
    pub fitted: Option<Params>,
    /// How many settings were tried.
    pub candidates: usize,
    /// How many times a better setting held across folds and was kept.
    pub kept: usize,
    /// How many times a setting decided better on all the evidence but not across folds, and was refused.
    pub refused: usize,
    /// Why it was not fitted, where it was not.
    pub skipped: Option<String>,
    /// Evidence that could not be decided, and was left out.
    pub invalid: usize,
}

/// The fitting, with the numbers it starts from and how many folds it checks with.
#[derive(Clone, Debug)]
pub struct ReAnchor {
    start: IndexMap<String, Params>,
    folds: usize,
    passes: usize,
    gain: f64,
}

impl Default for ReAnchor {
    fn default() -> Self {
        Self::new()
    }
}

impl ReAnchor {
    /// From the defaults, with up to [`FOLDS`] folds.
    pub fn new() -> Self {
        Self {
            start: IndexMap::new(),
            folds: FOLDS,
            passes: 3,
            gain: 1.0,
        }
    }

    /// Start from these numbers (a module's [`all_params`](crate::Rdm::all_params)), not the
    /// defaults. A number stays unless another decides better.
    pub fn start(mut self, params: IndexMap<String, Params>) -> Self {
        self.start = params;
        self
    }

    /// Split the evidence into up to this many parts for the fold check. Fewer than two
    /// checks nothing: every gain on the evidence is kept.
    pub fn folds(mut self, folds: usize) -> Self {
        self.folds = folds;
        self
    }

    /// Fitted together: go over every output at most this many times (3 unless said), and
    /// stop sooner once a pass keeps nothing.
    pub fn passes(mut self, passes: usize) -> Self {
        self.passes = passes;
        self
    }

    /// Fitted together: keep a setting only where the held-out folds score at least this much
    /// more with it, in the metric's units (1 unless said: one example's worth of a metric
    /// that scores each example 0 or 1). Every setting tried is judged on the same folds, so
    /// a gain smaller than one example is more likely the folds than the setting.
    pub fn gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Fit the numbers of every output the evidence names.
    pub fn fit(&self, evidence: &[Evidence]) -> Fitted {
        let mut by_output: IndexMap<&str, Vec<&Evidence>> = IndexMap::new();
        let mut fitted = Fitted {
            params: self.start.clone(),
            ..Fitted::default()
        };
        for item in evidence {
            by_output.entry(item.output.as_str()).or_default().push(item);
            if !item.model.is_empty() && !fitted.models.contains(&item.model) {
                fitted.models.push(item.model.clone());
            }
            if !item.served.is_empty() && !fitted.served.contains(&item.served) {
                fitted.served.push(item.served.clone());
            }
        }
        for (output, rows) in by_output {
            let start = self.start.get(output).cloned().unwrap_or_default();
            let fitting = self.fit_output(output, &rows, &start);
            if let Some(params) = &fitting.fitted {
                fitted.params.insert(output.to_owned(), params.clone());
            }
            fitted.outputs.push(fitting);
        }
        fitted
    }

    fn fit_output(&self, output: &str, rows: &[&Evidence], start: &Params) -> Fitting {
        let mut fitting = Fitting {
            output: output.to_owned(),
            ..Fitting::default()
        };
        let kind = rows[0].question.kind;
        if rows.iter().any(|row| row.question.kind != kind) {
            fitting.skipped = Some("it was asked as more than one kind of question".to_owned());
            return fitting;
        }
        fitting.kind = Some(kind);
        let asked = rows.len();
        let rows: Vec<&Evidence> = rows
            .iter()
            .copied()
            .filter(|row| wanted_of(kind, &row.want) && decide(output, &row.question, &row.answer, start).is_ok())
            .collect();
        fitting.invalid = asked - rows.len();
        fitting.calls = rows.len();
        if rows.is_empty() {
            fitting.skipped = Some("no evidence could be decided".to_owned());
            return fitting;
        }
        let mut examples: Vec<&str> = rows.iter().map(|row| row.example.as_str()).collect();
        examples.sort_unstable();
        examples.dedup();
        fitting.examples = examples.len();
        let mut questions: Vec<String> = rows.iter().map(|row| row.question.instructions.to_string()).collect();
        questions.sort_unstable();
        questions.dedup();
        fitting.questions = questions.len();

        let base = scores(output, &rows, start);
        fitting.before = mean(&base);
        let folds = folds(&rows, self.folds);
        let search = match kind {
            Kind::Noul => Some(fit_threshold(output, &rows, start, &base, &folds)),
            Kind::Choice => fit_weights(output, &rows, start, &base, &folds),
            Kind::Score => fit_cuts(output, &rows, start, &base, &folds),
        };
        let Some(search) = search else {
            fitting.skipped = Some(match kind {
                Kind::Choice => "its options differ between requests, so no weights fit them all".to_owned(),
                _ => "its levels differ between requests, so no cuts fit them all".to_owned(),
            });
            fitting.after = fitting.before;
            return fitting;
        };
        fitting.candidates = search.candidates;
        fitting.kept = search.kept;
        fitting.refused = search.refused;
        fitting.after = mean(&search.scores);
        if search.params != *start {
            fitting.fitted = Some(search.params);
        }
        fitting
    }
}

/// Whether what was wanted is an answer a question of this kind can have.
fn wanted_of(kind: Kind, want: &Want) -> bool {
    matches!(
        (kind, want),
        (Kind::Noul, Want::Noul(_)) | (Kind::Choice, Want::Choice(_)) | (Kind::Score, Want::Level(_))
    )
}

/// Whether the evidence is decided as it should be with these numbers: 1 or 0. Evidence that
/// cannot be decided with them is decided wrong.
fn right(output: &str, row: &Evidence, params: &Params) -> f64 {
    let decided = match (decide(output, &row.question, &row.answer, params), &row.want) {
        (Ok(Decision::Noul(noul)), Want::Noul(want)) => noul.value == *want,
        (Ok(Decision::Choice(choice)), Want::Choice(want)) => choice.value == *want,
        (Ok(Decision::Score(score)), Want::Level(want)) => score.level == *want,
        _ => false,
    };
    if decided { 1.0 } else { 0.0 }
}

fn scores(output: &str, rows: &[&Evidence], params: &Params) -> Vec<f64> {
    rows.iter().map(|row| right(output, row, params)).collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (mean * 10_000.0).round() / 10_000.0
}

/// A hash of an example's identity that is the same everywhere (FNV-1a), for the folds.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The evidence's indexes split into up to `k` parts, an example's evidence together, the
/// same way every time.
fn folds(rows: &[&Evidence], k: usize) -> Vec<Vec<usize>> {
    parts(&rows.iter().map(|row| row.example.as_str()).collect::<Vec<_>>(), k)
}

/// The indexes of `examples` split into up to `k` parts, each example's together, the same
/// way every time.
fn parts(examples: &[&str], k: usize) -> Vec<Vec<usize>> {
    let mut named: Vec<(u64, &str)> = examples.iter().map(|example| (fnv(example), *example)).collect();
    named.sort_unstable();
    named.dedup();
    let k = k.min(named.len());
    if k < 2 {
        return Vec::new();
    }
    let part: HashMap<&str, usize> = named.iter().enumerate().map(|(i, (_, example))| (*example, i % k)).collect();
    let mut folds = vec![Vec::new(); k];
    for (i, example) in examples.iter().enumerate() {
        folds[part[example]].push(i);
    }
    folds
}

/// A setting tried: how it scored on each piece of evidence, what breaks a tie in its favour
/// (larger first), and the numbers.
struct Tried {
    scores: Vec<f64>,
    key: Vec<f64>,
    params: Params,
}

fn total(values: &[f64], rows: &[usize]) -> f64 {
    rows.iter().map(|&i| values[i]).sum()
}

/// The setting that scores best on `rows`, or none where none scores strictly better than `start` there.
fn pick<'a>(start: &[f64], tried: &'a [Tried], rows: &[usize]) -> Option<&'a Tried> {
    let best = tried.iter().max_by(|a, b| {
        total(&a.scores, rows).total_cmp(&total(&b.scores, rows)).then_with(|| {
            a.key
                .iter()
                .zip(&b.key)
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    })?;
    (total(&best.scores, rows) > total(start, rows)).then_some(best)
}

/// The setting to keep, or none to keep the start, and whether the fold check refused a better one.
fn select<'a>(start: &[f64], tried: &'a [Tried], folds: &[Vec<usize>]) -> (Option<&'a Tried>, bool) {
    let (kept, refused, _) = select_by(start, tried, folds, 0.0);
    (kept, refused)
}

/// As `select`, keeping a setting only where the held-out folds gain at least `gain` with it
/// (any gain at all where `gain` is 0), and saying how much they gained.
fn select_by<'a>(start: &[f64], tried: &'a [Tried], folds: &[Vec<usize>], gain: f64) -> (Option<&'a Tried>, bool, f64) {
    let all: Vec<usize> = (0..start.len()).collect();
    let Some(best) = pick(start, tried, &all) else {
        return (None, false, 0.0);
    };
    if folds.is_empty() {
        let gained = total(&best.scores, &all) - total(start, &all);
        return if gained >= gain {
            (Some(best), false, gained)
        } else {
            (None, true, gained)
        };
    }
    let (mut held, mut held_start) = (0.0, 0.0);
    for fold in folds {
        let rest: Vec<usize> = all.iter().copied().filter(|i| !fold.contains(i)).collect();
        let chosen = pick(start, tried, &rest);
        held += total(chosen.map_or(start, |t| &t.scores), fold);
        held_start += total(start, fold);
    }
    let gained = held - held_start;
    if gained > 0.0 && gained >= gain {
        (Some(best), false, gained)
    } else {
        (None, true, gained)
    }
}

/// The shortest decimal strictly between `lo` and `hi`, rounded from `x`.
fn tidy(x: f64, lo: f64, hi: f64) -> f64 {
    for digits in 1..12 {
        let scale = 10f64.powi(digits);
        let rounded = (x * scale).round() / scale;
        if lo < rounded && rounded < hi {
            return rounded;
        }
    }
    x
}

/// The candidate settings between `lo` and `hi`, and the width of the gap each sits in: the
/// midpoints between neighbouring distinct values. `geometric` takes them on a log scale.
fn gaps(values: &[f64], lo: f64, hi: f64, geometric: bool) -> Vec<(f64, f64)> {
    let mut inner: Vec<f64> = values.iter().copied().filter(|v| v.is_finite() && lo < *v && *v < hi).collect();
    inner.sort_by(f64::total_cmp);
    inner.dedup();
    if inner.len() > CANDIDATES - 1 {
        let step = (inner.len() - 1) as f64 / (CANDIDATES - 2) as f64;
        let mut thinned: Vec<f64> = (0..CANDIDATES - 1).map(|i| inner[(i as f64 * step).round() as usize]).collect();
        thinned.dedup();
        inner = thinned;
    }
    let mut points = vec![lo];
    points.extend(inner);
    points.push(hi);
    points
        .windows(2)
        .filter_map(|pair| {
            let (a, b) = (pair[0], pair[1]);
            let (middle, width) = if geometric {
                ((a * b).sqrt(), (b / a).ln())
            } else {
                ((a + b) / 2.0, b - a)
            };
            (a < middle && middle < b).then(|| (tidy(middle, a, b), width))
        })
        .collect()
}

/// What a search came to.
struct Search {
    params: Params,
    scores: Vec<f64>,
    candidates: usize,
    kept: usize,
    refused: usize,
}

fn fit_threshold(output: &str, rows: &[&Evidence], start: &Params, base: &[f64], folds: &[Vec<usize>]) -> Search {
    let begin = start.threshold.unwrap_or(0.5);
    let probabilities: Vec<f64> = rows.iter().filter_map(|row| row.answer.noul).collect();
    let candidates = threshold_candidates(&probabilities);
    let tried: Vec<Tried> = candidates
        .into_iter()
        .map(|(threshold, width)| {
            let params = Params {
                threshold: Some(threshold),
                ..start.clone()
            };
            Tried {
                scores: scores(output, rows, &params),
                key: vec![width, -(threshold - begin).abs(), threshold],
                params,
            }
        })
        .collect();
    let (kept, refused) = select(base, &tried, folds);
    Search {
        params: kept.map_or_else(|| start.clone(), |t| t.params.clone()),
        scores: kept.map_or_else(|| base.to_vec(), |t| t.scores.clone()),
        candidates: tried.len(),
        kept: usize::from(kept.is_some()),
        refused: usize::from(refused),
    }
}

/// The thresholds to try for a noul answered with these probabilities, and the width of the
/// gap each sits in.
fn threshold_candidates(probabilities: &[f64]) -> Vec<(f64, f64)> {
    let mut candidates = gaps(probabilities, 0.0, 1.0, false);
    let mut distinct = probabilities.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    for pair in distinct.windows(2) {
        let (lower, upper) = (pair[0], pair[1]);
        if lower >= 0.0 && f64::from_bits(lower.to_bits() + 1) == upper {
            // No float lies between them, but a yes at `upper` and over still tells them apart.
            candidates.push((upper, upper - lower));
        }
    }
    if probabilities.contains(&0.0) {
        candidates.push((0.0, 0.0)); // A threshold of 0 is the one that makes a yes of 0.
    }
    candidates
}

/// The options every request offered under this output, in the order the first declared
/// them, if they were the same options each time.
fn labels(rows: &[&Evidence]) -> Option<Vec<String>> {
    let first = rows[0].question.labels();
    let mut sorted = first.clone();
    sorted.sort_unstable();
    rows.iter()
        .all(|row| {
            let mut labels = row.question.labels();
            labels.sort_unstable();
            labels == sorted
        })
        .then_some(first)
}

fn probability(row: &Evidence, label: &str) -> f64 {
    row.answer.probabilities.as_ref().and_then(|p| p.get(label)).copied().unwrap_or(0.0)
}

fn fit_weights(output: &str, rows: &[&Evidence], start: &Params, base: &[f64], folds: &[Vec<usize>]) -> Option<Search> {
    let labels = labels(rows)?;
    let held = start.weights.clone().unwrap_or_default();
    let mut best: IndexMap<String, f64> = labels
        .iter()
        .map(|label| (label.clone(), held.get(label).copied().unwrap_or(1.0)))
        .collect();
    let mut search = Search {
        params: start.clone(),
        scores: base.to_vec(),
        candidates: 0,
        kept: 0,
        refused: 0,
    };
    for label in &labels {
        let flips: Vec<f64> = rows
            .iter()
            .filter_map(|row| {
                let own = probability(row, label);
                let rival = labels
                    .iter()
                    .filter(|other| *other != label)
                    .map(|other| best[other] * probability(row, other))
                    .fold(0.0, f64::max);
                (own > 0.0 && rival > 0.0).then(|| rival / own)
            })
            .collect();
        let (mut low, mut high) = WEIGHTS;
        if let (Some(min), Some(max)) = (flips.iter().copied().reduce(f64::min), flips.iter().copied().reduce(f64::max)) {
            low = low.max(min / 4.0);
            high = high.min(max * 4.0);
        }
        let tried: Vec<Tried> = gaps(&flips, low, high, true)
            .into_iter()
            .filter(|(weight, _)| *weight != best[label])
            .map(|(weight, width)| {
                let mut weights = best.clone();
                weights.insert(label.clone(), weight);
                let params = Params {
                    weights: Some(weights),
                    ..start.clone()
                };
                Tried {
                    scores: scores(output, rows, &params),
                    key: vec![width],
                    params,
                }
            })
            .collect();
        search.candidates += tried.len();
        let (kept, refused) = select(&search.scores, &tried, folds);
        search.refused += usize::from(refused);
        if let Some(kept) = kept {
            search.kept += 1;
            search.scores = kept.scores.clone();
            search.params = kept.params.clone();
            best = kept.params.weights.clone().unwrap_or_default();
        }
    }
    Some(search)
}

/// The expected level of a score's answer: the mean of the level indexes, by probability.
fn expected(row: &Evidence, levels: usize) -> Option<f64> {
    let p: Vec<f64> = (0..levels).map(|level| probability(row, &level.to_string())).collect();
    let total: f64 = p.iter().sum();
    (total > 0.0).then(|| p.iter().enumerate().map(|(level, p)| level as f64 * p).sum::<f64>() / total)
}

fn fit_cuts(output: &str, rows: &[&Evidence], start: &Params, base: &[f64], folds: &[Vec<usize>]) -> Option<Search> {
    let levels = rows[0].question.labels().len();
    if levels < 2 || rows.iter().any(|row| row.question.labels().len() != levels) {
        return None;
    }
    let top = (levels - 1) as f64;
    let means: Vec<f64> = rows.iter().filter_map(|row| expected(row, levels)).collect();
    let mut best: Vec<f64> = start
        .cuts
        .clone()
        .unwrap_or_else(|| (0..levels - 1).map(|level| level as f64 + 0.5).collect());
    let mut search = Search {
        params: start.clone(),
        scores: base.to_vec(),
        candidates: 0,
        kept: 0,
        refused: 0,
    };
    for i in 0..best.len() {
        let below = if i > 0 { best[i - 1] } else { 0.0 };
        let above = best.get(i + 1).copied().unwrap_or(top);
        let tried: Vec<Tried> = gaps(&means, below, above, false)
            .into_iter()
            .filter(|(cut, _)| *cut != best[i])
            .map(|(cut, width)| {
                let mut cuts = best.clone();
                cuts[i] = cut;
                let params = Params {
                    cuts: Some(cuts),
                    ..start.clone()
                };
                Tried {
                    scores: scores(output, rows, &params),
                    key: vec![width],
                    params,
                }
            })
            .collect();
        search.candidates += tried.len();
        let (kept, refused) = select(&search.scores, &tried, folds);
        search.refused += usize::from(refused);
        if let Some(kept) = kept {
            search.kept += 1;
            search.scores = kept.scores.clone();
            search.params = kept.params.clone();
            best = kept.params.cuts.clone().unwrap_or_default();
        }
    }
    Some(search)
}

// Fitted together.

/// The first request of a run as it was asked: what each of its outputs was asked, and what
/// the model answered.
///
/// A joint fit ([`ReAnchor::fit_joint`]) decides every output of it again under candidate
/// numbers and hands the decisions to a metric, so that numbers are fitted to the decision a
/// program makes of all its outputs together, not to each output alone.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Answered {
    /// What the example is known by: the tag of its run, written as JSON.
    pub example: String,
    /// The request's place in its run (`opened.index`).
    pub index: usize,
    /// Each output asked, with its question and the model's answer, in the order asked.
    pub outputs: IndexMap<String, (Question, Answer)>,
    /// The model it was put to, by the name it was asked as.
    #[serde(default)]
    pub model: String,
    /// The model the server says answered, where it said.
    #[serde(default)]
    pub served: String,
}

/// The requests read from a run's events, and what was passed over, counted by why.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Collected {
    /// One for each run's first request that could be read.
    pub answered: Vec<Answered>,
    /// What was passed over, counted by the reason.
    pub skipped: IndexMap<String, usize>,
}

/// Read each run's first request from the events: the one `opened` with no parent, at depth
/// 0, on the first turn. A fit's sub-calls and a tool's are passed over, as is a run whose
/// tag `keep` refuses, a request that failed or was kept without what it asked, and the
/// outputs a request left unanswered (the request itself is kept). An output decided without
/// being asked (a choice cut to one option) is in no `asked`, so it is in no [`Answered`]
/// either: a metric that needs it can say so by scoring the example as nothing.
pub fn gather_answered<'a>(events: impl IntoIterator<Item = &'a Event>, mut keep: impl FnMut(&Json) -> bool) -> Collected {
    let mut tags: HashMap<u64, Json> = HashMap::new();
    let mut first: std::collections::HashSet<(u64, usize)> = std::collections::HashSet::new();
    let mut collected = Collected::default();
    let mut skip = |why: &str| *collected.skipped.entry(why.to_owned()).or_default() += 1;
    let mut answered = Vec::new();
    for event in events {
        match event {
            Event::Began { run, tag, .. } => {
                tags.insert(*run, tag.clone());
            }
            Event::Opened {
                run,
                index,
                parent: None,
                depth: 0,
                iteration: 0,
                ..
            } => {
                first.insert((*run, *index));
            }
            Event::Asked {
                run,
                index,
                model,
                request,
                response,
                ..
            } => {
                if !first.contains(&(*run, *index)) {
                    skip("not the first request of its run");
                    continue;
                }
                let Some(tag) = tags.get(run) else {
                    skip("no `began` for its run");
                    continue;
                };
                if !keep(tag) {
                    skip("left out by the host");
                    continue;
                }
                let (Some(request), Some(response)) = (request, response) else {
                    skip(if request.is_none() {
                        "the request was not kept"
                    } else {
                        "the request failed"
                    });
                    continue;
                };
                let mut outputs = IndexMap::new();
                for (output, question) in &request.questions {
                    match response.answers.get(output) {
                        Some(answer) => {
                            outputs.insert(output.clone(), (question.clone(), answer.clone()));
                        }
                        None => skip("an output was not answered"),
                    }
                }
                answered.push(Answered {
                    example: tag.to_string(),
                    index: *index,
                    outputs,
                    model: model.clone(),
                    served: response.model.clone(),
                });
            }
            _ => {}
        }
    }
    collected.answered = answered;
    collected
}

/// Every output of a request decided with these numbers. One that cannot be decided with them is left out.
pub fn decide_all(answered: &Answered, params: &IndexMap<String, Params>) -> IndexMap<String, Decision> {
    let default = Params::default();
    answered
        .outputs
        .iter()
        .filter_map(|(output, (question, answer))| {
            decide(output, question, answer, params.get(output).unwrap_or(&default))
                .ok()
                .map(|decision| (output.clone(), decision))
        })
        .collect()
}

/// What a metric made of a set of requests under some numbers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    /// The mean over the requests the metric scored.
    pub mean: f64,
    /// How many it scored.
    pub scored: usize,
    /// How many it could not.
    pub unscored: usize,
}

/// Score requests with a metric under these numbers: for a set kept out of the fit, once,
/// after it. The metric is as [`ReAnchor::fit_joint`]'s.
pub fn score_joint(
    answered: &[Answered],
    metric: impl Fn(&str, &IndexMap<String, Decision>, &IndexMap<String, Params>) -> Option<f64>,
    params: &IndexMap<String, Params>,
) -> Scored {
    let values: Vec<Option<f64>> = answered
        .iter()
        .map(|a| metric(&a.example, &decide_all(a, params), params))
        .collect();
    let scored: Vec<f64> = values.iter().flatten().copied().collect();
    Scored {
        mean: mean(&scored),
        scored: scored.len(),
        unscored: values.len() - scored.len(),
    }
}

/// How far a probability is over a threshold, as a decision says it (`systemone::decide`'s
/// confidence): `(p - t) / max(t, 1 - t)`, negative under it.
fn margin(p: f64, t: f64) -> f64 {
    (p - t) / t.max(1.0 - t)
}

/// The thresholds at which `p` stands as far over its own threshold as another yes stands
/// over its: where a rule that takes the surest yes would change its mind.
fn crossings(p: f64, other: f64) -> impl Iterator<Item = f64> {
    let low = (other < 1.0).then(|| (p - other) / (1.0 - other)).filter(|t| *t > 0.0 && *t <= 0.5);
    let high = Some(p / (1.0 + other)).filter(|t| *t > 0.5 && *t < 1.0);
    low.into_iter().chain(high)
}

/// The candidate settings for one output of a joint fit, each as the numbers it would be
/// decided with, and what breaks a tie in its favour.
fn joint_candidates(output: &str, kind: Kind, rows: &[&Answered], params: &IndexMap<String, Params>) -> Vec<(Params, Vec<f64>)> {
    let current = params.get(output).cloned().unwrap_or_default();
    let answers = || {
        rows.iter()
            .filter_map(|row| row.outputs.get(output).map(|(_, answer)| (*row, answer)))
    };
    match kind {
        Kind::Noul => {
            let probabilities: Vec<f64> = answers().filter_map(|(_, answer)| answer.noul).collect();
            let mut candidates = threshold_candidates(&probabilities);
            let mut crossed: Vec<f64> = Vec::new();
            for (row, answer) in answers() {
                let Some(p) = answer.noul else { continue };
                for (other, (question, theirs)) in &row.outputs {
                    let (Some(q), Kind::Noul) = (theirs.noul, question.kind) else {
                        continue;
                    };
                    if other == output {
                        continue;
                    }
                    let t = params.get(other).and_then(|params| params.threshold).unwrap_or(0.5);
                    if q >= t {
                        crossed.extend(crossings(p, margin(q, t)));
                    }
                }
            }
            crossed.sort_by(f64::total_cmp);
            crossed.dedup();
            if crossed.len() > CANDIDATES {
                let step = (crossed.len() - 1) as f64 / (CANDIDATES - 1) as f64;
                crossed = (0..CANDIDATES).map(|i| crossed[(i as f64 * step).round() as usize]).collect();
                crossed.dedup();
            }
            candidates.extend(crossed.into_iter().map(|t| (t, 0.0)));
            let begin = current.threshold.unwrap_or(0.5);
            candidates
                .into_iter()
                .filter(|(t, _)| Some(*t) != current.threshold)
                .map(|(t, width)| {
                    (
                        Params {
                            threshold: Some(t),
                            ..current.clone()
                        },
                        vec![width, -(t - begin).abs(), t],
                    )
                })
                .collect()
        }
        Kind::Choice | Kind::Score => {
            // One number at a time, as a fit of each output alone moves them: a weight for each
            // option, a cut for each level, from the answers this output was given.
            let evidence: Vec<Evidence> = answers()
                .map(|(row, answer)| Evidence {
                    example: row.example.clone(),
                    output: output.to_owned(),
                    question: row.outputs[output].0.clone(),
                    answer: answer.clone(),
                    want: Want::Level(0),
                    model: String::new(),
                    served: String::new(),
                })
                .collect();
            let rows: Vec<&Evidence> = evidence.iter().collect();
            if kind == Kind::Choice {
                let Some(labels) = labels(&rows) else { return Vec::new() };
                let held = current.weights.clone().unwrap_or_default();
                let best: IndexMap<String, f64> = labels.iter().map(|l| (l.clone(), held.get(l).copied().unwrap_or(1.0))).collect();
                let mut out = Vec::new();
                for label in &labels {
                    let flips: Vec<f64> = rows
                        .iter()
                        .filter_map(|row| {
                            let own = probability(row, label);
                            let rival = labels
                                .iter()
                                .filter(|o| *o != label)
                                .map(|o| best[o] * probability(row, o))
                                .fold(0.0, f64::max);
                            (own > 0.0 && rival > 0.0).then(|| rival / own)
                        })
                        .collect();
                    let (mut low, mut high) = WEIGHTS;
                    if let (Some(min), Some(max)) = (flips.iter().copied().reduce(f64::min), flips.iter().copied().reduce(f64::max)) {
                        low = low.max(min / 4.0);
                        high = high.min(max * 4.0);
                    }
                    for (weight, width) in gaps(&flips, low, high, true) {
                        if weight != best[label] {
                            let mut weights = best.clone();
                            weights.insert(label.clone(), weight);
                            out.push((
                                Params {
                                    weights: Some(weights),
                                    ..current.clone()
                                },
                                vec![width],
                            ));
                        }
                    }
                }
                out
            } else {
                let levels = rows[0].question.labels().len();
                if levels < 2 || rows.iter().any(|row| row.question.labels().len() != levels) {
                    return Vec::new();
                }
                let top = (levels - 1) as f64;
                let means: Vec<f64> = rows.iter().filter_map(|row| expected(row, levels)).collect();
                let best: Vec<f64> = current
                    .cuts
                    .clone()
                    .unwrap_or_else(|| (0..levels - 1).map(|l| l as f64 + 0.5).collect());
                let mut out = Vec::new();
                for i in 0..best.len() {
                    let below = if i > 0 { best[i - 1] } else { 0.0 };
                    let above = best.get(i + 1).copied().unwrap_or(top);
                    for (cut, width) in gaps(&means, below, above, false) {
                        if cut != best[i] {
                            let mut cuts = best.clone();
                            cuts[i] = cut;
                            out.push((
                                Params {
                                    cuts: Some(cuts),
                                    ..current.clone()
                                },
                                vec![width],
                            ));
                        }
                    }
                }
                out
            }
        }
    }
}

impl ReAnchor {
    /// Fit the numbers of every output of the requests together, against the decision a
    /// program makes of them all.
    ///
    /// `metric` is given an example's identity, every output of its request decided with the
    /// numbers being tried, and those numbers, and says how right that is (1 for right and 0
    /// for wrong, as a rule), or nothing where the example cannot be scored from what was asked
    /// (an output it needs was not). It must not panic: in a WebAssembly module that ends the
    /// module. Better than writing the metric again is handing it the very function a program
    /// decides with, so that the rule fitted is the rule that runs.
    ///
    /// The outputs are fitted in turn, one number at a time, as [`ReAnchor::fit`] fits them,
    /// with more candidates for a threshold: the points where its yes stands as far over its
    /// own threshold as another output's yes stands over its, which is where a rule that takes
    /// the surest yes changes its mind. A setting is kept only where it scores better on all of
    /// the examples and the held-out folds gain at least [`ReAnchor::gain`] with it. Every
    /// output is gone over again until a pass keeps nothing, at most [`ReAnchor::passes`] times.
    ///
    /// ```
    /// use indexmap::IndexMap;
    /// use rdm::reanchor::{Answered, ReAnchor};
    /// use systemone::{Answer, Decision, Params, Question};
    ///
    /// // A shy model: "asked" sits at 0.35 for a request and 0.2 otherwise.
    /// let answered: Vec<Answered> = (0..10)
    ///     .map(|i| Answered {
    ///         example: i.to_string(),
    ///         outputs: IndexMap::from([(
    ///             "asked".to_owned(),
    ///             (Question::noul("Is it put to the computer?"), Answer::noul(if i < 5 { 0.35 } else { 0.2 })),
    ///         )]),
    ///         ..Answered::default()
    ///     })
    ///     .collect();
    /// // The program's decision: a request for the first five, and nothing for the rest.
    /// let metric = |example: &str, decided: &IndexMap<String, Decision>, _: &IndexMap<String, Params>| {
    ///     let asked = decided.get("asked")?.as_noul()?.value;
    ///     Some(if asked == (example.parse::<usize>().ok()? < 5) { 1.0 } else { 0.0 })
    /// };
    /// let fitted = ReAnchor::new().fit_joint(&answered, metric);
    /// let threshold = fitted.params["asked"].threshold.unwrap();
    /// assert!(0.2 < threshold && threshold <= 0.35);
    /// assert_eq!((fitted.steps[0].before, fitted.steps[0].after), (0.5, 1.0));
    /// ```
    pub fn fit_joint(
        &self,
        answered: &[Answered],
        metric: impl Fn(&str, &IndexMap<String, Decision>, &IndexMap<String, Params>) -> Option<f64>,
    ) -> Fitted {
        let mut fitted = Fitted::default();
        for item in answered {
            if !item.model.is_empty() && !fitted.models.contains(&item.model) {
                fitted.models.push(item.model.clone());
            }
            if !item.served.is_empty() && !fitted.served.contains(&item.served) {
                fitted.served.push(item.served.clone());
            }
        }
        let mut params = self.start.clone();
        let score = |rows: &[&Answered], params: &IndexMap<String, Params>| -> Vec<Option<f64>> {
            rows.iter()
                .map(|row| metric(&row.example, &decide_all(row, params), params))
                .collect()
        };
        let all: Vec<&Answered> = answered.iter().collect();
        let first = score(&all, &params);
        let rows: Vec<&Answered> = all.iter().zip(&first).filter(|(_, s)| s.is_some()).map(|(row, _)| *row).collect();
        let unscored = all.len() - rows.len();
        if unscored > 0 {
            fitted.skipped.insert("the metric could not score it".to_owned(), unscored);
        }
        let mut current: Vec<f64> = first.into_iter().flatten().collect();
        let folds = parts(&rows.iter().map(|row| row.example.as_str()).collect::<Vec<_>>(), self.folds);

        let mut kinds: IndexMap<String, Option<Kind>> = IndexMap::new();
        for row in &rows {
            for (output, (question, _)) in &row.outputs {
                let seen = kinds.entry(output.clone()).or_insert(Some(question.kind));
                if *seen != Some(question.kind) {
                    *seen = None;
                }
            }
        }
        let mut fittings: IndexMap<String, Fitting> = kinds
            .iter()
            .map(|(output, kind)| {
                let asked: Vec<&Answered> = rows.iter().copied().filter(|row| row.outputs.contains_key(output)).collect();
                let mut examples: Vec<&str> = asked.iter().map(|row| row.example.as_str()).collect();
                examples.sort_unstable();
                examples.dedup();
                let mut questions: Vec<String> = asked.iter().map(|row| row.outputs[output].0.instructions.to_string()).collect();
                questions.sort_unstable();
                questions.dedup();
                let fitting = Fitting {
                    output: output.clone(),
                    kind: *kind,
                    calls: asked.len(),
                    examples: examples.len(),
                    questions: questions.len(),
                    before: mean(&current),
                    after: mean(&current),
                    skipped: kind.is_none().then(|| "it was asked as more than one kind of question".to_owned()),
                    ..Fitting::default()
                };
                (output.clone(), fitting)
            })
            .collect();

        for pass in 0..self.passes {
            let mut kept_any = false;
            for (output, kind) in &kinds {
                let Some(kind) = kind else { continue };
                let tried: Vec<Tried> = joint_candidates(output, *kind, &rows, &params)
                    .into_iter()
                    .map(|(candidate, key)| {
                        let mut trial = params.clone();
                        trial.insert(output.clone(), candidate.clone());
                        Tried {
                            scores: score(&rows, &trial).into_iter().map(|s| s.unwrap_or(0.0)).collect(),
                            key,
                            params: candidate,
                        }
                    })
                    .collect();
                let fitting = &mut fittings[output.as_str()];
                fitting.candidates += tried.len();
                let (kept, refused, gained) = select_by(&current, &tried, &folds, self.gain);
                fitting.refused += usize::from(refused);
                if let Some(kept) = kept {
                    let before = mean(&current);
                    current = kept.scores.clone();
                    params.insert(output.clone(), kept.params.clone());
                    fitting.kept += 1;
                    fitted.steps.push(Step {
                        pass,
                        output: output.clone(),
                        params: kept.params.clone(),
                        before,
                        after: mean(&current),
                        held_out_gain: gained,
                    });
                    kept_any = true;
                }
            }
            if !kept_any {
                break;
            }
        }
        for (output, fitting) in &mut fittings {
            fitting.after = mean(&current);
            if params.get(output) != self.start.get(output) {
                fitting.fitted = params.get(output).cloned();
            }
        }
        fitted.outputs = fittings.into_values().collect();
        fitted.params = params;
        fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noul(example: usize, p: f64, want: bool) -> Evidence {
        Evidence {
            example: example.to_string(),
            output: "asked".into(),
            question: Question::noul("Is `inputs.said` put to the computer?"),
            answer: Answer::noul(p),
            want: Want::Noul(want),
            model: "edge".into(),
            served: String::new(),
        }
    }

    #[test]
    fn the_candidates_are_the_midpoints_between_the_values_seen() {
        assert_eq!(gaps(&[0.2, 0.4], 0.0, 1.0, false), [(0.1, 0.2), (0.3, 0.2), (0.7, 0.6)]);
        assert!(gaps(&(0..500).map(|i| i as f64 / 1000.0).collect::<Vec<_>>(), 0.0, 1.0, false).len() <= CANDIDATES);
    }

    #[test]
    fn a_shy_model_is_given_the_threshold_between_its_yeses_and_its_noes() {
        let evidence: Vec<Evidence> = (0..10)
            .map(|i| {
                if i < 5 {
                    noul(i, 0.33 + i as f64 / 100.0, true)
                } else {
                    noul(i, 0.15 + i as f64 / 100.0, false)
                }
            })
            .collect();
        let fitted = ReAnchor::new().fit(&evidence);
        let fitting = &fitted.outputs[0];
        assert_eq!((fitting.before, fitting.after), (0.5, 1.0));
        let threshold = fitted.params["asked"].threshold.unwrap();
        assert!(0.24 < threshold && threshold < 0.33, "{threshold}");
        assert_eq!(fitted.models, ["edge"]);
    }

    #[test]
    fn a_gain_that_rests_on_one_example_is_refused() {
        // Right at 0.5 but for one, which only a threshold no other example backs would fix.
        let mut evidence: Vec<Evidence> = (0..8).map(|i| noul(i, if i < 4 { 0.9 } else { 0.1 }, i < 4)).collect();
        evidence.push(noul(8, 0.45, true));
        let fitted = ReAnchor::new().fit(&evidence);
        assert_eq!(fitted.outputs[0].fitted, None);
        assert_eq!(fitted.outputs[0].refused, 1);
        assert!(fitted.params.is_empty());
    }

    #[test]
    fn an_option_picked_too_often_is_weighted_down() {
        let options = [("a", Some("The first.".to_owned())), ("b", Some("The second.".to_owned()))];
        let evidence: Vec<Evidence> = (0..10)
            .map(|i| Evidence {
                example: i.to_string(),
                output: "which".into(),
                question: Question::choice("Which is it?", options.clone()),
                // "a" leads by a little where "b" is wanted, and by a lot where "a" is.
                answer: if i < 5 {
                    Answer::choice([("a", 0.55), ("b", 0.45)])
                } else {
                    Answer::choice([("a", 0.9), ("b", 0.1)])
                },
                want: Want::Choice(if i < 5 { "b" } else { "a" }.into()),
                model: "edge".into(),
                served: String::new(),
            })
            .collect();
        let fitted = ReAnchor::new().fit(&evidence);
        assert_eq!(fitted.outputs[0].after, 1.0);
        let weights = fitted.params["which"].weights.clone().unwrap();
        assert!(weights["a"] < 0.45 / 0.55 || weights["b"] > 0.55 / 0.45, "{weights:?}");
    }

    #[test]
    fn options_that_differ_between_requests_are_not_weighted() {
        let ask = |i: usize, options: [&str; 2]| Evidence {
            example: i.to_string(),
            output: "thing".into(),
            question: Question::choice("Which thing?", options.map(|o| (o, None::<String>))),
            answer: Answer::choice([(options[0], 0.6), (options[1], 0.4)]),
            want: Want::Choice(options[1].into()),
            model: String::new(),
            served: String::new(),
        };
        let fitted = ReAnchor::new().fit(&[ask(0, ["Reload", "Bold"]), ask(1, ["Sign in", "Pricing"])]);
        assert!(fitted.outputs[0].skipped.is_some());
        assert!(fitted.params.is_empty());
    }

    #[test]
    fn a_score_s_cuts_move_to_where_its_levels_fall() {
        let evidence: Vec<Evidence> = (0..10)
            .map(|i| Evidence {
                example: i.to_string(),
                output: "urgency".into(),
                question: Question::score("How urgent?", ["low", "high"]),
                // Expected levels of 0.30 to 0.39 are wanted high, those of 0.10 to 0.19 low.
                answer: if i < 5 {
                    Answer::score([0.7 - i as f64 / 100.0, 0.3 + i as f64 / 100.0])
                } else {
                    Answer::score([0.9 - i as f64 / 100.0, 0.1 + i as f64 / 100.0])
                },
                want: Want::Level(usize::from(i < 5)),
                model: String::new(),
                served: String::new(),
            })
            .collect();
        let fitted = ReAnchor::new().fit(&evidence);
        assert_eq!(fitted.outputs[0].after, 1.0);
        let cut = fitted.params["urgency"].cuts.clone().unwrap()[0];
        assert!(0.19 < cut && cut < 0.3, "{cut}");
    }

    #[test]
    fn evidence_that_cannot_be_decided_is_left_out_and_nothing_panics() {
        let mut broken = noul(0, 0.5, true);
        broken.answer = Answer::default();
        let fitted = ReAnchor::new().fit(&[broken]);
        assert_eq!(fitted.outputs[0].calls, 0);
        assert!(fitted.outputs[0].skipped.is_some());
    }

    #[test]
    fn what_cannot_be_read_from_the_events_is_passed_over_and_counted() {
        use std::sync::{Arc, Mutex};

        use crate::{Ask, Rdm, Signature, inputs};

        let events = Arc::new(Mutex::new(Vec::new()));
        let kept = events.clone();
        let module = Rdm::new(Signature::new("Hear it.").input("said").noul("asked", "Is it put to the computer?"))
            .model(systemone::Scripted::new(|_| [("asked", Answer::noul(0.4))]).named("edge"))
            .record(move |event: &Event| {
                kept.lock().unwrap().push(event.clone());
                Ok(())
            });
        for i in 0..3 {
            module.call_with(Ask::new(inputs! { "said" => "open files" }).tag(i)).unwrap();
        }
        let mut events = events.lock().unwrap().clone();
        // The host kept the second run's evidence without what its request showed.
        let mut asked = 0;
        for event in &mut events {
            if let Event::Asked { request, .. } = event {
                asked += 1;
                if asked == 2 {
                    *request = None;
                }
            }
        }
        let gathered = gather(&events, |tag, _| (tag.as_u64()? != 2).then_some(Want::Noul(true)));
        assert_eq!(gathered.evidence.len(), 1);
        assert_eq!(gathered.evidence[0].example, "0");
        assert_eq!(gathered.evidence[0].model, "edge");
        assert_eq!(gathered.skipped["the request was not kept"], 1);
        assert_eq!(gathered.skipped["no label"], 1);
    }

    /// Twelve requests: four ask to find, four to answer, four nothing. `asked` is shy, `find`
    /// is shy, and `answer` says yes to everything.
    fn requests() -> Vec<Answered> {
        (0..12)
            .map(|i| {
                let (asked, find, answer) = match i / 4 {
                    0 => (0.35, 0.3, 0.52),
                    1 => (0.35, 0.1, 0.55),
                    _ => (0.2, 0.1, 0.52),
                };
                let outputs = [("asked", asked), ("find", find), ("answer", answer)]
                    .into_iter()
                    .map(|(name, p)| (name.to_owned(), (Question::noul(format!("Is it {name}?")), Answer::noul(p))))
                    .collect();
                Answered {
                    example: i.to_string(),
                    outputs,
                    model: "edge".into(),
                    ..Answered::default()
                }
            })
            .collect()
    }

    /// The program's rule: put to the computer, and then the kind is the surest yes.
    fn kind(decided: &IndexMap<String, Decision>) -> Option<&'static str> {
        if !decided.get("asked")?.as_noul()?.value {
            return None;
        }
        ["find", "answer"]
            .into_iter()
            .filter_map(|kind| decided.get(kind)?.as_noul().filter(|n| n.value).map(|n| (kind, n.confidence)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(kind, _)| kind)
    }

    fn metric(example: &str, decided: &IndexMap<String, Decision>, _: &IndexMap<String, Params>) -> Option<f64> {
        let want = match example.parse::<usize>().ok()? / 4 {
            0 => Some("find"),
            1 => Some("answer"),
            _ => None,
        };
        Some(if kind(decided) == want { 1.0 } else { 0.0 })
    }

    #[test]
    fn numbers_fitted_together_make_the_program_s_decision_right() {
        let fitted = ReAnchor::new().fit_joint(&requests(), metric);
        let steps = &fitted.steps;
        assert_eq!(steps.first().map(|s| s.before), Some(0.3333));
        assert_eq!(steps.last().map(|s| s.after), Some(1.0));
        assert!(steps.iter().all(|step| step.held_out_gain >= 1.0));
        let asked = fitted.params["asked"].threshold.unwrap();
        let find = fitted.params["find"].threshold.unwrap();
        assert!(0.2 < asked && asked <= 0.35, "{asked}");
        assert!(0.1 < find && find <= 0.3, "{find}");
        assert_eq!(fitted.models, ["edge"]);
        assert_eq!(score_joint(&requests(), metric, &fitted.params).mean, 1.0);
    }

    #[test]
    fn nothing_is_kept_where_nothing_is_gained() {
        let right = |_: &str, _: &IndexMap<String, Decision>, _: &IndexMap<String, Params>| Some(1.0);
        let fitted = ReAnchor::new().fit_joint(&requests(), right);
        assert!(fitted.steps.is_empty());
        assert!(fitted.params.is_empty());
    }

    #[test]
    fn what_the_metric_cannot_score_is_left_out_and_counted() {
        let some = |example: &str, decided: &IndexMap<String, Decision>, params: &IndexMap<String, Params>| {
            (example != "0").then(|| metric(example, decided, params)).flatten()
        };
        let fitted = ReAnchor::new().fit_joint(&requests(), some);
        assert_eq!(fitted.skipped["the metric could not score it"], 1);
        let scored = score_joint(&requests(), some, &IndexMap::new());
        assert_eq!((scored.scored, scored.unscored), (11, 1));
    }

    #[test]
    fn only_the_first_request_of_a_run_is_read() {
        use std::sync::{Arc, Mutex};

        use crate::{Ask, Rdm, Signature, inputs};

        let events = Arc::new(Mutex::new(Vec::new()));
        let kept = events.clone();
        let module = Rdm::new(Signature::new("Hear it.").input("said").noul("asked", "Is it put to the computer?"))
            .model(systemone::Scripted::new(|_| [("asked", Answer::noul(0.4))]).named("edge"))
            .record(move |event: &Event| {
                kept.lock().unwrap().push(event.clone());
                Ok(())
            });
        for i in 0..2 {
            module.call_with(Ask::new(inputs! { "said" => "open files" }).tag(i)).unwrap();
        }
        let mut events = events.lock().unwrap().clone();
        // A request a fit made inside the first run, as a sub-call of its first request.
        let (sub_opened, sub_asked) = events
            .iter()
            .find_map(|event| match event {
                Event::Asked { run, .. } => Some(*run),
                _ => None,
            })
            .map(|run| {
                let asked = events.iter().find(|e| matches!(e, Event::Asked { .. })).cloned().unwrap();
                let Event::Asked {
                    key,
                    model,
                    request,
                    response,
                    error,
                    params,
                    ..
                } = asked
                else {
                    unreachable!()
                };
                (
                    Event::Opened {
                        run,
                        index: 1,
                        parent: Some(0),
                        depth: 1,
                        iteration: 0,
                        tool: None,
                        inputs: None,
                        repeat: false,
                        skipped: None,
                    },
                    Event::Asked {
                        run,
                        index: 1,
                        key,
                        model,
                        request,
                        response,
                        error,
                        params,
                    },
                )
            })
            .unwrap();
        events.push(sub_opened);
        events.push(sub_asked);
        let collected = gather_answered(&events, |tag| tag.as_u64() != Some(1));
        assert_eq!(collected.answered.len(), 1);
        assert_eq!(collected.answered[0].example, "0");
        assert_eq!(collected.answered[0].index, 0);
        assert_eq!(collected.skipped["not the first request of its run"], 1);
        assert_eq!(collected.skipped["left out by the host"], 1);
    }

    #[test]
    fn a_fit_is_the_same_every_time() {
        let evidence: Vec<Evidence> = (0..12)
            .map(|i| noul(i, (i as f64 * 7.0 % 10.0) / 10.0 + 0.01, i % 3 == 0))
            .collect();
        assert_eq!(ReAnchor::new().fit(&evidence), ReAnchor::new().fit(&evidence));
    }
}
