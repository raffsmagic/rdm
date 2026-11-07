//! The inputs of a request: plain values, and values that are read each time the request is put.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Map;
use systemone::Json;

use crate::error::{Error, Result};

type Read = dyn Fn() -> std::result::Result<Json, String> + Send + Sync;

#[derive(Clone)]
enum Kept {
    Plain(Json),
    Live(Arc<Read>),
    /// A value that could not be made into JSON, with why. Calling with it is a usage error.
    Unusable(String),
}

/// One input: a plain value, or a live one that is read each time a request shows it.
///
/// A tool that changes the world changes what the next request should show. A plain value is
/// a copy and cannot change, so the world is given as a live input: something read at every
/// turn. [`Shared`] is the usual one.
///
/// It is read when a request is made: at a call, at each turn of a pointer's loop, and once
/// for every sub-call of a batch when the batch is made. What was read then is what the
/// request is, all the way: what the model is shown, what the trajectory records, and what
/// makes the request a repeat of another. (The Python module shows its model the object as it
/// is when the model is asked, which for a later member of a batch can be after its record.)
#[derive(Clone)]
pub struct Input(Kept);

impl Input {
    /// A value that stays as it is.
    pub fn plain(value: impl Into<Json>) -> Self {
        Input(Kept::Plain(value.into()))
    }

    /// A value read each time a request shows it.
    ///
    /// ```
    /// use std::sync::atomic::{AtomicU64, Ordering};
    /// use std::sync::Arc;
    ///
    /// let clock = Arc::new(AtomicU64::new(7));
    /// let shown = clock.clone();
    /// let input = rdm::Input::live(move || shown.load(Ordering::Relaxed).into());
    /// clock.store(8, Ordering::Relaxed);
    /// assert_eq!(input.value(), 8);
    /// ```
    pub fn live(read: impl Fn() -> Json + Send + Sync + 'static) -> Self {
        Input(Kept::Live(Arc::new(move || Ok(read()))))
    }

    /// Whether the value is read each time it is shown.
    pub fn is_live(&self) -> bool {
        matches!(self.0, Kept::Live(_))
    }

    /// The value as it is now. Null for one that cannot be shown.
    pub fn value(&self) -> Json {
        self.read().unwrap_or(Json::Null)
    }

    pub(crate) fn read(&self) -> std::result::Result<Json, String> {
        match &self.0 {
            Kept::Plain(value) => Ok(value.clone()),
            Kept::Live(read) => read(),
            Kept::Unusable(why) => Err(why.clone()),
        }
    }
}

impl fmt::Debug for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.read() {
            Ok(value) if self.is_live() => write!(f, "live {value}"),
            Ok(value) => write!(f, "{value}"),
            Err(why) => write!(f, "unusable ({why})"),
        }
    }
}

/// Anything that is plain data is a plain input.
impl<T: Serialize> From<T> for Input {
    fn from(value: T) -> Self {
        match serde_json::to_value(value) {
            Ok(value) => Input(Kept::Plain(value)),
            Err(error) => Input(Kept::Unusable(error.to_string())),
        }
    }
}

/// A shared value is a live input: each request shows it as it is then.
impl<T: Serialize + Send + 'static> From<Shared<T>> for Input {
    fn from(shared: Shared<T>) -> Self {
        Input(Kept::Live(Arc::new(move || {
            serde_json::to_value(&*shared.lock()).map_err(|error| error.to_string())
        })))
    }
}

/// A shared value is a live input: each request shows it as it is then.
impl<T: Serialize + Send + 'static> From<&Shared<T>> for Input {
    fn from(shared: &Shared<T>) -> Self {
        Input::from(shared.clone())
    }
}

impl From<&Input> for Input {
    fn from(input: &Input) -> Self {
        input.clone()
    }
}

/// A value that tools change and requests show: the house, the board, the counter.
///
/// It is an `Arc<Mutex<T>>`. A tool keeps a clone and changes what is inside; given as an
/// input it is live, so the next request shows it as it is by then, while the trajectory
/// keeps what each request showed.
///
/// ```
/// use rdm::{Shared, inputs};
/// use serde_json::json;
///
/// let house = Shared::new(json!({"tv": "off", "lights": "on"}));
/// let inputs = inputs! { "task" => "Movie time.", "house" => &house };
/// house.update(|house| house["tv"] = json!("on"));
/// assert_eq!(inputs.snapshot().unwrap()["house"], json!({"tv": "on", "lights": "on"}));
/// ```
pub struct Shared<T>(Arc<Mutex<T>>);

impl<T> Shared<T> {
    /// A value to share, as it is to begin with.
    pub fn new(value: T) -> Self {
        Shared(Arc::new(Mutex::new(value)))
    }

    /// The value, held until the guard is dropped. Let it go before a sub-call, or before
    /// taking it again: a request that shows the value has to read it, and would wait for
    /// ever. A tool that panicked while it held the value leaves it as it was at that moment.
    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Change the value, and return whatever the change comes to.
    pub fn update<R>(&self, change: impl FnOnce(&mut T) -> R) -> R {
        change(&mut self.lock())
    }

    /// Put another value in its place, and return the one that was there.
    pub fn set(&self, value: T) -> T {
        std::mem::replace(&mut self.lock(), value)
    }
}

impl<T: Clone> Shared<T> {
    /// A copy of the value as it is now.
    pub fn get(&self) -> T {
        self.lock().clone()
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Shared(self.0.clone())
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Shared::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Shared").field(&*self.lock()).finish()
    }
}

/// The inputs of a call, by name, in the order they were given.
///
/// ```
/// use rdm::{Inputs, inputs};
///
/// let inputs = inputs! { "text" => "Flat is better than nested.", "question" => "Is flat better than nested?" };
/// assert_eq!(inputs.names(), ["text", "question"]);
/// assert_eq!(inputs.value("text").unwrap(), "Flat is better than nested.");
/// assert_eq!(Inputs::new().with("n", 3).value("n").unwrap(), 3);
/// ```
#[derive(Clone, Default)]
pub struct Inputs(IndexMap<String, Input>);

impl Inputs {
    /// No inputs yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The same inputs and one more.
    pub fn with(mut self, name: impl Into<String>, value: impl Into<Input>) -> Self {
        self.set(name, value);
        self
    }

    /// Give an input: in place of the one of that name, or after the others.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<Input>) -> &mut Self {
        self.0.insert(name.into(), value.into());
        self
    }

    /// An input as it was given: a live one is still live when it is passed on to a sub-call.
    pub fn get(&self, name: &str) -> Option<&Input> {
        self.0.get(name)
    }

    /// An input's value as it is now.
    pub fn value(&self, name: &str) -> Option<Json> {
        self.0.get(name).map(Input::value)
    }

    /// Take an input away, and return it if it was there.
    pub fn remove(&mut self, name: &str) -> Option<Input> {
        self.0.shift_remove(name)
    }

    /// Whether an input of that name was given.
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// The names of the inputs, in the order they were given.
    pub fn names(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }

    /// Each input with its name, in the order they were given.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Input)> {
        self.0.iter().map(|(name, input)| (name.as_str(), input))
    }

    /// How many inputs there are.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Every input as it is now: what a request made at this moment shows.
    pub fn snapshot(&self) -> Result<Map<String, Json>> {
        let mut shown = Map::new();
        for (name, input) in &self.0 {
            let value = input
                .read()
                .map_err(|why| Error::Usage(format!("Input '{name}' cannot be shown as JSON: {why}")))?;
            shown.insert(name.clone(), value);
        }
        Ok(shown)
    }
}

impl fmt::Debug for Inputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.0.iter()).finish()
    }
}

/// Plain values, as a fit is given them and gives them back.
impl From<Map<String, Json>> for Inputs {
    fn from(values: Map<String, Json>) -> Self {
        Inputs(values.into_iter().map(|(name, value)| (name, Input(Kept::Plain(value)))).collect())
    }
}

impl<N: Into<String>, V: Into<Input>> FromIterator<(N, V)> for Inputs {
    fn from_iter<I: IntoIterator<Item = (N, V)>>(pairs: I) -> Self {
        Inputs(pairs.into_iter().map(|(name, value)| (name.into(), value.into())).collect())
    }
}

/// The inputs of a call: `inputs! { "text" => zen, "question" => question }`.
///
/// A value is anything that is plain data (it is copied as JSON), or a [`Shared`] or
/// [`Input::live`], which every request shows as it is then.
#[macro_export]
macro_rules! inputs {
    () => { $crate::Inputs::new() };
    ($($name:expr => $value:expr),+ $(,)?) => {{
        let mut inputs = $crate::Inputs::new();
        $( inputs.set($name, $value); )+
        inputs
    }};
}
