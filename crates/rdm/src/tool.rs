//! Tools: what the answers select.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use serde::Serialize;
use systemone::Json;

use crate::call::Call;
use crate::error::{Error, Result, listed, sorted};
use crate::inputs::Inputs;
use crate::module::Rdm;
use crate::signature::OBSERVATION;

type Does = dyn Fn(&Call) -> Result<Json> + Send + Sync;

/// A name, a description for the model, and a thing for the code to do.
///
/// The model only decides and a tool only does. What a tool returns is its observation: any
/// plain data, which the module does not read. A tool takes the request that selected it, the
/// [`Call`], and reads its arguments from it by name: [`Call::arg`] looks in the request's
/// answers and then in its inputs. Calling the module through the call is a sub-call.
///
/// A tool that fails (`Err("...".into())`, or any error that is not the module's own) is not
/// the end of the run: the failure is its observation, `Execution error in {name}: {error}`.
/// A tool that panics has failed too, and what the panic said is the error: an index out of
/// bounds or a missing key in a tool is an observation, as an exception is in the Python
/// module, and the tools selected beside it are not lost. That takes a build that unwinds;
/// where a panic aborts, it aborts. A panic anywhere under the tool is caught at the tool: in
/// a sub-call it made, or in the model that sub-call asked.
///
/// ```
/// use rdm::{Rdm, Signature, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// let act = Signature::new("Operate a smart home.")
///     .input("task")
///     .noul("switch", "Does `inputs.task` ask for one device to be set to another state?")
///     .choice("device", "Which device does `inputs.task` mean?", ["tv", "lights"])
///     .choice("to", "Which state does `inputs.task` ask for?", ["on", "off"]);
///
/// // Its arguments are other answers of the same request: the model chooses them, it does not write them.
/// let switch = Tool::new("switch", "Set one device to one state.", |call| {
///     let (device, to): (String, String) = (call.arg("device")?, call.arg("to")?);
///     Ok(format!("{device}: now {to}"))
/// });
///
/// let model = Scripted::new(|_| {
///     [
///         ("switch", Answer::noul(0.9)),
///         ("device", Answer::chose("tv", ["tv", "lights"])),
///         ("to", Answer::chose("on", ["on", "off"])),
///     ]
/// });
/// let result = Rdm::new(act).tool(switch).model(model).call(inputs! { "task" => "Turn the TV on." }).unwrap();
/// assert_eq!(result.observations["switch"], "tv: now on");
/// ```
#[derive(Clone)]
pub struct Tool {
    name: Arc<str>,
    description: Arc<str>,
    does: Arc<Does>,
}

impl Tool {
    /// A tool: its name on the map, what the model is told of it, and what it does.
    pub fn new<F, V>(name: impl Into<String>, description: impl Into<String>, does: F) -> Self
    where
        F: Fn(&Call) -> Result<V> + Send + Sync + 'static,
        V: Serialize,
    {
        Self {
            name: name.into().into(),
            description: description.into().into(),
            does: Arc::new(move |call| Ok(serde_json::to_value(does(call)?)?)),
        }
    }

    /// Another module as a tool. Selected, it joins the run as a sub-call: one level deeper,
    /// in the same trajectory, under the run's limits and its own, to a depth counted from
    /// where it was called. Its inputs are taken by name from the request's answers, then its
    /// inputs, and what it returns is its prediction.
    pub fn module(name: impl Into<String>, description: impl Into<String>, module: &Rdm) -> Self {
        let (name, module) = (name.into(), module.clone());
        let known = name.clone();
        Tool::new(name, description, move |call| {
            let signature = module.signature();
            let (mut inputs, mut missing) = (Inputs::new(), Vec::new());
            for field in signature.inputs().keys() {
                if let Some(answer) = call.answer_value(field) {
                    inputs.set(field, answer);
                } else if let Some(input) = call.inputs().get(field) {
                    inputs.set(field, input);
                } else if !(field == OBSERVATION && signature.pointer().is_some()) {
                    missing.push(field.clone());
                }
            }
            if !missing.is_empty() {
                return Err(Error::Usage(format!(
                    "Tool '{known}' takes {}, which name no answer or input of the request. An input of a module is taken by name from {}.",
                    listed(&missing),
                    sorted(call.names())
                )));
            }
            call.on(&module).sub(inputs)
        })
    }

    /// The name the tool has on the map: what an output is named to call it, and what the pointer is offered.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the model is told of the tool, where an output points at it.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The same tool under another name.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into().into();
        self
    }

    /// The same tool with another description.
    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = description.into().into();
        self
    }

    /// Do what the tool does, for a request. What it fails with is returned as it is: it is
    /// the module that makes an observation of it.
    pub fn call(&self, call: &Call) -> Result<Json> {
        // A branch arbitration has stopped begins no tool, however the tool is reached.
        call.go_on()?;
        (self.does)(call)
    }
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

/// A map of tools by name, in the order they were put on it.
///
/// A module has one, and each run works on a copy of its own: a tool put on the run's map
/// ([`Call::tools`]) is on offer from the run's next request, and one put on the module's
/// ([`Rdm::tools`]) from the next run. It is a handle: a clone is the same map.
#[derive(Clone, Default)]
pub struct Tools(Arc<Mutex<IndexMap<String, Tool>>>);

impl Tools {
    pub(crate) fn lock(&self) -> MutexGuard<'_, IndexMap<String, Tool>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A map of its own holding the same tools.
    pub(crate) fn copy(&self) -> Tools {
        Tools(Arc::new(Mutex::new(self.lock().clone())))
    }

    /// The tool of that name, if the map has one.
    pub fn get(&self, name: &str) -> Option<Tool> {
        self.lock().get(name).cloned()
    }

    /// Put a tool on the map under its name. Returns the tool that had the name, if one did.
    pub fn insert(&self, tool: Tool) -> Option<Tool> {
        self.lock().insert(tool.name().to_owned(), tool)
    }

    /// Take a name off the map.
    pub fn remove(&self, name: &str) -> Option<Tool> {
        self.lock().shift_remove(name)
    }

    /// Whether the map has a tool of that name.
    pub fn contains(&self, name: &str) -> bool {
        self.lock().contains_key(name)
    }

    /// The names on the map, in order.
    pub fn names(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// The tools on the map, in order.
    pub fn all(&self) -> Vec<Tool> {
        self.lock().values().cloned().collect()
    }

    /// How many tools are on the map.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the map has no tools.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

impl fmt::Debug for Tools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

/// Something that offers tools of its own, which change while a run goes on.
///
/// A module is given it whole ([`Rdm::source`]). Each request's map holds what it offers
/// then: a tool it puts up is on offer from the next request, and one it takes down is gone.
/// Its tools are never copied onto the module's own map. A run reads it from one request at
/// a time, and puts what it read on the map before it reads again.
///
/// ```
/// use rdm::{Rdm, Shared, Signature, Source, Tool, inputs};
/// use systemone::{Answer, Scripted};
///
/// /// A shelf of names, each a tool that says its name.
/// struct Shelf(Shared<Vec<String>>);
///
/// impl Source for Shelf {
///     fn offers(&self) -> Vec<Tool> {
///         let say = |name: &String| {
///             let said = name.clone();
///             Tool::new(name.clone(), format!("Say {name}."), move |_| Ok(said.clone()))
///         };
///         self.0.lock().iter().map(say).collect()
///     }
/// }
///
/// let names = Shared::new(vec!["hello".to_owned()]);
/// let model = std::sync::Arc::new(Scripted::new(|request| {
///     let labels = request.questions["next"].labels();
///     [("next", Answer::chose("finish", labels.iter().map(String::as_str)))]
/// }));
/// let rdm = Rdm::new(Signature::new("Say things.").input("task").tool("next", "What next?")).source(Shelf(names.clone())).model(model.clone());
/// rdm.call(inputs! { "task" => "t" }).unwrap();
/// names.lock().push("goodbye".to_owned());
/// rdm.call(inputs! { "task" => "t" }).unwrap();
/// let offered: Vec<Vec<String>> = model.requests().iter().map(|request| request.questions["next"].labels()).collect();
/// assert_eq!(offered, [vec!["hello", "finish"], vec!["hello", "goodbye", "finish"]]);
/// ```
pub trait Source: Send + Sync {
    /// The tools on offer now.
    fn offers(&self) -> Vec<Tool>;

    /// What the source is called, where a name it offers is already on the map.
    fn name(&self) -> String {
        let whole = std::any::type_name::<Self>();
        whole.rsplit("::").next().unwrap_or(whole).to_owned()
    }

    /// A source of its own that offers what this one does now, for [`Rdm::copy`]. None, as it
    /// is unless a source says otherwise, has the copy of the module read this same source:
    /// what a run of either puts up, both offer. A source that keeps something of its own, a
    /// sandbox its tools write into say, returns a copy here, so that a copy of the module
    /// does not write into the original's.
    fn copy(&self) -> Option<Arc<dyn Source>> {
        None
    }

    /// The source as itself, for whatever knows what kind of thing it is. None, as it is
    /// unless a source says otherwise. A source that copies itself returns itself here: a
    /// tool that holds a handle to it then finds, among [`Rdm::sources`], the one the module
    /// that is running was given, which in a copy of the module is not the one it holds.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }
}
