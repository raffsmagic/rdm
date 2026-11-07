# The examples

Every notebook cell of the site and every code
block of the post [Recursive Decision Models](https://raw.works/recursive-decision-models/), as a
Rust program. Each is a port of one Python cell, to be read beside it: the same signature, the
same tools, the same inputs, and the same lines printed (`hierarchy` prints more, as the post
does: see the end of this page).

```sh
cargo run -p rdm --example act                              # the cell as it is written
cargo run -p rdm --example act -- "Please shut the door."   # another request
cargo run -p rdm --example act -- --task 3                  # a preset of the site's form, by number
cargo run -p rdm --example act -- --help                    # the inputs, their presets, the models
RDM_TRACE=1 cargo run -p rdm --example zen                  # and every request of the run, as a tree
```

## Which models answer

An example asks a decision model, and some ask a language model too. Both are chosen by the
environment. No key is ever written on a command line, in a file here, or printed: a key is
read from the variable that holds it, by the client, when it asks.

| to run it | set |
|---|---|
| with stand-ins, and no network | nothing, if no key is in the environment. Or `RDM_DECISION_MODEL=standin RDM_LANGUAGE_MODEL=standin` |
| against hosted Jev | `RDM_DECISION_MODEL=typesafe`, with the key in `TYPESAFE_API_KEY` (`typesafe:jev-latest` names the model) |
| against OpenRouter | `RDM_DECISION_MODEL=openrouter`, with the key in `OPENROUTER_API_KEY`. The language model is then `openrouter:typesafe/jev-router`, as in the site's hosted mode |
| against a server on this machine | `RDM_DECISION_MODEL=local` for `http://127.0.0.1:8080`, where `llama-server -hf ggml-org/Laya-GGUF` listens, or the server's URL |
| through a gateway | `RDM_DECISION_MODEL=gateway:backends.json`, a [one-system](../../one-system) registry file. Its backends' keys are read from the variables the file names |
| through a gateway that is served | its URL, with `RDM_DECISION_MODEL_NAME` the gateway's name and `RDM_DECISION_MODEL_KEY` the name of the variable its key is in |
| with no decision model at all | `RDM_DECISION_MODEL=lm`: the language model answers the decision questions. It has to be a real one: a stand-in for a language model cannot decide |

`RDM_LANGUAGE_MODEL` is `standin`, an lm15 model string (`openrouter:typesafe/jev-router`,
`anthropic:claude-sonnet-5-5`), or the URL of a server on this machine that speaks Chat
Completions, `/v1` included, serving the model `RDM_LANGUAGE_MODEL_NAME` names.
`RDM_LANGUAGE_MODEL_MAX_TOKENS` is the most tokens one reply may be. Unset, the language model
taken by default (`openrouter:typesafe/jev-router`, when only `OPENROUTER_API_KEY` is set) is
asked for at most 512, as the site asks it, and a model that was named is asked with no such limit.

With `RDM_DECISION_MODEL` unset, it is hosted Jev if `TYPESAFE_API_KEY` is set, else OpenRouter
if `OPENROUTER_API_KEY` is set, else the stand-in, and one line on stderr says which. A
stand-in is not a model: it answers by a few rules written at the foot of its example, enough
for the cell's presets, so that the example prints what the cell documents with nothing asked
of anyone. Ask it something else and it mostly says no.

## As WebAssembly

Each example is built for WASI as it is, and run by wasmtime (`.cargo/config.toml`):

```sh
cargo run -p rdm --example zen --target wasm32-wasip1                 # in a module, on its stand-ins
RDM_DECISION_MODEL=http://127.0.0.1:8080 cargo run -p rdm --example act --target wasm32-wasip2
RDM_EXAMPLES_TARGET=wasm32-wasip1 cargo test -p rdm --test examples   # every one, beside its transcripts
```

A module is given the directory it is run in and the environment. One built for
`wasm32-wasip2` reaches its models through the host's HTTP, by any of the settings above.
One built for `wasm32-wasip1` has no network, and says so when it is asked to reach one:
there the models are the stand-ins.

## The examples

| example | the cell, and the section of the post | what it shows |
|---|---|---|
| `triage` | notebook, "One request" | one request; an output named after a tool calls it; its argument is another answer |
| `act` | smart home, "One request" | the same, on the house: one decision points at a tool and chooses its arguments |
| `zen` | notebook, "Recursion" | recursion: a yes about a text asks the same question of each half |
| `outline` | notebook, "The shape of an RLM"; post, "Self-similarity and external state" | the shape of an RLM with no language model; `state` |
| `handbook` | notebook, "With a language model"; post, "A Simple RDM" | the halving search, and at one line a language model writes the answer |
| `choose` | notebook, "Choosing the next question" | a sub-call with another signature: each answer decides the next question's options |
| `filter` | notebook, "Sorting and filtering" | `Filter`: the two likeliest at every level |
| `hierarchy` | its own tab, "Hierarchical search"; post, "Hierarchical search, without the beam" | Shopify's taxonomy, following every category at least half as likely as the likeliest |
| `fit` | notebook, "Any size" | `Fit`: inputs of any size |
| `map` | smart home, "The map"; post, "Communicating with Maps" | the pointer and its loop; the map changes: what worked is kept as a new name |
| `near` | smart home, "A map with labels" | `Near`: a map with labels |
| `dimensions` | notebook, "A map with dimensions" | `Near`: a map with dimensions, embeddings you can read |
| `route` | smart home, "With a language model"; post, the smart home with an LLM | a language model is a tool like any other; another module as a tool |
| `repl` | smart home, "A REPL" | a language model writes a function into the REPL and it is on the map |
| `write_map` | smart home, "Making a starter map" | a map written once and saved as JSON |
| `starter` | smart home, "A starter map" | the saved map, loaded and pointed at |
| `home` | smart home, "The house" | the house: six rooms and a thermostat, state kept between requests |

All seventeen are ported: `cargo test -p rdm --test yet_to_come` fails if a name of the table
has no program, or no record of what it prints. The Python of each is the cell of the same name
on [recursivedecisionmodels.com](https://recursivedecisionmodels.com).

One example is of no cell: `rlm` is `repl` with a writer that is an RLM, a language model in
a loop of its own. It defines the function in a step, calls it there to try it, and submits
it; the house is put back as it was when the writing ends, and the function is on the map
only once it has been tried and has passed the example's own check. The Python has no such
cell: its writer is a `dspy.RLM` with the house's actions taken away, so that what it writes
cannot be tried. `RDM_TRACE=1` prints every step the writer took.

In `assets/`: `handbook.md`, the 2,555 characters the cells search; `starter_map.json`, the map
the site ships; and `categories.txt`, Shopify's product taxonomy (version 2026-02, under the
licence beside it), which the `hierarchy` cell fetches and this keeps, so no run needs the
network for it.

The post writes three of its searches a little differently from the cells that came of them.
`RDM_POST=1` runs such an example as the post writes it, and it prints what the post says it prints:

```sh
RDM_POST=1 cargo run -p rdm --example handbook                  # "A Simple RDM": the whole handbook, halved down to a line, in 15 requests
RDM_POST=1 cargo run -p rdm --example outline -- --question 2   # "Self-similarity and external state": 15 again, the largest showing 272 characters
RDM_POST=1 cargo run -p rdm --example hierarchy -- --listing 4  # "Hierarchical search, without the beam": two functions around one request
```

So it is with the map. The post's, in "Communicating with Maps", is a loop written out by hand
around one request, for a house with a TV and a light:

```sh
RDM_POST=1 cargo run -p rdm --example map   # tv on, lights off; then the line kept as "movie time", found by "Let's watch a film."
```

The house of `home` keeps its state from one request to the next, so one run takes several in
turn, each put to the house as the one before left it:

```sh
cargo run -p rdm --example home -- "We are off to bed." "Morning!" "It is too hot in here." "Set the alarm."
cargo run -p rdm --example home -- --task 1 --then 2 --then 10 --then 11     # the same, by the numbers --help lists
```

## How one is written

One file, `examples/<name>.rs`, in the order of its cell:

```rust
//! The cell's opening comment, reworded only where Rust differs.
//!
//! ```text
//! cargo run -p rdm --example act
//! ```

mod common;

use common::{Form, Say, py};
use rdm::systemone::Question;
use rdm::{Json, Rdm, Signature, Tool, inputs};

fn main() -> common::Outcome {
    let decision_model = common::decision_model(standin)?; // the decision model the environment asks for

    // What the form sets: the request, and where the house starts.
    let form = Form::new("act", "One request", "One decision points at a tool and chooses its arguments.")
        .text("task", "Ask the house", &["Turn the TV on.", "Switch the lights off."])
        .house(serde_json::json!({"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}))
        .read();
    let (task, house) = (form.text_of("task"), form.house_of());

    // ... the tools, the signature and the module, as the cell has them ...

    common::show_house(&house, &task); // what the page draws: printed only with RDM_TRACE=1
    let result = act.call(inputs! { "task" => &task, "house" => &house })?;
    common::trace(&result); // the calls of the run, as the site shows them: only with RDM_TRACE=1
    py::print!(result.observations); // print(result.observations)
    Ok(())
}

/// The stand-in decision model: what each question is answered when no real model is asked.
fn standin(inputs: &Json, name: &str, question: &Question) -> Say {
    match name {
        "switch" => true.into(),                 // a yes, at 0.93 (false: a no, at 0.04; 0.7: that likely)
        "device" => "tv".into(),                 // the option chosen, at 0.95
        _ => vec![("on", 0.6), ("off", 0.4)].into(), // or how likely each option is
    }
}
```

- `py::print!` is Python's `print`: `True`, `['tv on']`, `{'switch': 'tv: now on'}`.
  `py::repr` and `py::str` are the two halves of it.
- The stand-in's rules stay in the example's own file. An example that writes has a second
  function, given to `common::language_model(writes)`: `writes(system, prompt)` returns the JSON
  a model would, and `rdm::lm::input_of(prompt, "task")` reads an input out of the prompt.
- `common/` is shared and is not edited to add an example. Neither is `Cargo.toml`: every
  name in the table has its `[[example]]` entry already.

What an example prints is recorded in `tests/examples/`, and `cargo test -p rdm` runs every
example with the stand-ins and sets what it prints beside the record. A record is a transcript:
its first line is the command and the rest is what the program then prints. `<name>.txt` is the
cell as it is written, and `<name>.<case>.txt` the same program with other inputs.

```text
$ cargo run -p rdm --example act -- --task 3
{'switch': 'door: now closed'}
```

```sh
cargo run -p rdm --example act                                              # run it
cargo test -p rdm --test examples -- --exact act                            # build and test this one alone
RDM_EXAMPLES_RECORD=1 cargo test -p rdm --test examples -- --exact act      # write down what it prints now
```

The second builds only `act`, so it works while other examples are half written. A transcript
is meant to say what the Python cell prints, so what was recorded is to be read, and set beside
the cell's own output or the post's.

`tests/examples/python/cells.py` does that setting beside: it runs the Python cell itself on the
same rules, written once more in `standins/<name>.py`, records the transcripts from what the
cell prints (`cells.py record <name>`), and says whether the Rust example prints the same lines
and sends the same requests, byte for byte (`cells.py same <name>`). It needs the site's cells
and a Python with the dspy fork, and is how the transcripts here were made, but for one.

Fourteen of the seventeen are the same, and `same` says so. Requests that are asked at once
reach a server in any order, so for `fit`, `near` and `dimensions` it says the requests are the
same ones in another order. Three differ, each as it was meant to, and `same` says DIFFERS of them:

- `hierarchy` prints each step of the walk as the post prints it (`All products -> ['Luggage &
  Bags']`), where its cell draws the step on the page with `show_branch` and prints nothing.
  The requests are the same. Its transcripts are therefore the Rust example's own, written with
  `RDM_EXAMPLES_RECORD=1`, and `cells.py record hierarchy` is refused: it would write the
  cell's five lines over them.
- `repl` and `write_map` print the same lines and differ in one request for each function
  written: here the writer is one request to a language model, and what it wrote is the next
  turn's observation, where the cell gives a `dspy.RLM` the `define` function.
