# rdm: Recursive Decision Models

A **decision model** reads a state and a few typed questions about it, and answers each with
a decision and a probability: yes or no, one of several options, or a level on a scale. It
never writes text. [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is
one. Laya is a small open one that runs on a laptop.

A **Recursive Decision Model** (RDM) puts a decision model in a loop. Its answers pick what
happens next: a function runs, a language model writes, or the same question is asked again
about a smaller piece of the state. That last one is the recursion. It is the idea of a
Recursive Language Model (RLM), for a model that decides instead of writing.

```text
call(request):                         a request is a state and typed questions
    answers <- predict(request)        asked once, and never again
    for each tool the answers select:  the tools run at once
        observation <- tool(...)       inside a tool, calling the module is a sub-call, one level down
        if an output pointed at it:    the same question again, about the state as it is now
    return answers, observations
```

The idea is written up in the post [Recursive Decision Models](https://raw.works/recursive-decision-models/),
and it can be tried in a browser at [recursivedecisionmodels.com](https://recursivedecisionmodels.com).
This repository is the same thing in Rust, and every example of both runs here.

## Run one in five minutes

You need Rust 1.96 or later ([rustup.rs](https://rustup.rs)) and a C compiler (`cc`; on macOS,
`xcode-select --install`). Then:

```sh
git clone https://github.com/rawwerks/rdm
cd rdm
cargo run -p rdm --example handbook
```

The first build takes a minute or two. No key is needed. With none set, every example is
answered by a **stand-in**: a few fixed rules at the foot of the example's file, so that it
runs with no network and prints the same thing every time. Two lines on stderr say so, and
then it prints:

```text
decision model: jev-stand-in | language model: lm-stand-in

What is the nightly hotel limit in London?
-> ['260 euros a night in London.']
7 requests to the decision model, each a yes or a no
```

The example searches a short company handbook for the answer to a question. It asks one
yes-or-no question of the whole text: is the answer in here? On a yes it splits the text in
half and asks the same question of each half, and so on down to a line. At that line a
language model writes the answer.

## The post's examples

The post walks through three programs, and each is an example here. `RDM_POST=1` runs one as
the post writes it, and it prints what the post shows.

**The halving search** ("A Simple RDM"): the whole handbook, halved down to the passage that
answers the question.

```text
$ RDM_POST=1 cargo run -p rdm --example handbook
['The nightly limit for accommodation is 180 euros in most cities and 260 euros in London, Zurich and New York. Breakfast is included in that limit.']
15 requests, each a yes or a no
```

**The taxonomy** ("Hierarchical search, without the beam"): a product placed among Shopify's
12,378 product categories by walking the tree one level at a time, following every branch at
least half as likely as the likeliest. The taxonomy is kept in the repository, so nothing is
fetched.

```text
$ cargo run -p rdm --example hierarchy
decision model: jev-stand-in
12378 categories
All products -> ['Luggage & Bags']
Luggage & Bags -> ['Backpacks']
Backpacks -> ['Laptop Backpacks']

1 leaf reached
Luggage & Bags > Backpacks > Laptop Backpacks

$ RDM_POST=1 cargo run -p rdm --example hierarchy -- --listing 4
All products -> ['Animals & Pet Supplies']
Animals & Pet Supplies -> ['Pet Supplies']
Pet Supplies -> ['Cat Supplies', 'Pet Beds']
Cat Supplies -> ['Cat Furniture']
Cat Furniture -> ['Cat Window Beds & Perches']
Pet Beds -> ['Hammocks', 'Pet Cots', 'Pillow Beds', 'Radiator Beds']
Animals & Pet Supplies > Pet Supplies > Cat Supplies > Cat Furniture > Cat Window Beds & Perches
```

**The house** ("Communicating with Maps", and the site's smart home): six rooms and a
thermostat, with each request put to the house as the one before left it. The decision model
points at a room or a routine on a *map*, a list of names each with a description. The code
acts, and the same question is asked again until the model says `finish`.

```text
$ cargo run -p rdm --example home -- "We are off to bed." "Morning!" "It is too hot in here."
We are off to bed.
good night: good night: done (living room tv: already off; living room fan: now off; … entry lock: now locked)
finish

Morning!
good morning: good morning: done (entry alarm: already off; entry lock: now unlocked; kitchen lights: now on; kitchen coffee maker: now on)
finish

It is too hot in here.
thermostat: thermostat mode: refused while the bedroom window is open
bedroom: bedroom window: now closed
thermostat: thermostat mode: now cool, and the house is heading for 22 degrees
finish
```

(The first answer is cut short here: the example names every device.) The post's own
smaller house, a TV and a light with the loop written out by hand, is
`RDM_POST=1 cargo run -p rdm --example map`.

Add `RDM_TRACE=1` to any of them to see every request of the run as a tree, and `-- --help`
for the inputs and presets each takes. There are eighteen examples in all, and
[their README](crates/rdm/examples/README.md) lists them all.

## Real models

A stand-in knows only the presets of its example. To ask a real model, set a key, or set
`RDM_DECISION_MODEL`.

**On your own machine, with no key.** Laya is an open decision model (Apache-2.0, a 450 MB
download), served by [llama.cpp](https://github.com/ggml-org/llama.cpp) through its System
One API:

```sh
llama-server -hf ggml-org/Laya-GGUF          # listens on 127.0.0.1:8080
RDM_DECISION_MODEL=local cargo run -p rdm --example home -- "It is too hot in here."
```

**With your own Jev key**, from [TypeSafe](https://typesafe.ai):

```sh
export TYPESAFE_API_KEY=...                  # with this set, the examples ask hosted Jev
cargo run -p rdm --example dimensions
```

| to ask | set |
|---|---|
| the stand-ins, with no network | nothing, if no key is set; or `RDM_DECISION_MODEL=standin` |
| Laya on this machine | `RDM_DECISION_MODEL=local`, or the server's URL |
| hosted Jev | `TYPESAFE_API_KEY` |
| Jev through OpenRouter | `OPENROUTER_API_KEY` |
| an ordinary language model, deciding | `RDM_DECISION_MODEL=lm`, with `RDM_LANGUAGE_MODEL` |

Some examples ask a language model to write as well: the last step of `handbook`, and
`route`, `repl` and `rlm`. `RDM_LANGUAGE_MODEL` picks it: any [lm15](https://lm15.dev) model
string (`anthropic:claude-sonnet-5-5`, with its key in `ANTHROPIC_API_KEY`), or the URL of a
local server that speaks Chat Completions. Unset, it is the stand-in, or Jev's router when
only `OPENROUTER_API_KEY` is set. A key is read only from its variable, and never printed.
[The examples' README](crates/rdm/examples/README.md#which-models-answer) has every setting.

## What is in it

| crate | what it is |
|---|---|
| [`systemone`](crates/systemone) | The System One protocol: a request and a response as they are on the wire, the `DecisionModel` trait every decision model implements, and how a decision is made from the probabilities a model answered. No network. |
| [`one-system`](crates/one-system) | A client for any System One server (hosted Jev, OpenRouter, llama.cpp serving Laya); a gateway that sends each request to one of several of them and is itself a decision model; and an HTTP server for it. A Rust version of the Go gateway [rawwerks/one-system](https://github.com/rawwerks/one-system). |
| [`rdm`](crates/rdm) | The loop: `Signature` (the questions), `Tool` (what an answer can call), `Rdm` (the module), `Call` (a running request, which a tool asks again through), and the trajectory of every request. Beside it, each behind a feature: `Filter`, `Fit`, `Near` (a map with dimensions), `Repl` (a [Monty](https://github.com/pydantic/monty) Python sandbox), `lm` (language models through [lm15](https://lm15.dev)), `Rlm` (a language model in a loop over the sandbox) and `reanchor` (the thresholds a decision is made with, fitted to labels). |
| [`rdm-wasm`](crates/rdm-wasm) | The search under [One module](#one-module), built as a WebAssembly module with nothing under it, and a page of JavaScript that hosts it. See [In WebAssembly](#in-webassembly). |

A decision model is reached only through `systemone::DecisionModel`, and a language model only
through lm15. Everything is synchronous: a tool is a closure that asks the module again, and
recursion reads as recursion.

This is the core of `dspy.experimental.RDM` on the
[rawwerks/dspy fork](https://github.com/rawwerks/dspy/tree/feat/rdm-v0), without DSPy. For the
site's notebook cells, the decision requests it sends are byte for byte the ones the Python
sends, apart from the few [set out below](#what-the-core-keeps-and-what-it-leaves).

## One module

A request is a signature whose outputs are decisions. An output named after a tool calls it on
a yes, and inside a tool, asking the module again is a sub-call, one level down: that is the
recursion. Here a yes about a text asks the same question of each half, down to one line.

```rust
use rdm::systemone::Answer;
use rdm::{Rdm, Scripted, Signature, Tool, inputs};

fn main() -> Result<(), rdm::Error> {
    let search = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");

    // Runs on a yes. Asks the same question about each half, one level down.
    let look = Tool::new("look", "Look in each half of the text.", |call| {
        let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() < 2 {
            return Ok(vec![text]); // one line, and the model said yes to it
        }
        let mut found = Vec::new();
        for half in lines.chunks(lines.len().div_ceil(2)) {
            let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
        }
        Ok(found)
    });

    // A decision model that is a function. In use: one_system::Client::typesafe().
    let model = Scripted::new(|request| {
        let text = request.state["inputs"]["text"].as_str().unwrap_or_default();
        [("look", Answer::noul(if text.contains("nested") { 0.9 } else { 0.1 }))]
    });

    let zen = "Beautiful is better than ugly.\nSimple is better than complex.\nFlat is better than nested.\nSparse is better than dense.";
    let result = Rdm::new(search)
        .tool(look)
        .max_depth(6)
        .model(model)
        .call(inputs! { "text" => zen, "question" => "Is flat better than nested?" })?;

    assert_eq!(result.observed::<Vec<String>>("look").unwrap_or_default(), ["Flat is better than nested."]);
    assert_eq!(result.trajectory.len(), 5); // five requests, each a yes or a no
    Ok(())
}
```

A call returns its own answers, each with its probability, what its tools came to
(`observations`), why it ended (`termination_reason`), and, for the call that started the run,
the trajectory: every request of the run in the order it was made.

## The map

A decision model can only point, so something has to say what it is pointing at. A map is
that: a set of names, each with a description for the model and a thing for the code. An output
declared with `.tool(...)` points at one of the module's tools. The loop is the module's: the
tool pointed at runs, and the same question is asked again about the state as it now is. The
map can change while a program runs, and what worked can be kept as one more name.

`Near` gives the map dimensions. Each name answers the questions of a signature once, and the
answers are its coordinates, every one with a name. Each request is asked the same questions
and is offered only the names nearest it. It is an embedding that can be read:

```text
$ cargo run -p rdm --example dimensions
I got charged twice this month, pls fix asap -> ['refund', 'finish']

money 0.98 access 0.01 fault 0.54 buying 0.71 | I got charged twice this month, pls fix asap
money 0.94 access 0.01 fault 0.08 buying 0.51 | refund
money 0.97 access 0.01 fault 0.04 buying 0.57 | send invoice
money 0.01 access 0.97 fault 0.09 buying 0.02 | reset password
money 0.01 access 0.71 fault 0.33 buying 0.02 | unlock account
money 0.01 access 0.01 fault 0.93 buying 0.02 | file a bug
money 0.02 access 0.02 fault 0.74 buying 0.02 | check status page
money 0.53 access 0.02 fault 0.03 buying 0.97 | quote a plan
money 0.02 access 0.01 fault 0.02 buying 0.58 | book a demo

nearest to {'money': 1.0, 'access': 0.0, 'fault': 0.0, 'buying': 0.5}
  1.00 send invoice
  1.00 refund
  0.82 quote a plan
```

That run was answered by hosted Jev. With no key in the environment the same command prints
the same rows from a stand-in's rules, with a stand-in's numbers: 0.93 for a yes and 0.04 for a
no, so that `refund` and `send invoice` are nearest, at 0.91 each.

## Arbitration

The tools one request selects are what might be done, and they run at once. What is done is
one thing after another. A module given an arbiter (`Rdm::arbiter`) goes from the one to the
other as results come back: it does not wait for every tool, but each time one comes back, or
one still running says what it has so far, it rules on them all as they are then. Which
results does the request come to? Which tools are still worth waiting for, which are to be
let finish unwaited, which are stopped, and which not yet started start now? The call is over
once nothing it still waits for is running, and it returns while what it stopped is still
winding down.

`Arbitrate` makes each ruling a decision: one request to the decision model, answers only, a
sub-call of the request it settles, recorded and replayed as any request is. It asks yes or no
of each tool, and one choice of what happens next: finish, wait, start one more, or say what is
not known. It sees what the request was for and answered, its inputs as they are now, and the
rulings before it. It can be asked before anything starts, to hold back what costs until it is
needed, and it can have a decision model of its own. A module with no arbiter waits for all
and uses all, as before. `crates/rdm/src/arbitrate.rs` says the rest.

## An RLM

A decision model cannot write, so what must be written is written by a language model, as a
tool. `lm::Generate` is one request. `Rlm` is a language model in a loop of its own, in the
sandbox: it writes a step of Python, the step is run, it is shown what the step printed, and
it writes the next, until it submits its outputs with `SUBMIT`. It is `dspy.RLM` on this
crate's REPL. An input is a variable there and is not in the prompt, so a text of any length
costs the model a preview.

```rust
use rdm::inputs;
use rdm::lm::{Lm, ScriptedLm, Type, input_of};
use rdm::rlm::Rlm;
use serde_json::json;

fn main() -> Result<(), rdm::Error> {
    // A language model that is a function: it looks first, and then it counts. In use: Lm::new("anthropic:...").
    let lm = Lm::of(
        ScriptedLm::new(|_system, prompt| {
            let code = match input_of(prompt, "iteration") {
                Some("1/5") => "print(len(text.splitlines()))",
                _ => "SUBMIT(len([line for line in text.splitlines() if 'heat' in line]))",
            };
            json!({"reasoning": "", "code": code}).to_string()
        }),
        "scripted",
    );

    let count = Rlm::new("Count the lines of the text that are about heat.")
        .input("text")
        .output_typed("count", Type::Int)
        .lm(lm)
        .max_iters(5);
    let ran = count.call(inputs! { "text" => "The heat is on.\nThe door is shut.\nNo heat upstairs." })?;

    assert_eq!(ran.outputs["count"], 2);
    assert_eq!(ran.steps[0].output, "3"); // what its first step printed
    Ok(())
}
```

The language model chooses nothing that happens:

- it runs when a decision model selects it (`rlm.tool(...)`), in a scratch REPL that is made
  for the run and gone after it;
- what it submits is a candidate, put to checks that are the caller's (`rlm.check(...)`): what
  a check that fails says is what the model reads next;
- a function it writes (`rlm.writer(&repl)`) is checked as `define` checks it, tried in the
  scratch, and only then defined where it is kept, as a name a decision model points at.

Code that is tried is run, so what the scratch is given decides what trying costs. Give it
stand-ins for the functions that act, or a checkpoint (`rlm.checkpoint(...)`), which is taken
when a run starts and put back when it ends: the example `rlm` cools the house to 18 while it
tries its function, and the house is 21 again when the decision model is asked what is next.

`rlm::deciding(repl, model)` gives a REPL `decide(question, about)` and `choose(question,
options, about)`. A function that was written can then loop, which a decision model cannot,
over what the decision model judges, which code cannot, with no language model asked when it
runs. Run for a request, each question is a sub-call of that request.

## The examples

Every notebook cell of the site and every code block of the post is a program in
[`crates/rdm/examples`](crates/rdm/examples), a port of one Python cell to be read beside it:

```sh
cargo run -p rdm --example triage       # one request; a yes calls the tool of the same name
cargo run -p rdm --example zen          # recursion
cargo run -p rdm --example outline      # the shape of an RLM, with no language model
cargo run -p rdm --example hierarchy    # Shopify's taxonomy, without the beam
cargo run -p rdm --example map          # the pointer and its loop; the map changes
cargo run -p rdm --example dimensions   # a map with dimensions
cargo run -p rdm --example route        # a language model is a tool like any other
cargo run -p rdm --example repl         # a language model writes a function, and it is on the map
cargo run -p rdm --example rlm          # the same, by a language model that tries what it writes
cargo run -p rdm --example home         # the house
RDM_TRACE=1 cargo run -p rdm --example zen      # every request of the run, as a tree
RDM_POST=1 cargo run -p rdm --example handbook  # as the post writes it: "15 requests, each a yes or a no"
```

The others are `act`, `handbook`, `choose`, `filter`, `fit`, `near`, `write_map` and `starter`;
[the examples' README](crates/rdm/examples/README.md) has the table.

A stand-in is a port of the rules the Python uses to test the same cell, so an example prints
what the cell documents with nothing asked of anyone. `cargo test` runs every example that way
and compares what it prints with a transcript in `crates/rdm/tests/examples`. Two more ways to
choose the models, beside those under [Real models](#real-models):

```sh
RDM_DECISION_MODEL=gateway:backends.json cargo run -p rdm --example map    # a one-system registry file
RDM_DECISION_MODEL=lm cargo run -p rdm --example triage                    # no decision model: a language model gives the evidence
```

## What the core keeps, and what it leaves

Kept, from `rdm.py`: one request asked once; outputs named after tools; arguments taken by
name from the answers, then the inputs; the pointer and its loop with `finish`; sub-calls that
join the run; `max_depth` as a leaf rule, `max_iters`, `max_predictor_calls`, `max_tokens`, and
a reason that says what stopped a run when it was not a decision; a request already asked on
the way to it is not asked again (`repeat`); a choice with one option is answered without
asking; a tool that fails is an observation; `state` that tools read and no request shows; and
the `fit` hook, which `Fit` and `Near` are written on.

Left, with DSPy: optimizers but one, demos, adapters, save and load, callbacks, async. A language
model's part is small on purpose: `lm::Generate` completes a signature as JSON and is a tool
like any other, and `lm::LmDecisionModel` lets an ordinary language model give the evidence
where there is no decision model. The optimizer kept is ReAnchor's (`reanchor`): it changes no
instruction and asks no model, only the numbers a decision is made with, so it is a function of
what a run recorded and what it should have decided.

Where Rust and Python part ways, on purpose:

- **Sub-calls are explicit.** A tool is given the `Call` that selected it and asks again
  through it (`call.sub`, `call.batch`). Python finds the running call in a context variable.
- **Inputs as they are now.** Python's loop relies on a tool changing the very dict the request
  holds. Here an input is a plain value or a live one: `Shared<T>` is shown by every request as
  it is then, and the trajectory keeps what each request showed.
- **Arguments are read, not injected.** `call.arg::<String>("device")?` looks in the answers,
  then the inputs. A name that is neither is an error for the caller, not an observation.
- **The writer submits, and does not define.** Python's `Repl` examples give a `dspy.RLM` the
  `define` function. Here a writer is one `lm::Generate` that returns a function as data, with
  a refusal coming back as the next turn's observation; or it is an `Rlm`, a language model in
  a loop of its own, which tries its function in a scratch REPL and submits it. Either way it
  is the caller that defines it, after the caller's checks. An `Rlm` can try what it writes
  where the Python writer cannot: the house's actions were taken away from that one after it
  cooled the house while writing, and here a checkpoint puts the house back.
- **`Fit` measures a part as it is sent**, as JSON, so no request is over its limit. Python
  measures the raw text, and a part can come out longer than was allowed. On the reference
  page's own example this asks 41 requests where Python asks 36.
- **Arbitration is Rust's alone.** `Rdm::arbiter` and `Arbitrate` (see [Arbitration](#arbitration))
  have no counterpart in `dspy.experimental.RDM`: there the tools one request selects run at
  once, and it waits for all of them and uses all of them, as a module here with no arbiter
  does. So no request a ruling asks is one the Python sends.

The requests themselves are the same. For the notebook cells, the Rust examples were run beside
the Python cells on the same rules, and the decision requests they send are byte for byte the
ones `dspy.experimental.RDM` sends, apart from the cases above.

## Using it

The crates are not on crates.io yet. A program depends on them through git:

```toml
[dependencies]
rdm = { git = "https://github.com/rawwerks/rdm" }         # with `default-features = false`, the core alone
one-system = { git = "https://github.com/rawwerks/rdm" }  # for a real decision model: `one_system::Client`
```

The program under "One module" needs the first line and no more.

## In WebAssembly

Every crate builds as WebAssembly, and the loop there is the same loop. A module has less
under it than a program has, and what it has depends on what it was built for:

| target | under the module | what runs in it |
|---|---|---|
| `wasm32-wasip1` | WASI 0.1: files, a clock, the environment | everything but a network: the core, every helper, the sandbox, and the examples on their stand-ins |
| `wasm32-wasip2` | WASI 0.2: those, and the host's HTTP | the same, and a decision model and a language model are asked through `wasi:http` |
| `wasm32-unknown-unknown` | nothing: a browser, a worker, an engine of one's own | the core, `Filter`, `Fit`, `Near` and `lm`, as a module that imports nothing, and whatever network the host gives it |

```sh
cargo test --workspace --target wasm32-wasip1                          # the suite, in a module, run by wasmtime
cargo run -p rdm --example zen --target wasm32-wasip2                   # one example
RDM_DECISION_MODEL=typesafe cargo run -p rdm --example map --target wasm32-wasip2   # hosted Jev, through wasi:http
RDM_EXAMPLES_TARGET=wasm32-wasip1 cargo test -p rdm --test examples     # every example, beside its transcripts
cargo build -p rdm-wasm --target wasm32-unknown-unknown --release       # a module with nothing under it,
node crates/rdm-wasm/host.mjs                                           # and a host for it
```

`mise install` brings the three targets and wasmtime, and `.cargo/config.toml` names wasmtime
as what runs the first two. Both suites pass, and every example prints from a module what its
transcript records.

What a module does differently is what it cannot do at all:

- **One thing at a time.** A module has one thread. The tools one request selects, the
  sub-calls of one batch and the questions `Near` asks together are done one after another, in
  the order they would have been started, and what comes back is the same.
- **One stack.** On a machine a run that goes deep is given another stack every few calls, so
  it meets a limit and not the end of a stack. A module has the stack it was linked with, a
  megabyte unless told otherwise: about 40 calls deep in a debug build and 150 in a release
  one. The call that would not fit is not made, and the tool that asked for it is told so, as
  an observation: `The run is too deep to go on`. `-C link-arg=-zstack-size=8388608` links a
  module with more.
- **A panic ends the module.** Stable Rust cannot unwind in WebAssembly, so a panic in a tool
  is not caught and made an observation as it is elsewhere. A tool that returns an error is
  one, there as anywhere.
- **The network is the host's.** A client asks through a transport
  (`one_system::Transport`): this crate's own HTTP on a machine, `wasi:http` in a module built
  for WASI 0.2, and in any other module whatever the host gives it with `Client::transport`.
  A language model is the same: lm15 is built as the wire alone, and its router is given
  `wasi:http` in a module built for WASI 0.2, and the host's own transport in any other
  (`Lm::with_config`). A model with no network says whose it is, and nothing panics.
- **No clock, with nothing under it.** A retry in `wasm32-unknown-unknown` is sent at once,
  since there is nothing to wait by.
- **No server.** A module can listen on no port: `one_system::Server` and the `one-system`
  binary are for a machine.

Everything here waits for its answers, and a host written in JavaScript answers with a
promise. That is closed by the engine and not by this code. With JavaScript Promise
Integration the host wraps its `fetch` in `WebAssembly.Suspending`, and the module is
suspended while the promise is pending: to the loop, the host's function returned.
[`crates/rdm-wasm`](crates/rdm-wasm) is that, whole: the search under "One module" in a
module of one import, and a page of JavaScript that carries its requests. Where an engine
has no JSPI the same module runs in a worker, with a synchronous `XMLHttpRequest`.

The `repl` feature runs in a module built for WASI as it is. In `wasm32-unknown-unknown`
Monty reads its clock and its randomness through `wasm-bindgen`, so the module is one
`wasm-bindgen` makes the glue for, and the program that is built turns on getrandom's
`wasm_js` feature, as getrandom's own documentation says to.

## Building

```sh
mise install            # stable Rust, its WebAssembly targets and wasmtime, as mise.toml asks (or any Rust 1.96 or later)
cargo test --workspace  # every test, and every example beside its transcript, on stand-ins with no network
```

CI runs the same on every push: `cargo fmt --check`, clippy with warnings denied on the machine
and on both WASI targets, the tests on the machine and as WebAssembly, and a build of
`rdm-wasm` ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

The gateway's server is the binary `one-system`, configured from the environment;
[its README](crates/one-system/README.md) says how to run it in front of local Laya and hosted Jev.

The default build needs a C compiler: lm15 builds `aws-lc-sys`, and the `repl` feature builds
`psm`, through the parser Monty reads Python with. The tests build `ring` as well, through
one-system's `ureq`. None of them needed cmake on Linux x86_64. `rdm` without any feature
(`--no-default-features`) is the core alone, and builds with no C.

## Links

- The post: [Recursive Decision Models](https://raw.works/recursive-decision-models/)
- The site, where every example runs in a browser: [recursivedecisionmodels.com](https://recursivedecisionmodels.com)
- The Python reference, inside DSPy: [rawwerks/dspy, branch `feat/rdm-v0`](https://github.com/rawwerks/dspy/tree/feat/rdm-v0)
- The Go gateway: [rawwerks/one-system](https://github.com/rawwerks/one-system)
- System One models and Jev: [typesafe.ai](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Laya for llama.cpp: [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF)

## Licence

MIT. The Shopify product taxonomy in `crates/rdm/examples/assets` is under its own licence,
beside it.
