# rdm-wasm

An RDM as a WebAssembly module with nothing under it, and the JavaScript that hosts it.

The module is the search of the [README](../../README.md#one-module), built for
`wasm32-unknown-unknown`: no system, no clock, no network, no thread, and no `wasm-bindgen`.
It imports one function, and exports its memory and three:

```text
imports   host.http(ptr, len) -> ptr     one HTTP request out, as JSON, and its response back
exports   memory, rdm_alloc(len), rdm_free(ptr, len)
          rdm_search(ptr, len) -> ptr    {"text", "question", "server", "model", "key"} in, the result out
```

Its decision model is `one_system::Client`, as it is anywhere: the client writes the request,
checks the answers against the questions, and reads a refusal. Only the network is
elsewhere. The client's transport is the import, and [`host.mjs`](host.mjs) is the host: it
instantiates the module and carries each request with `fetch`.

```sh
cargo build -p rdm-wasm --target wasm32-unknown-unknown --release
node crates/rdm-wasm/host.mjs                                    # a stand-in server, in host.mjs
RDM_SERVER=http://127.0.0.1:8080 node crates/rdm-wasm/host.mjs   # llama-server -hf ggml-org/Laya-GGUF
RDM_SERVER=https://api.typesafe.ai RDM_MODEL=jev-latest RDM_KEY_ENV=TYPESAFE_API_KEY node crates/rdm-wasm/host.mjs
```

```text
{"found":["Flat is better than nested."],"requests":5,"reason":"final"}
5 requests carried by the host
```

## A module that waits, and a host that cannot be waited on

The loop waits for each answer, and `fetch` answers with a promise. Nothing here was made
asynchronous for that. JavaScript Promise Integration lets the engine do it: the host wraps
the import in `WebAssembly.Suspending` and the export in `WebAssembly.promising`, and the
module is suspended, stack and all, while the promise is pending. Node has it from 24, and
so have the browsers built on Chromium. Where an engine has none, the same module runs in a
worker, and the import is a synchronous `XMLHttpRequest`.

## What is the host's

- **The key.** `host.mjs` reads it from the variable `RDM_KEY_ENV` names and adds it to the
  request as it leaves. The module is never given it. A page has no place to keep a key at
  all: there the request goes to a server of one's own, which adds it.
- **Redirects.** The host does not follow one (`redirect: 'manual'`): the request carries a
  key, and goes nowhere it was not sent.
- **A panic.** This target cannot unwind, so a panic ends the module, and the host sees a
  trap. A tool that fails returns an error, which is an observation.

This is one program, to show the seam. It is not a binding: a signature and its tools are
written in Rust here, and a host that is to give its own would need tools that are imports
too.
