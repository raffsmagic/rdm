# one-system

A gateway for decision models: local and hosted System One servers behind one interface.

A decision model is given a state and typed questions about it, and answers each with a
probability: a **noul** (yes or no), a **choice** (one of several) or a **score** (a level on a
rubric). The crate [`systemone`](../systemone) is that protocol. This crate is what stands
between a program and the models that speak it. It is a Rust variation of the Go gateway
[rawwerks/one-system](https://github.com/rawwerks/one-system): the routing is the same, the
registry files load unchanged, and the operational extras (the SQLite log, the persistent
cache) are left out.

There are four things in it, and each of the first two is a `systemone::DecisionModel`:

| | |
|---|---|
| `Client` | One System One server over HTTP: hosted TypeSafe, OpenRouter, llama.cpp serving Laya, another gateway. |
| `Gateway` | Several decision models behind one. It sends each request to one backend and answers as that backend did. |
| `Registry` | A gateway written down as JSON, in the format of the Go gateway's `backends.json`. |
| `Server` | A gateway served over HTTP, and the binary `one-system` (feature `server`, on by default). |

A gateway's backends are any decision models, so they nest: a gateway can route among clients,
other gateways, and functions (`systemone::Scripted`).

What carries a client's requests is a `Transport`. On a machine it is this crate's own HTTP.
In a WebAssembly module the network is the host's: `wasi:http` where the module is built for
WASI 0.2 (`wasm32-wasip2`), and anywhere else a transport the client is given, which can be
a function (`Client::transport`). A module can listen on no port, so a build for
WebAssembly has the client, the gateway and the registry, and no server.

## Ask one model

```rust,no_run
use one_system::{Client, Retry};
use systemone::{DecisionModel, Params, Question, Request, decide};

// Hosted Jev: https://api.typesafe.ai, model jev-latest, key from $TYPESAFE_API_KEY.
let jev = Client::typesafe();
// Or Jev through OpenRouter (key from $OPENROUTER_API_KEY), or Laya on this machine:
let _openrouter = Client::openrouter();
let _laya = Client::local("http://127.0.0.1:8080"); // llama-server -hf ggml-org/Laya-GGUF

let question = Question::noul("Is the customer in the state asking for a refund?");
let request = Request::new("I was charged twice this month.").question("refund", question.clone());
let response = jev.decide(&request)?;
let refund = decide("refund", &question, &response.answers["refund"], &Params::default())?;
println!("{} answered {} ({} tokens)", response.model, refund.value(), response.usage.total());

// One call is one request. Retries are for the caller to turn on:
let _patient = Client::typesafe().retry(Retry::default());
# Ok::<(), systemone::Error>(())
```

The key is given directly (`.key(..)`) or named as an environment variable (`.key_env(..)`,
read at each call). It is not in the client's `Debug` output and not in any error. A refusal
comes back as `systemone::Error::Api` whichever error body the server wrote (TypeSafe's two,
llama.cpp's, a gateway's), a connection that failed or timed out as `Error::Transport`, and
answers that are not for the questions asked as `Error::Evidence`. Redirects are not followed.
A base URL is no place for a key: credentials written into one (`https://user:password@host`)
are not kept, printed or sent, and such a client refuses every call with `Error::Invalid`.

A proxy is taken from the environment when a client is built, by the rules of Go's
`http.ProxyFromEnvironment`, which the Go gateway uses:

- A server on this machine (`localhost`, or a loopback address such as `127.0.0.1` or `[::1]`)
  is never asked through a proxy, whatever is set. A local backend is where a state is meant
  to stay.
- An `https` server goes by `HTTPS_PROXY` and an `http` one by `HTTP_PROXY` (or the same names
  in lower case). Neither uses the other's.
- A host `NO_PROXY` names is asked directly: `*`, a domain (itself and what is under it),
  `.domain` (what is under it), an address, or a block such as `10.0.0.0/8`, any with a port.
- `ALL_PROXY` stands in for a scheme whose own variable is not set. Go reads no such variable.

## Route among several

The request's `model` says where it goes. A backend's id sends it straight there. The
gateway's own name, or no name, lets the gateway choose:

1. The backends *able* to answer are found (hard `Capabilities`: question kinds, question
   count, options per question, text-only state). Then those that would *rather not* have a
   request of this size are set aside (soft `Limits`), unless that leaves none.
2. One left: it is asked, and no selector is.
3. Several left: the selector backend decides. Either it answers one choice among the
   backends' descriptions, or, with a `Selection`, it answers questions about the task and a
   rule escalates when its value is strictly above its threshold.
4. The selector failed: the request goes to the fallback, if there is one.

The backend is asked once. If it fails, its error is returned as it is: nothing is retried
and nothing is sent elsewhere. The response's `model` is the model that answered, and its
usage is the selector's added to the backend's.

```rust
use one_system::{Backend, Capabilities, Gateway, Limits, Picked, Rule, Selection};
use systemone::{Answer, DecisionModel, Question, Request, Scripted};

// Two models that are functions, so this runs anywhere. In use:
// Client::local("http://127.0.0.1:8080") and Client::typesafe().
let laya = Scripted::new(|request| match request.questions.get("domain") {
    // Asked about the task, it says how likely each domain is.
    Some(_) => vec![("domain", Answer::choice([("math_or_logic", 0.8), ("chitchat", 0.2)]))],
    None => vec![("correct", Answer::noul(0.6))],
})
.named("laya");
let jev = Scripted::new(|_| [("correct", Answer::noul(0.99))]).named("jev-1.13.0");

let domain = Question::choice("What domain does `request` belong to?", [
    ("math_or_logic", "mathematics, logic puzzles, proofs".to_string()),
    ("chitchat", "casual conversation, greetings".to_string()),
]);
let gateway = Gateway::builder("privacy-demo")
    .backend(
        Backend::new("local", laya)
            .description("short simple English text, yes or no, ratings")
            .capabilities(Capabilities::all().max_questions(16).min_criteria(2))
            .limits(Limits::default().max_characters(1200).max_non_ascii_letter_fraction(0.02)),
    )
    .backend(Backend::new("hosted", jev).description("other languages, long documents, arithmetic"))
    .selector("local")
    .fallback("local")
    .selection(Selection::new([("domain", domain)], "hosted").rule(Rule::choice("domain", "math_or_logic", 0.5)))
    .cache(1024) // optional: remember answers, and give one again for nothing
    .build()?;

let request = Request::new("17 + 25 = 42").question("correct", Question::noul("Is the arithmetic right?"));
let routed = gateway.route(&request); // `gateway.decide(&request)` is the same without the account
assert_eq!(routed.result?.model, "jev-1.13.0");
assert_eq!(routed.route.picked, Some(Picked::Escalated { question: "domain".into(), value: 0.8 }));
// A backend can be named.
assert_eq!(gateway.decide(&request.clone().model("local"))?.model, "laya");
# Ok::<(), Box<dyn std::error::Error>>(())
```

Limits are routing hints and not a privacy boundary: a request only one backend can take goes
to that backend, whoever runs it. In the rule mode the selector sees the instructions of the
caller's questions and nothing of the state. Otherwise it sees the state only if it is also
the fallback, and else the question definitions and the size of the state.

## Run it against local Laya and hosted Jev

Start Laya with llama.cpp (it listens on `127.0.0.1:8080`, wants no key, and ignores `model`):

```sh
llama-server -hf ggml-org/Laya-GGUF
```

Write a registry, `backends.json`. This one keeps a request on Laya unless Laya gives
`math_or_logic` a probability above 0.5 for the task, and then sends it to Jev:

```json
{
  "name": "privacy-demo",
  "selector": "local",
  "fallback": "local",
  "selection": {
    "questions_file": "routing.questions.json",
    "escalate_to": "hosted",
    "rules": [{"question": "domain", "choice": "math_or_logic", "above": 0.5}]
  },
  "backends": [
    {"id": "local", "base_url": "http://127.0.0.1:8080", "model": "laya",
     "api_key_env": "LOCAL_API_KEY",
     "description": "short simple English text, basic sorting, yes or no, ratings",
     "limits": {"max_characters": 1200, "max_non_ascii_letter_fraction": 0.02, "max_questions": 16, "max_criteria": 20},
     "capabilities": {"question_types": ["choice", "score", "noul"], "max_questions": 16, "min_criteria": 2}},
    {"id": "hosted", "base_url": "https://api.typesafe.ai", "model": "jev-latest",
     "api_key_env": "TYPESAFE_API_KEY",
     "description": "other languages, long documents, arithmetic, multi-step reasoning"}
  ]
}
```

and `routing.questions.json`:

```json
{"domain": {"type": "choice", "instructions": "What domain does `request` belong to?",
            "criteria": {"math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
                         "chitchat": "casual conversation, greetings, small talk"}}}
```

The registries that ship with the Go gateway (`backends.json`, `examples/*.backends.json`)
load as they are. Every backend names the environment variable its key is in, and every one
of them must be set, so a server that wants no key still needs some value there.

A backend that reads pictures says so in its capabilities, by OpenRouter's name for it:
`"input_modalities": ["text", "image"]`. A request whose state shows pictures
(`systemone::image`) goes only to such a backend. One that declares nothing, or a state that
must be text (`"structured_state": false`), is text only, and a selector that cannot see is
given the size of the state and how many pictures it has, never the pictures. A `Client` at
TypeSafe's address refuses such a request: its JSON endpoint does not read image parts
(checked 2026-10-17), where OpenRouter's decisions do.

Serve it:

```sh
export ONE_SYSTEM_API_KEY=...   # the key callers present
export TYPESAFE_API_KEY=...     # hosted Jev's
export LOCAL_API_KEY=unused     # llama.cpp asks for none
ONE_SYSTEM_CONFIG=backends.json ONE_SYSTEM_ADDR=127.0.0.1:8090 cargo run -p one-system
```

and ask it, with `curl` or with a `Client`:

```sh
curl -H "Authorization: Bearer $ONE_SYSTEM_API_KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8090/v1/systemone --data-binary '{
    "model": "privacy-demo",
    "state": "17 + 25 = 42",
    "questions": {"correct": {"type": "noul", "instructions": "Is the arithmetic calculation correct?"}}
  }'
```

```rust,no_run
use one_system::Client;
let gateway = Client::new("http://127.0.0.1:8090").model("privacy-demo").key_env("ONE_SYSTEM_API_KEY");
```

Or skip the server and route in the process, from the same file:

```rust,no_run
let gateway = one_system::Gateway::load("backends.json")?; // keys from the environment
# Ok::<(), one_system::ConfigError>(())
```

The server answers `POST /v1/systemone`, `GET /v1/models` and `GET /v1/capabilities`, all
behind the one key. Every error is `{"detail":[{"loc":["body"],"msg":"...","type":"..."}]}`
with a fixed message: `not_found` (404), `unauthorized` (401), `method_not_allowed` (405),
then 422 `invalid_body` (over 8 MiB), `schema_validation`, `unsupported_model`,
`unsupported_capability`, and for a backend that failed `upstream_unavailable`,
`upstream_rejected` (with the backend's status, but 502 for its 400, 401 and 403),
`invalid_upstream_response` (what it sent is not a response) or `mismatched_answers` (it
answered other questions than it was asked). A backend's own words are never passed on. Each
request routed is one line on standard error. `ONE_SYSTEM_CACHE_ENTRIES=1024` turns the
cache on. It is a small server for this machine or a trusted network.

Over HTTP the gateway is asked for by its name: a `model` that is empty is 422
`unsupported_model`, as any other name nobody has. (In the process, a request that names no
model lets the gateway choose.) A path is matched as it reads once its percent escapes are
decoded.

What the server sends is always a whole response by TypeSafe's published schema. That schema
requires of a choice its `choice` and `confidence`, and of a score its `score`, `confidence`
and `legend`. A backend that left one out (a function has only probabilities to give) has it
worked out from the probabilities and the question: the likeliest option, the expected level,
the confidence TypeSafe defines for each, the rubric by level. What a backend did say is sent
as it said it. What cannot be worked out, or written as JSON, is a 502.

## What is different from the Go gateway

- Backends are trait objects, not URLs: anything that decides can be one, and the gateway is
  used in the process as readily as over HTTP.
- A request is parsed and written again with `serde_json`, so members the protocol does not
  name are dropped, and a number is carried as its value and not as it was spelled:
  `1.2300e+04` reaches a backend as `12300.0`, and `-0` as `-0.0`.
- Two kinds of JSON the Go gateway forwards are refused here, with 422 `schema_validation`:
  a number too large for a 64-bit float (`1e400`), and a body nested 128 levels deep or
  more. Both are bounds of the JSON parser, kept on purpose: the first would need every
  number in the workspace held as text, and the second is what stands between a hostile body
  and the stack.
- Decisions the gateway makes itself (which backend a selector chose, what a score rule
  reads) are made from the probabilities, as `systemone::decide` makes them, and not from
  the `choice` and `score` a server reports.
- A response is checked more strictly in one way and less in another. A choice or a score
  must give a probability for each option or level declared and for no other, which Go does
  not ask. A `choice`, `score`, `confidence` or `legend` a backend left out, and a `model` or
  `usage` it left out, are filled in (the backend's id, zero tokens), where Go answers 502
  `invalid_upstream_response`.
- The size of a state is measured on the value, written compactly. Go measures the bytes it
  was sent, so a state typed with escapes (`"中文"`) is to Go a run of ASCII, and
  here the two characters it stands for. A letter is Unicode's category L in both, by the
  tables of Unicode 17 here.
- A registry member written twice in one object is refused. Go keeps the last of the two.
- A client also reads `ALL_PROXY`, for a scheme with no proxy variable of its own.
- The cache is a map in memory, keyed by SHA-256 of the request as serialized. No SQLite
  ledger, no replay mode, no exchange log, no configuration revision.
- No 300-second deadline over a whole request, and a caller that goes away does not cancel
  the backend call. A client's own timeout (180 seconds unless set) bounds each call.

## Tests

```sh
cargo test -p one-system
```

Nothing in the tests leaves the machine: the servers the client talks to are started inside
the tests on `127.0.0.1`. The tests of what a proxy in the environment does run the test
binary again as a child with that environment, and the proxy is one of those servers. The Go
gateway's own registries are loaded from a checkout of it
when one is found (set `ONE_SYSTEM_GO_REPOSITORY` to say where), and passed over quietly
when not. The examples in this file are run as doc tests.
