//! The server, end to end: a gateway listening on 127.0.0.1, asked by the client and by hand.
#![cfg(feature = "server")]
// The servers these tests ask are on 127.0.0.1, and a WebAssembly module can listen on no port.
#![cfg(not(target_family = "wasm"))]

mod common;

use std::sync::Arc;

use common::{Reply, Upstream, answers, api, failing, http, model};
use one_system::{Backend, Capabilities, Client, Gateway, MAX_BODY_BYTES, Running, Server};
use serde_json::json;
use systemone::{Answer, DecisionModel, Error, Kind, Question, Request, Scripted};

const KEY: &str = "gateway-key-for-tests";

fn refund() -> Request {
    Request::new("Please refund the duplicate charge.").question("refund", Question::noul("Does the customer request a refund?"))
}

/// A gateway of two models that are functions, the hosted one selecting the local one.
fn gateway() -> (Gateway, Arc<Scripted>, Arc<Scripted>) {
    let local = model("laya-english", 7);
    let hosted = Arc::new(
        Scripted::new(|request| {
            let mut given = answers(request);
            if let Some(question) = request.questions.get("backend") {
                let labels = question.labels();
                given.insert(
                    "backend".into(),
                    systemone::Answer::chose("local", labels.iter().map(String::as_str)),
                );
            }
            given
        })
        .named("jev-1.13.0")
        .tokens(300),
    );
    let gateway = Gateway::builder("demo")
        .backend(
            Backend::new("local", local.clone())
                .description("Short, plain English.")
                .capabilities(Capabilities::all().max_questions(2)),
        )
        .backend(Backend::new("hosted", hosted.clone()).description("Everything else."))
        .selector("hosted")
        .build()
        .unwrap();
    (gateway, local, hosted)
}

fn serve(gateway: Gateway) -> Running {
    Server::bind("127.0.0.1:0", gateway, KEY).unwrap().spawn()
}

#[test]
fn a_caller_slow_to_send_holds_up_nobody_else() {
    use std::io::{Read, Write};
    let server = serve(gateway().0);
    // Callers that say a body is coming and send none of it, with the key and without.
    let mut stalled = Vec::new();
    for caller in 0..48 {
        let authorization = if caller % 2 == 0 {
            format!("Authorization: Bearer {KEY}\r\n")
        } else {
            String::new()
        };
        let mut stream = std::net::TcpStream::connect(server.addr()).unwrap();
        write!(
            stream,
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\n{authorization}Content-Length: 50000\r\n\r\n"
        )
        .unwrap();
        stalled.push(stream);
        // One at a time, as callers come: tiny_http's pool can queue a connection that arrives
        // in the same instant as the one before, behind a thread these callers then hold.
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Those without the key have been told so already, body or no body.
    stalled[1].set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    let mut heard = [0u8; 12];
    stalled[1].read_exact(&mut heard).unwrap();
    assert_eq!(&heard, b"HTTP/1.1 401");
    // And everyone else is answered while they dawdle.
    let body = serde_json::to_vec(&refund().model("local")).unwrap();
    assert_eq!(post(&server, &body).status, 200);
    assert_eq!(http(server.addr(), "GET", "/v1/models", Some(KEY), None).status, 200);

    // A body that stops short of the length it declared was not read, and is told so.
    let mut stream = std::net::TcpStream::connect(server.addr()).unwrap();
    write!(
        stream,
        "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\nContent-Length: 50000\r\n\r\n{{\"model\":"
    )
    .unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    let mut answer = String::new();
    let _ = stream.read_to_string(&mut answer);
    assert!(answer.starts_with("HTTP/1.1 422") && answer.contains("invalid_body"), "{answer}");
}

fn post(server: &Running, body: &[u8]) -> common::Raw {
    http(server.addr(), "POST", "/v1/systemone", Some(KEY), Some(body))
}

#[test]
fn a_round_trip() {
    let (gateway, local, hosted) = gateway();
    let server = serve(gateway);
    let client = Client::new(server.url()).key(KEY);

    // By the gateway's name: the selector is asked, then the backend it chose.
    let response = client.decide(&refund().model("demo")).unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(response.answers["refund"].noul, Some(0.9));
    assert_eq!(response.usage.input_tokens, 307);
    assert_eq!((local.requests().len(), hosted.requests().len()), (1, 1));

    // By a backend's id: straight to it.
    let response = client.clone().model("hosted").decide(&refund()).unwrap();
    assert_eq!(response.model, "jev-1.13.0");
    assert_eq!(response.usage.input_tokens, 300);
    assert_eq!(local.requests().len(), 1);

    // As it is on the wire.
    let body = serde_json::to_vec(&refund().model("local")).unwrap();
    let raw = post(&server, &body);
    assert_eq!(raw.status, 200);
    assert_eq!(raw.header("content-type"), Some("application/json"));
    assert_eq!(raw.header("cache-control"), Some("no-store"));
    assert_eq!(
        raw.json(),
        json!({"model": "laya-english", "answers": {"refund": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 7, "output_tokens": 0}})
    );
}

#[test]
fn every_endpoint_wants_the_key() {
    let server = serve(gateway().0);
    let unauthorized = (401, "unauthorized".to_string(), "A valid bearer API key is required".to_string());
    assert_eq!(
        api(&Client::new(server.url()).key("some-other-key").decide(&refund()).unwrap_err()),
        unauthorized
    );
    assert_eq!(api(&Client::new(server.url()).decide(&refund()).unwrap_err()), unauthorized);
    assert_eq!(api(&Client::new(server.url()).models().unwrap_err()), unauthorized);

    for path in ["/v1/systemone", "/v1/models", "/v1/capabilities"] {
        for key in [
            None,
            Some("some-other-key"),
            Some("gateway-key-for-test"),
            Some("gateway-key-for-tests2"),
        ] {
            let raw = http(server.addr(), "GET", path, key, None);
            assert_eq!((raw.status, raw.code().as_str()), (401, "unauthorized"), "{path} {key:?}");
            assert_eq!(raw.header("www-authenticate"), Some("Bearer"));
            assert_eq!(raw.header("content-type"), Some("application/json"));
        }
    }
    // The scheme is part of what is compared, as it is written.
    let stream = |authorization: &str| {
        use std::io::{Read, Write};
        let mut stream = std::net::TcpStream::connect(server.addr()).unwrap();
        write!(
            stream,
            "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\nAuthorization: {authorization}\r\n\r\n"
        )
        .unwrap();
        let mut text = String::new();
        let _ = stream.read_to_string(&mut text);
        text
    };
    assert!(stream(&format!("Bearer {KEY}")).starts_with("HTTP/1.1 200"));
    assert!(stream(&format!("bearer {KEY}")).starts_with("HTTP/1.1 401"));
    assert!(stream(KEY).starts_with("HTTP/1.1 401"));
    assert!(stream(&format!("Basic {KEY}")).starts_with("HTTP/1.1 401"));
}

#[test]
fn the_path_is_checked_first_then_the_key_then_the_method() {
    let server = serve(gateway().0);
    // No such path: 404, key or no key.
    for path in ["/", "/v1", "/v1/systemone/", "/v1/chat/completions", "/health", "/v1/Models"] {
        for key in [None, Some(KEY)] {
            let raw = http(server.addr(), "POST", path, key, Some(b"{}"));
            assert_eq!((raw.status, raw.code().as_str()), (404, "not_found"), "{path}");
            assert_eq!(
                raw.json()["detail"],
                json!([{"loc": ["body"], "msg": "Endpoint not found", "type": "not_found"}])
            );
        }
    }
    // Through the client: a base that leads nowhere.
    let lost = Client::new(format!("{}/nowhere", server.url()))
        .key("some-other-key")
        .decide(&refund())
        .unwrap_err();
    assert_eq!(api(&lost), (404, "not_found".into(), "Endpoint not found".into()));

    // The wrong method without the key is a matter of the key.
    assert_eq!(http(server.addr(), "DELETE", "/v1/models", None, None).status, 401);
    assert_eq!(http(server.addr(), "GET", "/v1/systemone", None, None).status, 401);
    // With the key it is a matter of the method, and the answer says which is allowed.
    for (method, path, allowed) in [
        ("POST", "/v1/models", "GET"),
        ("DELETE", "/v1/models", "GET"),
        ("PUT", "/v1/capabilities", "GET"),
        ("GET", "/v1/systemone", "POST"),
        ("PUT", "/v1/systemone", "POST"),
    ] {
        let raw = http(server.addr(), method, path, Some(KEY), Some(b"{}"));
        assert_eq!((raw.status, raw.code().as_str()), (405, "method_not_allowed"), "{method} {path}");
        assert_eq!(raw.header("allow"), Some(allowed));
        assert_eq!(raw.json()["detail"][0]["msg"], "Method not allowed");
    }
    // A query is no part of the path.
    assert_eq!(http(server.addr(), "GET", "/v1/models?x=1", Some(KEY), None).status, 200);
}

#[test]
fn a_body_may_be_eight_mebibytes_and_no_more() {
    let (gateway, local, _) = gateway();
    let server = serve(gateway);
    // A request that is exactly the most allowed: a state padded out to it.
    let empty = serde_json::to_vec(&Request::new("").model("local").question("q", Question::noul("?"))).unwrap();
    let request = Request::new("x".repeat(MAX_BODY_BYTES - empty.len()))
        .model("local")
        .question("q", Question::noul("?"));
    let exact = serde_json::to_vec(&request).unwrap();
    assert_eq!(exact.len(), MAX_BODY_BYTES);
    assert_eq!(post(&server, &exact).status, 200);
    assert_eq!(local.requests().len(), 1);

    // One byte more, through the client and by hand.
    let request = Request::new("x".repeat(MAX_BODY_BYTES - empty.len() + 1))
        .model("local")
        .question("q", Question::noul("?"));
    let error = Client::new(server.url()).key(KEY).decide(&request).unwrap_err();
    assert_eq!(
        api(&error),
        (422, "invalid_body".into(), "Request body is unreadable or exceeds 8 MiB".into())
    );
    let mut over = exact.clone();
    over.push(b' ');
    let raw = post(&server, &over);
    assert_eq!((raw.status, raw.code().as_str()), (422, "invalid_body"));
    // The size is checked after the key: too large and unauthorized is unauthorized.
    assert_eq!(http(server.addr(), "POST", "/v1/systemone", None, Some(&over)).status, 401);
    assert_eq!(local.requests().len(), 1);
}

#[test]
fn a_body_that_is_not_a_request_is_a_schema_error() {
    let (gateway, local, hosted) = gateway();
    let server = serve(gateway);
    let bodies: [&[u8]; 16] = [
        // Not JSON, or more than one value. Malformed JSON is a schema error, not a body error.
        b"",
        b"not json",
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"noul"}}"#,
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"noul"}}} {}"#,
        b"[]",
        b"null",
        // A member missing, or of the wrong kind.
        br#"{"state":"s","questions":{"q":{"type":"noul"}}}"#,
        br#"{"model":7,"state":"s","questions":{"q":{"type":"noul"}}}"#,
        br#"{"model":"demo","questions":{"q":{"type":"noul"}}}"#,
        br#"{"model":"demo","state":null,"questions":{"q":{"type":"noul"}}}"#,
        br#"{"model":"demo","state":42,"questions":{"q":{"type":"noul"}}}"#,
        br#"{"model":"demo","state":"s","questions":{}}"#,
        // A question that is not one.
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"essay"}}}"#,
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"Noul"}}}"#,
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"choice"}}}"#,
        br#"{"model":"demo","state":"s","questions":{"q":{"type":"score","criteria":[]}}}"#,
    ];
    for body in bodies {
        let raw = post(&server, body);
        assert_eq!(
            (raw.status, raw.code().as_str()),
            (422, "schema_validation"),
            "{}",
            String::from_utf8_lossy(body)
        );
        assert_eq!(
            raw.json(),
            json!({"detail": [{"loc": ["body"], "msg": "Request does not match the official SystemOneRequest schema", "type": "schema_validation"}]})
        );
        assert!(raw.body.ends_with('\n'));
    }
    assert!(local.requests().is_empty() && hosted.requests().is_empty());

    // What the schema allows is allowed: members it does not name, instructions left out, a state that is a list.
    let lenient = br#"{"model":"local","state":["a",1],"questions":{"q":{"type":"noul","criteria":null,"weight":3}},"opaque":{"x":1}}"#;
    assert_eq!(post(&server, lenient).status, 200);
}

#[test]
fn what_the_gateway_refuses_the_server_refuses_in_the_same_words() {
    let server = serve(gateway().0);
    let client = Client::new(server.url()).key(KEY);
    assert_eq!(
        api(&client.decide(&refund().model("gpt-9")).unwrap_err()),
        (422, "unsupported_model".into(), "Unknown model; see GET /v1/models".into())
    );
    // The client names no model and the request names none: the wire wants one.
    assert_eq!(api(&client.decide(&refund()).unwrap_err()).1, "schema_validation");
    // Three questions are more than the local model takes.
    let three = refund()
        .model("local")
        .question("b", Question::noul("?"))
        .question("c", Question::noul("?"));
    assert_eq!(
        api(&client.decide(&three).unwrap_err()),
        (
            422,
            "unsupported_capability".into(),
            "No requested backend supports these inputs".into()
        )
    );
}

#[test]
fn a_backend_that_fails_is_told_of_without_its_words() {
    let refused = |status: u16| Error::Api {
        status,
        code: "theirs".into(),
        message: "PRIVATE words from upstream".into(),
    };
    let cases = [
        (refused(429), 429, "upstream_rejected", "Upstream could not complete the request"),
        (refused(422), 422, "upstream_rejected", "Upstream could not complete the request"),
        (refused(529), 529, "upstream_rejected", "Upstream could not complete the request"),
        // A backend that refuses the gateway's own key, or its own request, is no fault of the caller's.
        (refused(401), 502, "upstream_rejected", "Upstream could not complete the request"),
        (refused(400), 502, "upstream_rejected", "Upstream could not complete the request"),
        (
            Error::Transport("PRIVATE cause".into()),
            502,
            "upstream_unavailable",
            "Upstream request failed",
        ),
        (
            Error::Evidence("PRIVATE answer".into()),
            502,
            "invalid_upstream_response",
            "Upstream returned an invalid SystemOneResponse",
        ),
    ];
    for (failure, status, code, message) in cases {
        let backend = failing("broken", failure);
        let gateway = Gateway::builder("solo")
            .backend(Backend::new("only", backend.clone()))
            .build()
            .unwrap();
        let server = serve(gateway);
        let body = serde_json::to_vec(&refund().model("solo")).unwrap();
        let raw = post(&server, &body);
        assert_eq!((raw.status, raw.code().as_str()), (status, code));
        assert_eq!(raw.json()["detail"][0]["msg"], message);
        assert!(!raw.body.contains("PRIVATE"));
        // Asked once.
        assert_eq!(backend.requests().len(), 1);
    }

    // A backend whose answers are for other questions.
    let confused = Scripted::new(|_| [("refund", systemone::Answer::chose("a", ["a", "b"]))]);
    let server = serve(Gateway::builder("solo").backend(Backend::new("only", confused)).build().unwrap());
    let error = Client::new(server.url()).key(KEY).model("solo").decide(&refund()).unwrap_err();
    assert_eq!(api(&error).1, "mismatched_answers");
    assert_eq!(error.status(), Some(502));
}

#[test]
fn the_catalogue_and_the_capabilities() {
    let server = serve(gateway().0);
    let models = Client::new(server.url()).key(KEY).models().unwrap();
    let names: Vec<&str> = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["name"].as_str().unwrap())
        .collect();
    // The gateway's own name first, then its backends in order.
    assert_eq!(names, ["demo", "hosted", "local"]);
    assert_eq!(models["models"][1]["description"], "Everything else.");

    let raw = http(server.addr(), "GET", "/v1/capabilities", Some(KEY), None);
    assert_eq!(raw.status, 200);
    assert_eq!(raw.header("cache-control"), Some("no-store"));
    assert_eq!(
        raw.json(),
        json!({"version": 1, "models": [
            {"name": "hosted", "capabilities": null},
            {"name": "local", "capabilities": {"question_types": ["choice", "score", "noul"], "max_questions": 2}},
        ]})
    );
}

#[test]
fn an_answer_from_the_cache_says_so() {
    let local = model("laya-english", 7);
    let gateway = Gateway::builder("solo")
        .backend(Backend::new("only", local.clone()))
        .cache(4)
        .build()
        .unwrap();
    let server = serve(gateway);
    let body = serde_json::to_vec(&refund().model("solo")).unwrap();
    let first = post(&server, &body);
    assert_eq!((first.status, first.header("x-one-system-cache")), (200, None));
    assert_eq!(first.json()["usage"]["input_tokens"], 7);
    let second = post(&server, &body);
    assert_eq!((second.status, second.header("x-one-system-cache")), (200, Some("hit")));
    assert_eq!(second.json()["usage"]["input_tokens"], 0);
    assert_eq!(second.json()["answers"], first.json()["answers"]);
    assert_eq!(local.requests().len(), 1);
}

#[test]
fn a_gateway_routes_to_another_gateway_over_http() {
    // A server that is a function, as llama.cpp would be; a gateway over it, served; and a
    // gateway in this process that routes between that server and a function.
    let laya = Upstream::answering("laya-english", 12);
    let inner = Gateway::builder("locals")
        .backend(Backend::new("laya", Client::local(laya.url())).capabilities(Capabilities::new([Kind::Noul, Kind::Choice])))
        .build()
        .unwrap();
    let served = serve(inner);
    let outer = Gateway::builder("everything")
        .backend(
            Backend::new("served", Client::new(served.url()).key(KEY).model("locals"))
                .capabilities(Capabilities::new([Kind::Noul, Kind::Choice])),
        )
        .backend(Backend::new("scores", model("scorer", 1)).capabilities(Capabilities::new([Kind::Score])))
        .selector("served")
        .build()
        .unwrap();

    let response = outer.decide(&refund()).unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(response.usage.input_tokens, 12);
    // What reached the model at the far end is what was asked, under no model's name.
    let seen = laya.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].request(), refund());
    assert_eq!(seen[0].header("authorization"), None);

    // A score goes the other way, and never leaves the process.
    let score = Request::new("state").question("rate", Question::score("?", ["low", "high"]));
    assert_eq!(outer.decide(&score).unwrap().model, "scorer");
    assert_eq!(laya.hits(), 1);

    // Asked directly, the served gateway refuses what its backend cannot take.
    let direct = Client::new(served.url()).key(KEY).model("laya");
    assert_eq!(api(&direct.decide(&score).unwrap_err()).1, "unsupported_capability");
}

#[test]
fn a_server_needs_a_key_that_can_be_one() {
    for bad in ["", "two words", "line\nbreak"] {
        let error = Server::bind("127.0.0.1:0", gateway().0, bad).err().unwrap();
        assert!(error.message().contains("ONE_SYSTEM_API_KEY"));
        assert!(bad.is_empty() || !error.message().contains(bad));
    }
    let server = Server::bind("127.0.0.1:0", gateway().0, KEY).unwrap();
    assert_eq!(server.url(), format!("http://{}", server.addr()));
    // Stopped, it no longer answers.
    let running = server.spawn();
    let address = running.addr();
    assert_eq!(http(address, "GET", "/v1/models", Some(KEY), None).status, 200);
    running.stop();
    let refused = (0..200).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::net::TcpStream::connect(address).is_err()
    });
    assert!(refused, "the address is still listened on");
}

#[test]
fn a_request_that_declares_terabytes_does_not_bring_the_server_down() {
    use std::io::{Read, Write};
    let server = serve(gateway().0);
    // With the key and without: the size is seen before either is looked at.
    for authorization in [String::new(), format!("Authorization: Bearer {KEY}\r\n")] {
        let mut stream = std::net::TcpStream::connect(server.addr()).unwrap();
        stream.set_read_timeout(Some(std::time::Duration::from_millis(300))).unwrap();
        write!(
            stream,
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\n{authorization}Content-Length: 9000000000000\r\n\r\n"
        )
        .unwrap();
        // It is not answered, and not read.
        let mut heard = [0u8; 64];
        assert_eq!(stream.read(&mut heard).unwrap_or(0), 0);
    }
    // The server is still there for everyone else.
    assert_eq!(http(server.addr(), "GET", "/v1/models", Some(KEY), None).status, 200);
    let body = serde_json::to_vec(&refund().model("local")).unwrap();
    assert_eq!(post(&server, &body).status, 200);
}

/// A gateway whose one backend is a server that says this, whatever it is asked.
fn before(upstream: &Upstream) -> Running {
    let gateway = Gateway::builder("solo")
        .backend(Backend::new("only", Client::new(upstream.url()).model("m").key("sk-upstream")))
        .build()
        .unwrap();
    serve(gateway)
}

fn tone() -> Request {
    Request::new("Thanks, that fixed it.")
        .model("solo")
        .question("yes", Question::noul("Is the customer content?"))
        .question("tone", Question::choice("Which tone?", [("warm", None), ("cold", None)]))
        .question("rate", Question::score("How content?", ["not at all", "somewhat", "wholly"]))
}

#[test]
fn a_server_behind_the_gateway_that_answers_other_questions_is_mismatched_answers() {
    let good = json!({"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {
        "yes": {"type": "noul", "noul": 0.5},
        "tone": {"type": "choice", "choice": "warm", "confidence": 0.2, "probabilities": {"warm": 0.6, "cold": 0.4}},
        "rate": {"type": "score", "score": 1.0, "confidence": 0.5, "legend": {"0": "not at all", "1": "somewhat", "2": "wholly"},
                 "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}}}});
    let altered = |change: fn(&mut serde_json::Value)| {
        let mut response = good.clone();
        change(&mut response);
        response.to_string()
    };
    let mismatched = (
        "mismatched_answers",
        "Upstream answers do not match the requested question IDs, types, or choices",
    );
    let invalid = ("invalid_upstream_response", "Upstream returned an invalid SystemOneResponse");
    let cases = [
        // Answers that are answers, to other questions than the ones asked.
        (
            altered(|response| {
                let answers = response["answers"].as_object_mut().unwrap();
                let moved = answers.shift_remove("yes").unwrap();
                answers.insert("other".into(), moved);
            }),
            mismatched,
        ),
        (
            altered(|response| response["answers"]["extra"] = json!({"type": "noul", "noul": 0.5})),
            mismatched,
        ),
        (
            altered(|response| response["answers"]["yes"] = response["answers"]["tone"].clone()),
            mismatched,
        ),
        (altered(|response| response["answers"]["tone"]["choice"] = json!("z")), mismatched),
        (
            altered(|response| response["answers"]["tone"]["probabilities"] = json!({"warm": 0.5, "hot": 0.5})),
            mismatched,
        ),
        // What is not a System One response at all.
        (
            altered(|response| {
                response["answers"]["yes"].as_object_mut().unwrap().shift_remove("noul");
            }),
            invalid,
        ),
        (
            altered(|response| {
                response["answers"]["yes"].as_object_mut().unwrap().shift_remove("type");
            }),
            invalid,
        ),
        (
            altered(|response| {
                response["answers"]["rate"].as_object_mut().unwrap().shift_remove("probabilities");
            }),
            invalid,
        ),
        (altered(|response| response["answers"] = json!([])), invalid),
        ("not json".to_string(), invalid),
    ];
    for (said, (code, message)) in cases {
        let shown = said.clone();
        let upstream = Upstream::start(move |_, _| Reply::ok(said.clone()));
        let server = before(&upstream);
        let raw = post(&server, &serde_json::to_vec(&tone()).unwrap());
        assert_eq!((raw.status, raw.code().as_str()), (502, code), "{shown}");
        assert_eq!(raw.json()["detail"][0]["msg"], message);
        assert_eq!(upstream.hits(), 1);
    }

    // The same answers from a backend that is a function are told of in the same words.
    let headless = Scripted::new(|request| {
        let mut given = answers(request);
        given["yes"].noul = None;
        given
    });
    let server = serve(Gateway::builder("solo").backend(Backend::new("only", headless)).build().unwrap());
    let raw = post(&server, &serde_json::to_vec(&tone()).unwrap());
    assert_eq!((raw.status, raw.code().as_str()), (502, invalid.0));
}

#[test]
fn what_the_server_sends_is_a_whole_response_by_the_official_schema() {
    // A server that gives the probabilities and leaves the rest to be worked out.
    let bare = json!({"answers": {
        "yes": {"type": "noul", "noul": 0.5},
        "tone": {"type": "choice", "probabilities": {"cold": 0.25, "warm": 0.75}},
        "rate": {"type": "score", "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}}})
    .to_string();
    let upstream = Upstream::start(move |_, _| Reply::ok(bare.clone()));
    let server = before(&upstream);
    let raw = post(&server, &serde_json::to_vec(&tone()).unwrap());
    assert_eq!(raw.status, 200, "{}", raw.body);
    assert_eq!(
        raw.json(),
        json!({"model": "only", "usage": {"input_tokens": 0, "output_tokens": 0}, "answers": {
            "yes": {"type": "noul", "noul": 0.5},
            "tone": {"type": "choice", "choice": "warm", "confidence": 0.5, "probabilities": {"cold": 0.25, "warm": 0.75}},
            "rate": {"type": "score", "score": 1.5, "confidence": systemone::score_confidence(&[0.0, 0.5, 0.5]),
                     "legend": {"0": "not at all", "1": "somewhat", "2": "wholly"},
                     "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}}})
    );

    // What a server did say is left as it said it.
    let said = json!({"model": "m", "usage": {"input_tokens": 3, "output_tokens": 1}, "answers": {
        "yes": {"type": "noul", "noul": 0.5},
        "tone": {"type": "choice", "choice": "cold", "confidence": 0.125, "probabilities": {"cold": 0.25, "warm": 0.75}},
        "rate": {"type": "score", "score": 2.0, "confidence": 0.875, "legend": {"0": "a", "1": "b", "2": "c"},
                 "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}}});
    let whole = said.to_string();
    let upstream = Upstream::start(move |_, _| Reply::ok(whole.clone()));
    let server = before(&upstream);
    assert_eq!(post(&server, &serde_json::to_vec(&tone()).unwrap()).json(), said);

    // A backend that is a function has no legend to give, and is given one.
    let server = serve(
        Gateway::builder("solo")
            .backend(Backend::new("only", model("laya", 2)))
            .build()
            .unwrap(),
    );
    let raw = post(&server, &serde_json::to_vec(&tone()).unwrap());
    assert_eq!(raw.status, 200);
    let answers = &raw.json()["answers"];
    assert_eq!(
        answers["rate"]["legend"],
        json!({"0": "not at all", "1": "somewhat", "2": "wholly"})
    );
    assert_eq!(answers["rate"]["score"], json!(0.0));
    assert_eq!(
        (&answers["tone"]["choice"], &answers["tone"]["confidence"]),
        (&json!("warm"), &json!(1.0))
    );

    // What cannot be worked out is not made up: a choice among no options has no option picked.
    let nothing = Request::new("s")
        .model("solo")
        .question("pick", Question::choice("?", Vec::<(String, Option<String>)>::new()));
    let empty = Scripted::new(|_| {
        [(
            "pick",
            Answer {
                kind: Some(Kind::Choice),
                probabilities: Some(Default::default()),
                ..Answer::default()
            },
        )]
    });
    let server = serve(Gateway::builder("solo").backend(Backend::new("only", empty)).build().unwrap());
    let raw = post(&server, &serde_json::to_vec(&nothing).unwrap());
    assert_eq!((raw.status, raw.code().as_str()), (502, "invalid_upstream_response"));

    // And a number that is no number cannot be written as JSON at all.
    let unsure = Scripted::new(|_| [("yes", Answer::noul(f64::NAN))]);
    let server = serve(Gateway::builder("solo").backend(Backend::new("only", unsure)).build().unwrap());
    let asked = Request::new("s").model("solo").question("yes", Question::noul("?"));
    let raw = post(&server, &serde_json::to_vec(&asked).unwrap());
    assert_eq!((raw.status, raw.code().as_str()), (502, "invalid_response"));
    assert_eq!(
        raw.json()["detail"][0]["msg"],
        "Upstream response cannot be returned as SystemOneResponse"
    );
}

#[test]
fn an_empty_model_names_nobody_over_http() {
    let (gateway, local, hosted) = gateway();
    // In the process, a request that names no model lets the gateway choose.
    assert!(gateway.decide(&refund()).is_ok());
    local.clear();
    hosted.clear();
    let server = serve(gateway);
    let raw = post(
        &server,
        br#"{"model":"","state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#,
    );
    assert_eq!((raw.status, raw.code().as_str()), (422, "unsupported_model"));
    assert_eq!(raw.json()["detail"][0]["msg"], "Unknown model; see GET /v1/models");
    assert!(local.requests().is_empty() && hosted.requests().is_empty());
    // A client with no model of its own sends no `model` at all, which is no request.
    let error = Client::new(server.url()).key(KEY).decide(&refund()).unwrap_err();
    assert_eq!(api(&error).1, "schema_validation");
    assert!(local.requests().is_empty() && hosted.requests().is_empty());
    // A body that is no request is still told that first.
    let raw = post(&server, br#"{"model":"","state":null,"questions":{}}"#);
    assert_eq!(raw.code(), "schema_validation");
}

#[test]
fn a_path_is_matched_as_it_reads_once_decoded() {
    let server = serve(gateway().0);
    let get = |path: &str| http(server.addr(), "GET", path, Some(KEY), None);
    assert_eq!(get("/v1/%6dodels").json(), get("/v1/models").json());
    assert_eq!(get("/v1/%6Dodels?page=%32").status, 200);
    assert_eq!(get("/v1%2Fcapabilities").status, 200);
    let body = serde_json::to_vec(&refund().model("local")).unwrap();
    assert_eq!(http(server.addr(), "POST", "/v1/system%6Fne", Some(KEY), Some(&body)).status, 200);
    // The checks keep their order: a path that decodes to an endpoint wants the key and the method.
    assert_eq!(http(server.addr(), "GET", "/v1/%6dodels", None, None).status, 401);
    assert_eq!(http(server.addr(), "POST", "/v1/%6dodels", Some(KEY), Some(b"{}")).status, 405);
    // What does not decode to one is nowhere, and so is an escape that is none.
    for nowhere in [
        "/v1/%6eodels",
        "/v1/models%2F",
        "/v1/models%3Fx",
        "/v1/%zzmodels",
        "/v1/models%",
        "/v1/models%6",
        "/v1/model%FFs",
    ] {
        assert_eq!(get(nowhere).code(), "not_found", "{nowhere}");
    }
}

#[test]
fn two_kinds_of_json_this_gateway_does_not_take() {
    let (gateway, local, hosted) = gateway();
    let server = serve(gateway);
    // A number too large for a float. The Go gateway carries it through as it was written.
    let raw = post(
        &server,
        br#"{"model":"local","state":{"n":1e400},"questions":{"q":{"type":"noul"}}}"#,
    );
    assert_eq!((raw.status, raw.code().as_str()), (422, "schema_validation"));
    // A body nested 128 levels deep or more: a state of this many lists, inside the request.
    let nested = |depth: usize| {
        format!(
            r#"{{"model":"local","state":{}"s"{},"questions":{{"q":{{"type":"noul"}}}}}}"#,
            "[".repeat(depth),
            "]".repeat(depth)
        )
    };
    for depth in [127, 200] {
        let raw = post(&server, nested(depth).as_bytes());
        assert_eq!((raw.status, raw.code().as_str()), (422, "schema_validation"), "{depth}");
    }
    assert!(local.requests().is_empty() && hosted.requests().is_empty());
    // What is inside both bounds is taken, and reaches the backend as the value it is and
    // not as it was spelled.
    assert_eq!(post(&server, nested(126).as_bytes()).status, 200);
    let raw = post(
        &server,
        br#"{"model":"local","state":{"n":1.2300e+04,"big":9007199254740993,"zero":-0},"questions":{"q":{"type":"noul"}}}"#,
    );
    assert_eq!(raw.status, 200);
    let state = &local.requests()[1].state;
    assert_eq!(state.to_string(), r#"{"n":12300.0,"big":9007199254740993,"zero":-0.0}"#);
}
