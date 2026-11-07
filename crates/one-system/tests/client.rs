//! The client, against servers that are functions on 127.0.0.1. Nothing here leaves the machine.
// The servers these tests ask are on 127.0.0.1, and a WebAssembly module can listen on no port.
#![cfg(not(target_family = "wasm"))]

mod common;

use std::net::TcpListener;
use std::time::{Duration, Instant};

use common::{Reply, Upstream, api, body};
use one_system::{Backend, Client, Retry};
use serde_json::json;
use systemone::{DecisionModel, Error, Question, Request};

const KEY: &str = "sk-test-SECRET-0123456789";

fn request() -> Request {
    Request::new(json!({"inputs": {"message": "I was charged twice."}}))
        .question("refund", Question::noul("Is the customer in `inputs.message` asking for a refund?"))
        .question(
            "team",
            Question::choice(
                "Which team?",
                [("billing", "Payment issue".to_string()), ("support", "Anything else".to_string())],
            ),
        )
        .question("urgency", Question::score("How urgent?", ["whenever", "this week", "today"]))
}

/// Retries that do not make a test wait.
fn quick() -> Retry {
    Retry {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(4),
        ..Retry::default()
    }
}

#[test]
fn a_request_goes_out_and_its_answers_come_back() {
    let server = Upstream::answering("jev-1.13.0", 291);
    let client = Client::new(server.url()).model("jev-latest").key(KEY);
    assert_eq!(client.name(), "jev-latest");

    let response = client.decide(&request()).unwrap();
    assert_eq!(response.model, "jev-1.13.0");
    assert_eq!(response.usage.input_tokens, 291);
    assert_eq!(response.answers.keys().collect::<Vec<_>>(), ["refund", "team", "urgency"]);
    assert_eq!(response.answers["refund"].noul, Some(0.9));
    assert_eq!(response.answers["team"].probabilities.as_ref().unwrap()["billing"], 1.0);

    let seen = server.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!((seen[0].method.as_str(), seen[0].url.as_str()), ("POST", "/v1/systemone"));
    assert_eq!(seen[0].header("authorization").unwrap(), format!("Bearer {KEY}"));
    assert_eq!(seen[0].header("content-type"), Some("application/json"));
    assert_eq!(seen[0].header("accept"), Some("application/json"));
    // The body is sent whole, with its length, and not in chunks: a local Laya refuses chunks.
    assert_eq!(seen[0].header("content-length"), Some(seen[0].body.len().to_string().as_str()));
    assert_eq!(seen[0].header("transfer-encoding"), None);
    // The client's model is named, since the request named none, and the rest is as given.
    assert_eq!(seen[0].request(), request().model("jev-latest"));
    assert_eq!(
        seen[0].json()["questions"]["team"]["criteria"],
        json!({"billing": "Payment issue", "support": "Anything else"})
    );
}

#[test]
fn a_model_the_request_names_is_the_one_asked_for() {
    let server = Upstream::answering("laya-english", 0);
    let client = Client::new(server.url()).model("jev-latest");
    client.decide(&request().model("local")).unwrap();
    assert_eq!(server.seen()[0].json()["model"], "local");
}

#[test]
fn a_local_server_is_sent_no_key_and_no_model() {
    let server = Upstream::answering("ggml-org/Laya-GGUF", 512);
    // A `/` at the end of the base makes no difference.
    let client = Client::local(format!("{}/", server.url()));
    assert_eq!(client.name(), server.url());
    assert_eq!(client.base_url(), server.url());
    let response = client.decide(&request()).unwrap();
    assert_eq!(response.model, "ggml-org/Laya-GGUF");
    let seen = server.seen();
    assert_eq!(seen[0].url, "/v1/systemone");
    assert_eq!(seen[0].header("authorization"), None);
    assert!(seen[0].json().get("model").is_none());
}

#[test]
fn a_path_in_the_base_is_kept() {
    let server = Upstream::answering("jev", 0);
    let client = Client::new(format!("{}/api/tenant%2Fteam", server.url())).key(KEY);
    client.decide(&request()).unwrap();
    assert_eq!(server.seen()[0].url, "/api/tenant%2Fteam/v1/systemone");
}

#[test]
fn the_hosted_constructors_name_their_servers() {
    let typesafe = format!("{:?}", Client::typesafe());
    assert!(typesafe.contains("https://api.typesafe.ai") && typesafe.contains("jev-latest") && typesafe.contains("$TYPESAFE_API_KEY"));
    let openrouter = format!("{:?}", Client::openrouter());
    assert!(
        openrouter.contains("https://openrouter.ai/api")
            && openrouter.contains("~typesafe/jev-latest")
            && openrouter.contains("$OPENROUTER_API_KEY")
    );
    assert_eq!(Client::typesafe().name(), "jev-latest");
}

#[test]
fn each_error_body_is_read() {
    let cases = [
        // A validation error from TypeSafe: where it was, and nothing of the input it echoes.
        (
            422,
            r#"{"detail":[{"type":"missing","loc":["body","questions","team","choice","criteria"],"msg":"Field required","input":{"type":"choice","instructions":"PRIVATE"}}]}"#,
            ("missing", "Field required (body.questions.team.choice.criteria)"),
        ),
        // Every error of a one-system gateway.
        (
            422,
            r#"{"detail":[{"loc":["body"],"msg":"Unknown model; see GET /v1/models","type":"unsupported_model"}]}"#,
            ("unsupported_model", "Unknown model; see GET /v1/models"),
        ),
        // Hosted TypeSafe's 400, 401 and 403.
        (
            401,
            r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}"#,
            (
                "authentication_error",
                "Cannot authenticate with the server. Please check your API key and try again.",
            ),
        ),
        (
            400,
            r#"{"detail":{"error_type":"api_usage_error","message":"Unknown model: no-such-model"}}"#,
            ("api_usage_error", "Unknown model: no-such-model"),
        ),
        // llama.cpp.
        (
            400,
            r#"{"error":{"code":400,"message":"request (9000 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error"}}"#,
            (
                "exceed_context_size_error",
                "request (9000 tokens) exceeds the available context size (2048 tokens), try increasing it",
            ),
        ),
        (
            501,
            r#"{"error":{"code":501,"message":"This model is not a decision model","type":"not_supported_error"}}"#,
            ("not_supported_error", "This model is not a decision model"),
        ),
        // Something that is not any of them.
        (502, "<html>Bad Gateway</html>", ("", "<html>Bad Gateway</html>")),
        (529, "", ("", "The server answered with status 529.")),
    ];
    for (status, text, (code, message)) in cases {
        let server = Upstream::start(move |_, _| Reply::new(status, text));
        let error = Client::new(server.url()).key(KEY).decide(&request()).unwrap_err();
        assert_eq!(api(&error), (status, code.to_string(), message.to_string()));
        assert!(!error.to_string().contains("PRIVATE"));
        assert_eq!(error.is_retryable(), status >= 500);
        // Nothing was asked twice.
        assert_eq!(server.hits(), 1);
    }
}

#[test]
fn a_server_that_takes_too_long_is_a_timeout() {
    let server = Upstream::start(|seen, _| Reply::ok(body(&seen.request(), "slow", 0)).after(Duration::from_millis(600)));
    let client = Client::new(server.url()).timeout(Duration::from_millis(100));
    let started = Instant::now();
    let error = client.decide(&request()).unwrap_err();
    assert!(started.elapsed() < Duration::from_millis(550), "it waited {:?}", started.elapsed());
    assert!(
        matches!(&error, Error::Transport(message) if message.contains("did not answer within 100ms")),
        "{error:?}"
    );
    assert!(error.is_retryable());
    // Given the time, the same server answers.
    assert!(
        Client::new(server.url())
            .timeout(Duration::from_secs(20))
            .decide(&request())
            .is_ok()
    );
}

#[test]
fn a_server_that_is_not_there_is_a_transport_error() {
    // A port that was free a moment ago, with nobody listening on it now.
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let client = Client::new(format!("http://127.0.0.1:{port}"))
        .key(KEY)
        .timeout(Duration::from_secs(5));
    let error = client.decide(&request()).unwrap_err();
    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    assert!(!error.to_string().contains(KEY));
    // With retries on it is tried again, and fails the same way.
    let error = client.retry(quick()).decide(&request()).unwrap_err();
    assert!(matches!(error, Error::Transport(_)));
}

#[test]
fn nothing_is_retried_unless_asked() {
    for status in [408, 429, 500, 503, 529] {
        let server = Upstream::start(move |_, _| Reply::new(status, ""));
        let error = Client::new(server.url()).decide(&request()).unwrap_err();
        assert_eq!(error.status(), Some(status));
        assert_eq!(server.hits(), 1);
    }
}

#[test]
fn with_retries_on_a_failure_that_may_pass_is_asked_again() {
    // Twice unlucky, then answered.
    for status in [408, 429, 500, 503, 529] {
        let server = Upstream::start(move |seen, before| {
            if before < 2 {
                Reply::new(status, r#"{"detail":{"error_type":"overloaded","message":"busy"}}"#)
            } else {
                Reply::ok(body(&seen.request(), "jev-1.13.0", 291))
            }
        });
        let response = Client::new(server.url()).retry(quick()).decide(&request()).unwrap();
        assert_eq!(response.model, "jev-1.13.0");
        assert_eq!(server.hits(), 3);
        // Each attempt sent the same request.
        let seen = server.seen();
        assert!(seen.iter().all(|attempt| attempt.body == seen[0].body));
    }

    // Unlucky every time: two retries, three requests, and the last error is the one returned.
    let server = Upstream::start(|_, before| {
        Reply::new(
            503,
            format!(r#"{{"detail":{{"error_type":"overloaded","message":"attempt {before}"}}}}"#),
        )
    });
    let error = Client::new(server.url()).retry(quick()).decide(&request()).unwrap_err();
    assert_eq!(api(&error), (503, "overloaded".into(), "attempt 2".into()));
    assert_eq!(server.hits(), 3);

    // As many retries as asked for.
    let server = Upstream::start(|_, _| Reply::new(503, ""));
    let patient = Retry { max_retries: 4, ..quick() };
    assert!(Client::new(server.url()).retry(patient).decide(&request()).is_err());
    assert_eq!(server.hits(), 5);

    // A timeout is retried too.
    let server = Upstream::start(|seen, before| {
        let wait = if before == 0 { Duration::from_millis(400) } else { Duration::ZERO };
        Reply::ok(body(&seen.request(), "jev", 0)).after(wait)
    });
    let client = Client::new(server.url()).timeout(Duration::from_millis(150)).retry(Retry {
        max_retries: 4,
        initial: Duration::from_millis(150),
        ..quick()
    });
    assert!(client.decide(&request()).is_ok());
    assert!(server.hits() >= 2);
}

#[test]
fn a_failure_that_will_not_pass_is_not_asked_again() {
    for status in [400, 401, 403, 404, 422] {
        let server = Upstream::start(move |_, _| Reply::new(status, ""));
        let error = Client::new(server.url()).retry(quick()).decide(&request()).unwrap_err();
        assert_eq!(error.status(), Some(status));
        assert_eq!(server.hits(), 1);
    }
    // Answers that are not for the questions will not get better either.
    let server = Upstream::start(|_, _| Reply::ok(r#"{"model":"m","answers":{},"usage":{"input_tokens":0,"output_tokens":0}}"#));
    assert!(matches!(
        Client::new(server.url()).retry(quick()).decide(&request()),
        Err(Error::Evidence(_))
    ));
    assert_eq!(server.hits(), 1);
}

#[test]
fn the_wait_a_server_asks_for_is_the_wait() {
    let limited = |header: &'static str, value: &'static str| {
        Upstream::start(move |seen, before| {
            if before == 0 {
                Reply::new(429, "").header(header, value)
            } else {
                Reply::ok(body(&seen.request(), "jev", 0))
            }
        })
    };
    // The client's own wait would be a millisecond.
    for (header, value, least) in [("retry-after-ms", "300", 300), ("retry-after", "0.25", 250)] {
        let server = limited(header, value);
        let started = Instant::now();
        assert!(Client::new(server.url()).retry(quick()).decide(&request()).is_ok());
        assert!(
            started.elapsed() >= Duration::from_millis(least),
            "{header}: it waited {:?}",
            started.elapsed()
        );
        assert_eq!(server.hits(), 2);
    }
    // A server that asks for more than is honoured gets the client's own wait.
    let server = limited("retry-after", "3600");
    let started = Instant::now();
    assert!(Client::new(server.url()).retry(quick()).decide(&request()).is_ok());
    assert!(started.elapsed() < Duration::from_secs(30));
}

#[test]
fn answers_that_are_not_for_the_questions_are_refused() {
    let good: serde_json::Value = serde_json::from_str(&body(&request(), "m", 0)).unwrap();
    let altered = |change: fn(&mut serde_json::Value)| {
        let mut response = good.clone();
        change(&mut response);
        response.to_string()
    };
    let bad = [
        // An answer missing, and an answer nobody asked for.
        altered(|response| {
            response["answers"].as_object_mut().unwrap().shift_remove("urgency");
        }),
        altered(|response| response["answers"]["extra"] = json!({"type": "noul", "noul": 0.5})),
        // An answer of another kind, or of none.
        altered(|response| {
            response["answers"]["refund"] = json!({"type": "choice", "choice": "a", "confidence": 1.0, "probabilities": {"a": 1.0}})
        }),
        altered(|response| response["answers"]["refund"] = json!({"noul": 0.5})),
        altered(|response| response["answers"]["refund"] = json!({"type": "noul"})),
        // Probabilities for options that were not declared, or not for all that were.
        altered(|response| response["answers"]["team"]["probabilities"] = json!({"billing": 0.5, "sales": 0.5})),
        altered(|response| response["answers"]["team"]["probabilities"] = json!({"billing": 1.0})),
        altered(|response| response["answers"]["team"]["choice"] = json!("sales")),
        altered(|response| response["answers"]["urgency"]["probabilities"] = json!({"0": 0.5, "1": 0.5})),
        altered(|response| {
            response["answers"]["urgency"]
                .as_object_mut()
                .unwrap()
                .shift_remove("probabilities");
        }),
        // Not a response at all.
        altered(|response| {
            response.as_object_mut().unwrap().shift_remove("answers");
        }),
        altered(|response| response["answers"] = json!([])),
        altered(|response| response["usage"]["input_tokens"] = json!(-1)),
        altered(|response| response["usage"]["input_tokens"] = json!(1.5)),
        "not json".to_string(),
        r#"{"model":"m","answers":{"#.to_string(),
        String::new(),
    ];
    for text in bad {
        let shown = text.clone();
        let server = Upstream::start(move |_, _| Reply::ok(text.clone()));
        let error = Client::new(server.url()).decide(&request()).unwrap_err();
        assert!(matches!(error, Error::Evidence(_)), "{shown} gave {error:?}");
        assert!(!error.is_retryable());
    }
    // What a server sends that the protocol does not name is let through: more usage, an id.
    let more = altered(|response| {
        response["answers"]["urgency"]["legend"] = json!({"0": "whenever", "1": "this week", "2": "today"});
        response["usage"]["cached_tokens"] = json!(12);
        response["request_id"] = json!("req_1");
    });
    let server = Upstream::start(move |_, _| Reply::ok(more.clone()));
    assert!(Client::new(server.url()).decide(&request()).is_ok());

    // Exactly 8 MiB is a response that fits.
    let padded = format!("{}{}", good, " ".repeat(one_system::MAX_BODY_BYTES - good.to_string().len()));
    assert_eq!(padded.len(), one_system::MAX_BODY_BYTES);
    let server = Upstream::start(move |_, _| Reply::ok(padded.clone()));
    assert!(Client::new(server.url()).decide(&request()).is_ok());
}

#[test]
fn a_redirect_is_not_followed() {
    let elsewhere = Upstream::answering("elsewhere", 0);
    let target = format!("{}/v1/systemone", elsewhere.url());
    for status in [301, 302, 307, 308] {
        let target = target.clone();
        let server = Upstream::start(move |_, _| Reply::new(status, "").header("Location", &target));
        let error = Client::new(server.url()).key(KEY).retry(quick()).decide(&request()).unwrap_err();
        assert_eq!(
            api(&error),
            (
                status,
                String::new(),
                "The server answered with a redirect, which is not followed.".into()
            )
        );
        assert_eq!(server.hits(), 1);
    }
    // The key went nowhere it was not sent.
    assert_eq!(elsewhere.hits(), 0);
}

#[test]
fn a_response_too_large_is_not_read() {
    let server = Upstream::start(|_, _| Reply::ok(" ".repeat(one_system::MAX_BODY_BYTES + 1)));
    let error = Client::new(server.url()).decide(&request()).unwrap_err();
    assert!(matches!(&error, Error::Evidence(message) if message.contains("8 MiB")), "{error:?}");
    // A refusal too large to read is still the refusal it was, without its words.
    let server = Upstream::start(|_, _| Reply::new(503, "x".repeat(one_system::MAX_BODY_BYTES + 1)));
    let error = Client::new(server.url()).decide(&request()).unwrap_err();
    assert_eq!(api(&error), (503, String::new(), "The server answered with status 503.".into()));
}

#[test]
fn the_key_is_in_no_error_and_no_debug_output() {
    let client = |url: String| Client::new(url).model("jev-latest").key(KEY);
    assert!(!format!("{:?}", client("http://127.0.0.1:1".into())).contains(KEY));
    assert!(format!("{:?}", client("http://127.0.0.1:1".into())).contains("[given]"));

    // A server that says the key back, in each of the bodies it could.
    let echoes = [
        format!(r#"{{"detail":[{{"loc":["body","{KEY}"],"msg":"bad key {KEY}","type":"{KEY}"}}]}}"#),
        format!(r#"{{"detail":{{"error_type":"{KEY}","message":"Bearer {KEY} is not valid"}}}}"#),
        format!(r#"{{"error":{{"code":401,"message":"Invalid API Key {KEY}","type":"{KEY}"}}}}"#),
        format!("<html>{KEY}</html>"),
    ];
    for echo in echoes {
        let server = Upstream::start(move |seen, _| {
            assert_eq!(seen.header("authorization").unwrap(), format!("Bearer {KEY}"));
            Reply::new(401, echo.clone())
        });
        let error = client(server.url()).decide(&request()).unwrap_err();
        assert_eq!(error.status(), Some(401));
        assert!(!error.to_string().contains(KEY), "{error}");
        assert!(!format!("{error:?}").contains(KEY), "{error:?}");
    }

    // Nor when the response is nonsense, the connection fails, or the time runs out.
    let nonsense = Upstream::start(|_, _| Reply::ok(format!("{{\"model\": \"{KEY}\"")));
    let slow = Upstream::start(|_, _| Reply::ok("").after(Duration::from_millis(300)));
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let failures = [
        client(nonsense.url()).decide(&request()),
        client(slow.url()).timeout(Duration::from_millis(50)).decide(&request()),
        client(format!("http://127.0.0.1:{port}")).decide(&request()),
    ];
    for failure in failures {
        let error = failure.unwrap_err();
        let shown = format!("{error} {error:?}");
        assert!(!shown.contains("SECRET"), "{shown}");
    }
}

#[test]
fn a_key_that_could_not_be_a_key_is_refused_before_anything_is_sent() {
    let server = Upstream::answering("jev", 0);
    for bad in ["", "   ", "two words", "tab\tbed", "clé-secrète"] {
        let error = Client::new(server.url()).key(bad).decide(&request()).unwrap_err();
        assert!(matches!(error, Error::Invalid(_)), "{error:?}");
        assert!(bad.trim().is_empty() || !error.to_string().contains(bad));
    }
    // Space round a key, as a file read whole would leave, is not part of it.
    Client::new(server.url()).key(format!("  {KEY}\n")).decide(&request()).unwrap();
    assert_eq!(server.hits(), 1);
    assert_eq!(server.seen()[0].header("authorization").unwrap(), format!("Bearer {KEY}"));
    // A key can be taken away again.
    Client::new(server.url()).key(KEY).no_key().decide(&request()).unwrap();
    assert_eq!(server.seen()[1].header("authorization"), None);
}

#[test]
fn a_key_is_read_from_the_variable_named_at_each_call() {
    let server = Upstream::answering("jev", 0);
    // A variable nobody sets: the error names the variable, and nothing is sent.
    let client = Client::new(server.url()).key_env("ONE_SYSTEM_TEST_VARIABLE_THAT_IS_NOT_SET");
    let error = client.decide(&request()).unwrap_err();
    assert_eq!(
        error,
        Error::Invalid("The environment variable ONE_SYSTEM_TEST_VARIABLE_THAT_IS_NOT_SET holds no API key.".into())
    );
    assert_eq!(server.hits(), 0);
    assert!(format!("{client:?}").contains("$ONE_SYSTEM_TEST_VARIABLE_THAT_IS_NOT_SET"));

    // A variable this process does have, of the ones that hold no secret. (A test does not set
    // one: the environment belongs to every test at once.) Its value is compared, never shown.
    let usable = |value: &String| !value.is_empty() && value.chars().all(|character| character.is_ascii_graphic());
    let Some((name, value)) = ["USER", "LOGNAME", "HOME", "SHELL", "LANG", "PWD"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(usable).map(|value| (name, value)))
    else {
        return;
    };
    Client::new(server.url()).key_env(name).decide(&request()).unwrap();
    let sent = server.seen()[0].header("authorization").unwrap().to_string();
    assert!(sent == format!("Bearer {value}"), "the key sent is not the value of ${name}");
}

#[test]
fn the_models_a_server_lists() {
    let server = Upstream::start(|seen, _| {
        assert_eq!((seen.method.as_str(), seen.url.as_str()), ("GET", "/v1/models"));
        Reply::ok(r#"{"models":[{"name":"jev-latest","description":"Jev","release_date":"2026-09-19"}]}"#)
    });
    let models = Client::new(server.url()).key(KEY).models().unwrap();
    assert_eq!(models["models"][0]["name"], "jev-latest");
    assert_eq!(server.seen()[0].header("authorization").unwrap(), format!("Bearer {KEY}"));

    let refusing = Upstream::start(|_, _| {
        Reply::new(
            403,
            r#"{"detail":{"error_type":"authentication_error","message":"Must supply an API key!"}}"#,
        )
    });
    let error = Client::new(refusing.url()).models().unwrap_err();
    assert_eq!(api(&error), (403, "authentication_error".into(), "Must supply an API key!".into()));
}

#[test]
fn a_client_is_a_decision_model_anywhere_one_is_wanted() {
    let server = Upstream::answering("jev", 5);
    let model: std::sync::Arc<dyn DecisionModel> = std::sync::Arc::new(Client::new(server.url()));
    // It can be shared between threads, and each asks for itself.
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| assert_eq!(model.decide(&request()).unwrap().usage.input_tokens, 5));
        }
    });
    assert_eq!(server.hits(), 4);
}

#[test]
fn a_key_a_server_repeats_is_taken_out_before_its_words_are_cut() {
    // Long enough to lie across the place where a server's words are cut off.
    const LONG: &str = "sk-live-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ";
    let filler = "x".repeat(440);
    let echoes = [
        (
            401,
            format!(r#"{{"error":{{"message":"{filler} invalid credential 'Bearer {LONG}'","type":"{filler}{filler}{LONG}"}}}}"#),
        ),
        (
            401,
            format!(r#"{{"detail":{{"error_type":"{filler}{filler}{LONG}","message":"{filler}{filler} {LONG}"}}}}"#),
        ),
        (403, format!("{filler} Authorization: Bearer {LONG}")),
        // At every place the cut could fall in it.
        (403, format!("{}{LONG}", "x".repeat(480))),
        (403, format!("{}{LONG}", "x".repeat(499))),
    ];
    for (status, echo) in echoes {
        let said = echo.clone();
        let server = Upstream::start(move |_, _| Reply::new(status, said.clone()));
        let client = Client::new(server.url()).model("jev-latest").key(LONG);
        for error in [client.decide(&request()).unwrap_err(), client.models().unwrap_err()] {
            assert_eq!(error.status(), Some(status));
            let shown = format!("{error} {error:?}");
            // Not the key, and not the start of it either.
            assert!(!shown.contains(&LONG[..12]), "{shown}");
            // Shown twice over, a code and a message each cut at 500 characters.
            assert!(shown.len() < 2400, "{shown}");
        }
    }
}

#[test]
fn credentials_written_into_a_base_url_are_not_kept_and_nothing_is_sent() {
    const PASSWORD: &str = "hunter2-password";
    let upstream = Upstream::answering("laya-english", 1);
    let urls = [
        format!("https://user:{PASSWORD}@example.invalid/api/"),
        format!("https://{PASSWORD}@example.invalid"),
        upstream.url().replacen("http://", &format!("http://user:{PASSWORD}@"), 1),
    ];
    for url in urls {
        let client = Client::new(url);
        let named = client.clone().model("jev-latest").key(KEY);
        let shown = [
            format!("{client:?}"),
            client.name(),
            client.base_url().to_string(),
            format!("{:?}", Backend::new("hosted", client.clone())),
            format!("{named:?}"),
        ];
        for text in shown {
            assert!(!text.contains(PASSWORD) && !text.contains("user"), "{text}");
        }
        // It is refused when it is used, in words that do not repeat it, and nothing is sent.
        for error in [
            client.decide(&request()).unwrap_err(),
            client.models().unwrap_err(),
            named.decide(&request()).unwrap_err(),
        ] {
            assert!(
                matches!(&error, Error::Invalid(message) if message.contains("base URL")),
                "{error:?}"
            );
            assert!(!format!("{error} {error:?}").contains(PASSWORD));
        }
    }
    assert_eq!(upstream.hits(), 0);
    assert_eq!(
        Client::new("https://example.invalid/api/").base_url(),
        "https://example.invalid/api"
    );
    // An at sign further on is part of a path, and no credential.
    assert_eq!(Client::new("https://example.invalid/a@b").base_url(), "https://example.invalid/a@b");
}

/// A server that answers every request well, in the HTTP it is told to speak. Returns its
/// address and a count of the connections it took.
///
/// `closes` is HTTP/1.0 as Python's `http.server` speaks it: the reply has a length, and the
/// connection is closed a moment after it. `says` is put where a `Connection` header goes. A
/// server that does not close takes the next request on the same connection.
fn speaking(version: &'static str, says: &'static str, closes: bool) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let listener = TcpListener::bind("127.0.0.1:0").expect("a port on this machine");
    let url = format!("http://{}", listener.local_addr().expect("its address"));
    let taken = std::sync::Arc::new(AtomicUsize::new(0));
    let connections = taken.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            connections.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().expect("the same connection, to read"));
                loop {
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        if let Some((name, value)) = line.split_once(':')
                            && name.eq_ignore_ascii_case("content-length")
                        {
                            length = value.trim().parse().unwrap_or(0);
                        }
                        if line.trim().is_empty() {
                            break;
                        }
                    }
                    let mut asked = vec![0; length];
                    if reader.read_exact(&mut asked).is_err() {
                        return;
                    }
                    let asked: Request = serde_json::from_slice(&asked).expect("a request");
                    let reply = body(&asked, "laya", 7);
                    let head = format!(
                        "{version} 200 OK\r\nContent-Type: application/json\r\n{says}Content-Length: {}\r\n\r\n",
                        reply.len()
                    );
                    let _ = stream.write_all(format!("{head}{reply}").as_bytes());
                    let _ = stream.flush();
                    if closes {
                        std::thread::sleep(Duration::from_millis(1));
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        return;
                    }
                }
            });
        }
    });
    (url, taken)
}

/// What failed of `each` requests on each of `width` threads.
fn failures(client: &Client, width: usize, each: usize) -> Vec<String> {
    let failed = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..width {
            scope.spawn(|| {
                for _ in 0..each {
                    if let Err(error) = client.decide(&request()) {
                        failed.lock().unwrap().push(error.to_string());
                    }
                }
            });
        }
    });
    failed.into_inner().unwrap()
}

#[test]
fn a_connection_an_http_1_0_server_will_close_is_not_used_again() {
    use std::sync::atomic::Ordering;

    // The server closes after each reply, as HTTP/1.0 has it unless a reply says otherwise. A
    // connection kept for the next request would be one the server has closed, or is closing.
    let (url, taken) = speaking("HTTP/1.0", "", true);
    let client = Client::local(&url);
    let failed = failures(&client, 8, 25);
    assert!(
        failed.is_empty(),
        "{} of 200 failed: {:?}",
        failed.len(),
        &failed[..failed.len().min(3)]
    );
    assert_eq!(
        taken.load(Ordering::SeqCst),
        200,
        "a connection for each request, and none used twice"
    );
    // One after another as well, where the close has had time to arrive or has not.
    let failed = failures(&client, 1, 20);
    assert!(
        failed.is_empty(),
        "{} of 20 failed: {:?}",
        failed.len(),
        &failed[..failed.len().min(3)]
    );
    assert_eq!(taken.load(Ordering::SeqCst), 220);

    // A reply that says it closes was never kept, and is not now.
    let (url, taken) = speaking("HTTP/1.0", "Connection: close\r\n", true);
    assert_eq!(failures(&Client::local(&url), 8, 5), Vec::<String>::new());
    assert_eq!(taken.load(Ordering::SeqCst), 40);
}

#[test]
fn a_connection_a_server_keeps_is_used_again() {
    use std::sync::atomic::Ordering;

    // HTTP/1.1 keeps a connection unless it says it closes it, and HTTP/1.0 where it says it keeps it.
    for (version, says) in [
        ("HTTP/1.1", ""),
        ("HTTP/1.0", "Connection: keep-alive\r\n"),
        ("HTTP/1.0", "connection: Keep-Alive\r\n"),
    ] {
        let (url, taken) = speaking(version, says, false);
        assert_eq!(failures(&Client::local(&url), 1, 10), Vec::<String>::new());
        assert_eq!(taken.load(Ordering::SeqCst), 1, "{version} {says}");
    }
}
