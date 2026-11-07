//! A proxy named in the environment: what goes through it, and what never does.
//!
//! The environment is read when a client is built, and it belongs to the whole process. So
//! each case here runs this test binary again, as a child with the environment the case is
//! about, and the child is the one that builds a client. Nothing leaves the machine: the proxy
//! is a server on 127.0.0.1 that refuses everything, and counts what it was asked.
// The servers these tests ask are on 127.0.0.1, and a WebAssembly module can listen on no port.
#![cfg(not(target_family = "wasm"))]

mod common;

use std::process::Command;

use common::{Reply, Upstream};
use one_system::{Client, Registry};
use systemone::{DecisionModel, Error, Question, Request};

/// Set, in a child, to the base URL its client is to ask.
const TARGET: &str = "ONE_SYSTEM_TEST_PROXY_TARGET";

const VARIABLES: [&str; 9] = [
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "NO_PROXY",
    "no_proxy",
    "REQUEST_METHOD",
];

fn request() -> Request {
    Request::new("PRIVATE_STATE_CANARY").question("q", Question::noul("?"))
}

/// Run one test of this file again, in a process with these variables and no others that
/// name a proxy. What it wrote is returned, to be shown if the parent's own checks fail.
fn child(test: &str, target: &str, environment: &[(&str, String)]) -> String {
    let mut command = Command::new(std::env::current_exe().expect("this test binary"));
    command.args([test, "--exact", "--nocapture", "--test-threads=1"]);
    for variable in VARIABLES {
        command.env_remove(variable);
    }
    command.env(TARGET, target);
    for (variable, value) in environment {
        command.env(variable, value);
    }
    let output = command.output().expect("the test binary runs");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{said}");
    said
}

#[test]
fn a_backend_on_this_machine_is_never_asked_through_a_proxy() {
    const NAME: &str = "a_backend_on_this_machine_is_never_asked_through_a_proxy";
    if let Ok(target) = std::env::var(TARGET) {
        // The child: a registry whose one backend is on this machine, as a privacy registry's is.
        let file = serde_json::json!({"name": "local-demo", "selector": "local", "backends": [
            {"id": "local", "base_url": target, "model": "laya-english", "api_key_env": "LOCAL_API_KEY", "description": "Laya, on this machine."}]});
        let gateway = Registry::from_json(&file.to_string(), |_| Err(std::io::ErrorKind::NotFound.into()))
            .unwrap()
            .gateway_with(|_| Some("sk-local-backend-key".into()))
            .unwrap();
        let response = gateway.decide(&request()).expect("the backend on this machine answers");
        assert_eq!(response.model, "laya-english");
        return;
    }
    for variable in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"] {
        let proxy = Upstream::start(|_, _| Reply::new(502, "the proxy could not reach it"));
        let local = Upstream::answering("laya-english", 7);
        let by_number = local.url();
        let by_name = by_number.replace("127.0.0.1", "localhost");
        for target in [by_number, by_name] {
            let said = child(NAME, &target, &[(variable, proxy.url())]);
            assert_eq!(proxy.hits(), 0, "{variable} sent {target} through the proxy\n{said}");
        }
        assert_eq!(local.hits(), 2, "{variable}");
    }
}

#[test]
fn a_server_elsewhere_is_asked_through_the_proxy_its_scheme_names() {
    const NAME: &str = "a_server_elsewhere_is_asked_through_the_proxy_its_scheme_names";
    if let Ok(target) = std::env::var(TARGET) {
        // The child: the result is written out, for the parent to read.
        match Client::new(target).model("m").key("sk-for-elsewhere").decide(&request()) {
            Err(Error::Transport(message)) => println!("TRANSPORT {message}"),
            other => println!("OTHER {other:?}"),
        }
        return;
    }
    // `example.invalid` is a name nobody has, and a proxy is asked without the name being
    // looked up here. So nothing in this test asks the network for anything: every case is
    // one that goes to the proxy. Which cases do not is settled where it is decided, in the
    // client's own tests, without a request being made.
    let proxy = Upstream::start(|_, _| Reply::new(502, "the proxy could not reach it"));
    let through = |target: &str, environment: &[(&str, String)]| {
        let before = proxy.hits();
        let said = child(NAME, target, environment);
        assert!(said.contains("TRANSPORT"), "{said}");
        assert!(!said.contains("sk-for-elsewhere"), "{said}");
        proxy.hits() - before
    };
    let http = "http://example.invalid";
    let https = "https://example.invalid";
    let nobody = || "http://127.0.0.1:1".to_string();
    // Each scheme has its own variable, and does not use the other's.
    assert_eq!(through(http, &[("HTTP_PROXY", proxy.url()), ("HTTPS_PROXY", nobody())]), 1);
    assert_eq!(through(https, &[("HTTPS_PROXY", proxy.url()), ("HTTP_PROXY", nobody())]), 1);
    assert_eq!(through(https, &[("https_proxy", proxy.url())]), 1);
    // A proxy for everything is for a scheme that names none of its own.
    assert_eq!(through(https, &[("ALL_PROXY", proxy.url()), ("HTTP_PROXY", nobody())]), 1);
    assert_eq!(through(https, &[("ALL_PROXY", nobody()), ("HTTPS_PROXY", proxy.url())]), 1);
    // A host the environment exempts is not this one.
    assert_eq!(
        through(
            https,
            &[("HTTPS_PROXY", proxy.url()), ("NO_PROXY", "other.invalid,localhost".into())]
        ),
        1
    );
    // The state went to no proxy, and neither did the key: a proxy is asked only to connect.
    assert_eq!(proxy.hits(), 6);
    for seen in proxy.seen() {
        assert_eq!(seen.method, "CONNECT");
        assert!(!String::from_utf8_lossy(&seen.body).contains("PRIVATE_STATE_CANARY"));
        assert_eq!(seen.header("authorization"), None);
    }
}
