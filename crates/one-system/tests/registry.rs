//! Registry files: the Go gateway's own, where they are on this machine, and the rules a file is held to.
// The servers these tests ask are on 127.0.0.1, and a WebAssembly module can listen on no port.
#![cfg(not(target_family = "wasm"))]

mod common;

use std::path::PathBuf;

use common::{Reply, Upstream, body};
use one_system::{Capabilities, Client, ConfigError, Gateway, Limits, Registry, Rule};
use serde_json::{Value, json};
use systemone::{DecisionModel, Kind, Question, Request};

/// The Go gateway's repository (github.com/rawwerks/one-system), if `ONE_SYSTEM_GO_REPOSITORY`
/// says where a checkout of it is.
fn go_repository() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("ONE_SYSTEM_GO_REPOSITORY")?);
    root.join("backends.json").is_file().then_some(root)
}

/// A key for any variable a file names: no file names one that is not in upper case.
fn any_key(variable: &str) -> Option<String> {
    variable
        .chars()
        .all(|character| character.is_ascii_uppercase() || character == '_')
        .then(|| "sk-for-tests".to_string())
}

#[test]
fn the_go_gateways_own_registries_load_unchanged() {
    let Some(root) = go_repository() else {
        return;
    };
    let laya = Capabilities::all().max_questions(16).min_criteria(2).structured_state(true);
    let laya_limits = Limits::default()
        .max_characters(1200)
        .max_non_ascii_letter_fraction(0.02)
        .max_questions(16)
        .max_criteria(20);

    // Hosted Jev selects between local Laya and itself.
    let registry = Registry::load(root.join("backends.json")).unwrap();
    assert_eq!(
        (registry.name.as_str(), registry.selector.as_str(), &registry.fallback),
        ("routing-demo", "hosted", &None)
    );
    assert_eq!(registry.selection, None);
    let [local, hosted] = &registry.backends[..] else {
        panic!("two backends");
    };
    assert_eq!(
        (
            local.id.as_str(),
            local.base_url.as_str(),
            local.model.as_str(),
            local.api_key_env.as_str()
        ),
        ("local", "http://127.0.0.1:8091", "laya-english", "LOCAL_API_KEY")
    );
    assert_eq!(local.capabilities, Some(laya.clone()));
    assert_eq!(local.limits, Some(laya_limits.clone()));
    assert!(local.description.starts_with("Local CPU Laya"));
    assert_eq!(
        (
            hosted.id.as_str(),
            hosted.base_url.as_str(),
            hosted.model.as_str(),
            hosted.api_key_env.as_str()
        ),
        ("hosted", "https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY")
    );
    assert_eq!((&hosted.capabilities, &hosted.limits), (&None, &None));
    let gateway = registry.gateway_with(any_key).unwrap();
    assert_eq!(
        (gateway.name(), gateway.selector(), gateway.fallback()),
        ("routing-demo".to_string(), "hosted", None)
    );
    assert_eq!(
        gateway.backends().map(|backend| backend.id.as_str()).collect::<Vec<_>>(),
        ["hosted", "local"]
    );

    // Local Laya selects, by a question about the task, and a rule escalates to hosted Jev.
    let registry = Registry::load(root.join("examples/privacy.backends.json")).unwrap();
    assert_eq!(
        (registry.name.as_str(), registry.selector.as_str(), registry.fallback.as_deref()),
        ("privacy-demo", "local", Some("local"))
    );
    let selection = registry.selection.as_ref().unwrap();
    assert_eq!(selection.escalate_to, "hosted");
    assert_eq!(selection.rules, [Rule::choice("domain", "math_or_logic", 0.5)]);
    let domain = &selection.questions["domain"];
    assert_eq!(domain.kind, Kind::Choice);
    assert_eq!(
        domain.labels(),
        ["code", "math_or_logic", "writing", "factual_lookup", "data_analysis", "chitchat"]
    );
    assert!(registry.gateway_with(any_key).is_ok());

    // One backend, and no selecting to do.
    let registry = Registry::load(root.join("examples/local.backends.json")).unwrap();
    assert_eq!((registry.name.as_str(), registry.backends.len()), ("local-demo", 1));
    assert_eq!(registry.backends[0].capabilities, Some(laya));
    assert!(registry.gateway_with(any_key).is_ok());

    let registry = Registry::load(root.join("examples/jev-lint.backends.json")).unwrap();
    assert_eq!(
        (registry.name.as_str(), registry.backends[0].model.as_str()),
        ("jev-lint", "jev-1.13.0")
    );

    // A server that answers choices and scores, and no nouls.
    let registry = Registry::load(root.join("examples/simple-jev.backends.json")).unwrap();
    let declared = registry.backends[0].capabilities.clone().unwrap();
    assert_eq!(
        declared,
        Capabilities::new([Kind::Choice, Kind::Score]).min_criteria(2).max_criteria(50)
    );
    let gateway = registry.gateway_with(any_key).unwrap();
    let noul = Request::new("state").question("q", Question::noul("?"));
    // Refused here, before anything is sent anywhere.
    assert_eq!(gateway.decide(&noul).unwrap_err().status(), Some(422));

    // Every registry in the repository loads, whatever else is there.
    let mut found = vec![root.join("backends.json")];
    for entry in std::fs::read_dir(root.join("examples")).unwrap().flatten() {
        if entry.file_name().to_string_lossy().ends_with(".backends.json") {
            found.push(entry.path());
        }
    }
    assert!(found.len() >= 5);
    for path in found {
        let registry = Registry::load(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        registry
            .gateway_with(any_key)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
}

#[test]
fn a_backend_whose_key_is_not_set_is_an_error_whichever_it_is() {
    let registry = Registry::from_json(&two().to_string(), no_file).unwrap();
    let missing = "a backend api_key_env must reference a nonempty bearer key without whitespace";
    assert_eq!(registry.gateway_with(|_| None).unwrap_err().message(), missing);
    assert_eq!(
        registry
            .gateway_with(|variable| (variable == "LOCAL_API_KEY").then(|| "k".to_string()))
            .unwrap_err()
            .message(),
        missing
    );
    assert_eq!(
        registry.gateway_with(|_| Some("two words".to_string())).unwrap_err().message(),
        missing
    );
    assert_eq!(registry.gateway_with(|_| Some(String::new())).unwrap_err().message(), missing);
    assert!(registry.gateway_with(any_key).is_ok());
    // From the environment: these variables are named so that nobody has set them.
    let unset = two()
        .to_string()
        .replace("LOCAL_API_KEY", "ONE_SYSTEM_TEST_UNSET_LOCAL")
        .replace("TYPESAFE_API_KEY", "ONE_SYSTEM_TEST_UNSET_HOSTED");
    assert_eq!(
        Registry::from_json(&unset, no_file).unwrap().gateway().unwrap_err().message(),
        missing
    );
}

fn no_file(_: &str) -> std::io::Result<String> {
    Err(std::io::ErrorKind::NotFound.into())
}

/// A registry of two backends, to break one rule at a time.
fn two() -> Value {
    json!({
        "name": "demo",
        "selector": "hosted",
        "backends": [
            {"id": "local", "base_url": "http://127.0.0.1:8091", "model": "laya-english", "api_key_env": "LOCAL_API_KEY", "description": "Local."},
            {"id": "hosted", "base_url": "https://api.typesafe.ai", "model": "jev-latest", "api_key_env": "TYPESAFE_API_KEY", "description": "Hosted."},
        ],
    })
}

const QUESTIONS: &str = r#"{"domain": {"type": "choice", "instructions": "What domain?", "criteria": {"math": "sums", "chat": null}},
                            "private": {"type": "noul", "instructions": "Is it private?"}}"#;

fn load(registry: &Value) -> Result<Registry, ConfigError> {
    Registry::from_json(&registry.to_string(), |path| {
        if path == "routing.json" {
            Ok(QUESTIONS.to_string())
        } else {
            no_file(path)
        }
    })
}

/// What loading says is wrong, after a change to a registry that was right.
fn broken(change: impl FnOnce(&mut Value)) -> String {
    let mut registry = two();
    registry["fallback"] = json!("local");
    registry["selection"] = json!({"questions_file": "routing.json", "escalate_to": "hosted", "rules": [{"question": "domain", "choice": "math", "above": 0.5}]});
    assert!(load(&registry).is_ok());
    change(&mut registry);
    load(&registry).expect_err("the change breaks a rule").message().to_string()
}

#[test]
fn a_registry_is_held_to_the_rules_the_go_gateway_holds_it_to() {
    const SHAPE: &str = "backend configuration must contain name, selector, and backends";
    assert_eq!(Registry::from_json("not json", no_file).unwrap_err().message(), SHAPE);
    assert_eq!(Registry::from_json("[]", no_file).unwrap_err().message(), SHAPE);
    assert_eq!(
        Registry::from_json(&format!("{} {{}}", two()), no_file).unwrap_err().message(),
        "backend configuration must contain one JSON document"
    );

    // A member nobody knows, at each level.
    assert_eq!(broken(|registry| registry["port"] = json!(8090)), SHAPE);
    assert_eq!(broken(|registry| registry["backends"][0]["api_key"] = json!("sk-inline")), SHAPE);
    assert_eq!(
        broken(|registry| registry["backends"][0]["limits"] = json!({"max_tokens": 5})),
        SHAPE
    );
    assert_eq!(
        broken(|registry| registry["backends"][0]["capabilities"] = json!({"question_types": ["noul"], "images": true})),
        SHAPE
    );
    assert_eq!(broken(|registry| registry["selection"]["mode"] = json!("fast")), SHAPE);
    assert_eq!(broken(|registry| registry["selection"]["rules"][0]["below"] = json!(0.1)), SHAPE);
    // A member of the wrong kind.
    assert_eq!(
        broken(|registry| registry["backends"][0]["limits"] = json!({"max_characters": 1.5})),
        SHAPE
    );
    assert_eq!(broken(|registry| registry["backends"] = json!({})), SHAPE);

    let name = "configuration name must contain only ASCII letters, digits, hyphens, or underscores";
    assert_eq!(broken(|registry| registry["name"] = json!("")), name);
    assert_eq!(broken(|registry| registry["name"] = json!(null)), name);
    assert_eq!(broken(|registry| registry["name"] = json!("routing demo")), name);
    assert_eq!(
        broken(|registry| {
            registry.as_object_mut().unwrap().remove("name");
        }),
        name
    );
    assert_eq!(
        broken(|registry| registry["backends"] = json!([])),
        "backend configuration requires at least one backend"
    );
    assert_eq!(
        broken(|registry| registry["backends"][0]["id"] = json!("demo")),
        "backend ID must differ from the configuration name"
    );

    let required = "every backend requires id, model, description, and api_key_env";
    for member in ["id", "model", "description", "api_key_env"] {
        assert_eq!(broken(|registry| registry["backends"][0][member] = json!("")), required);
        assert_eq!(
            broken(|registry| {
                registry["backends"][0].as_object_mut().unwrap().remove(member);
            }),
            required
        );
    }
    assert_eq!(
        broken(|registry| registry["backends"][0]["id"] = json!("lo cal")),
        "backend IDs must contain only ASCII letters, digits, hyphens, or underscores"
    );
    assert_eq!(
        broken(|registry| registry["backends"][0]["id"] = json!("hosted")),
        "backend IDs must be unique"
    );

    let url = "backend base_url must use HTTP(S) without credentials, query, or fragment";
    for bad in [
        "",
        "api.typesafe.ai",
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://example.com/?key=1",
        "https://example.com/#x",
    ] {
        assert_eq!(broken(|registry| registry["backends"][1]["base_url"] = json!(bad)), url, "{bad}");
    }
    // A key is sent with every request, so plain HTTP is for this machine only.
    let https = "backend base_url requires HTTPS except for loopback hosts";
    assert_eq!(
        broken(|registry| registry["backends"][1]["base_url"] = json!("http://api.typesafe.ai")),
        https
    );
    assert_eq!(
        broken(|registry| registry["backends"][0]["base_url"] = json!("http://192.168.1.20:8091")),
        https
    );

    let limits = |limits: Value| broken(|registry| registry["backends"][0]["limits"] = limits);
    assert_eq!(
        limits(json!({"max_characters": 0})),
        "a backend max_characters must be a positive integer"
    );
    assert_eq!(
        limits(json!({"max_characters": -4})),
        "a backend max_characters must be a positive integer"
    );
    assert_eq!(
        limits(json!({"max_non_ascii_letter_fraction": 1.2})),
        "a backend max_non_ascii_letter_fraction must fall between 0 and 1"
    );
    assert_eq!(
        limits(json!({"max_questions": 0})),
        "a backend max_questions must be a positive integer"
    );
    assert_eq!(limits(json!({"max_criteria": 1})), "a backend max_criteria must be at least 2");

    let capabilities = |capabilities: Value| broken(|registry| registry["backends"][0]["capabilities"] = capabilities);
    assert_eq!(capabilities(json!({})), "capabilities require nonempty question_types");
    assert_eq!(
        capabilities(json!({"question_types": []})),
        "capabilities require nonempty question_types"
    );
    assert_eq!(
        capabilities(json!({"question_types": ["noul", "noul"]})),
        "capabilities require unique native question types"
    );
    assert_eq!(
        capabilities(json!({"question_types": ["essay"]})),
        "capabilities require unique native question types"
    );
    assert_eq!(
        capabilities(json!({"question_types": ["Noul"]})),
        "capabilities require unique native question types"
    );
    assert_eq!(
        capabilities(json!({"question_types": ["noul"], "max_questions": 0})),
        "capability limits must be positive integers"
    );
    assert_eq!(
        capabilities(json!({"question_types": ["noul"], "min_criteria": 3, "max_criteria": 2})),
        "min_criteria must not exceed max_criteria"
    );

    assert_eq!(
        broken(|registry| registry["selector"] = json!("elsewhere")),
        "selector must name a configured backend ID"
    );
    assert_eq!(
        broken(|registry| {
            registry.as_object_mut().unwrap().remove("selector");
        }),
        "selector must name a configured backend ID"
    );
    assert_eq!(
        broken(|registry| registry["fallback"] = json!("elsewhere")),
        "fallback must name a configured backend ID"
    );
    assert_eq!(
        broken(|registry| registry["escalation_confidence"] = json!(1.01)),
        "escalation_confidence must fall between 0 and 1"
    );
    let without_selection = |change: fn(&mut Value)| {
        broken(|registry| {
            registry.as_object_mut().unwrap().remove("selection");
            change(registry);
        })
    };
    assert_eq!(
        without_selection(|registry| {
            registry["fallback"] = json!(null);
            registry["escalation_confidence"] = json!(0.8);
        }),
        "escalation_confidence requires a fallback backend ID"
    );

    assert_eq!(
        broken(|registry| registry["fallback"] = json!("")),
        "selection requires a fallback backend ID"
    );
    assert_eq!(
        broken(|registry| registry["selection"]["escalate_to"] = json!("elsewhere")),
        "selection escalate_to must name a configured backend ID"
    );
    assert_eq!(
        broken(|registry| registry["selection"]["escalate_to"] = json!("local")),
        "selection escalate_to must differ from the fallback backend"
    );
    assert_eq!(
        broken(|registry| registry["selection"]["rules"] = json!([])),
        "selection requires at least one rule"
    );
    assert_eq!(
        broken(|registry| registry["selection"]["questions_file"] = json!("elsewhere.json")),
        "could not read selection questions_file"
    );
    let rule = |rule: Value| broken(|registry| registry["selection"]["rules"][0] = rule);
    assert_eq!(
        rule(json!({"question": "weather", "above": 0.5})),
        "every selection rule must name a question from questions_file"
    );
    assert_eq!(
        rule(json!({"question": "domain", "choice": "poetry", "above": 0.5})),
        "every Choice selection rule must name a configured criterion"
    );
    assert_eq!(
        rule(json!({"question": "domain", "above": 0.5})),
        "every Choice selection rule must name a configured criterion"
    );
    assert_eq!(
        rule(json!({"question": "domain", "choice": "math", "above": 1.5})),
        "every selection rule threshold must fall between 0 and 1"
    );
    assert_eq!(
        rule(json!({"question": "private", "above": -0.1})),
        "every selection rule threshold must fall between 0 and 1"
    );

    // Nothing a file holds is ever said back.
    let said = broken(|registry| registry["backends"][0]["base_url"] = json!("https://user:hunter2@example.com"));
    assert!(!said.contains("hunter2"));
}

#[test]
fn a_questions_file_is_a_map_of_questions() {
    let with = |questions: &'static str| {
        let mut registry = two();
        registry["fallback"] = json!("local");
        registry["selection"] =
            json!({"questions_file": "q.json", "escalate_to": "hosted", "rules": [{"question": "private", "above": 0.5}]});
        Registry::from_json(&registry.to_string(), move |_| Ok(questions.to_string())).map_err(|error| error.message().to_string())
    };
    let object = "selection questions_file must contain a SystemOne questions object";
    let valid = "selection questions_file must contain valid SystemOne questions";
    assert_eq!(with("not json").unwrap_err(), object);
    assert_eq!(with("[]").unwrap_err(), object);
    assert_eq!(with("{}").unwrap_err(), object);
    // Every question in the file is checked, whether a rule reads it or not.
    assert_eq!(
        with(r#"{"private": {"type": "noul"}, "other": {"type": "essay"}}"#).unwrap_err(),
        valid
    );
    assert_eq!(
        with(r#"{"private": {"type": "noul"}, "other": {"type": "score", "criteria": []}}"#).unwrap_err(),
        valid
    );
    assert_eq!(
        with(r#"{"private": {"type": "noul"}, "other": {"type": "choice"}}"#).unwrap_err(),
        valid
    );
    let registry =
        with(r#"{"private": {"type": "noul", "instructions": "Is it private?"}, "level": {"type": "score", "criteria": ["a", "b"]}}"#)
            .unwrap();
    let selection = registry.selection.unwrap();
    assert_eq!(selection.questions.keys().collect::<Vec<_>>(), ["private", "level"]);
    assert_eq!(selection.rules, [Rule::above("private", 0.5)]);
}

#[test]
fn a_registry_file_finds_its_questions_from_where_it_is() {
    let directory = std::env::temp_dir().join(format!("one-system-registry-test-{}", std::process::id()));
    let examples = directory.join("examples");
    std::fs::create_dir_all(&examples).unwrap();
    let mut registry = two();
    registry["fallback"] = json!("local");
    // Named from the directory above the registry's, as the Go gateway's examples are.
    registry["selection"] = json!({"questions_file": "examples/routing.questions.json", "escalate_to": "hosted", "rules": [{"question": "private", "above": 0.5}]});
    std::fs::write(examples.join("demo.backends.json"), registry.to_string()).unwrap();
    std::fs::write(examples.join("routing.questions.json"), QUESTIONS).unwrap();

    let loaded = Registry::load(examples.join("demo.backends.json")).unwrap();
    assert_eq!(loaded.selection.unwrap().questions.len(), 2);
    assert_eq!(loaded.backends[0].base_url, "http://127.0.0.1:8091");

    // A file that is not there, and one too large to be a registry.
    assert_eq!(
        Registry::load(directory.join("absent.json")).unwrap_err().message(),
        "could not open ONE_SYSTEM_CONFIG"
    );
    assert_eq!(
        Gateway::load(directory.join("absent.json")).unwrap_err().message(),
        "could not open ONE_SYSTEM_CONFIG"
    );
    let padded = format!("{}{}", registry, " ".repeat(one_system::MAX_CONFIG_BYTES));
    std::fs::write(directory.join("large.json"), padded).unwrap();
    assert_eq!(
        Registry::load(directory.join("large.json")).unwrap_err().message(),
        "could not open ONE_SYSTEM_CONFIG"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn a_gateway_from_a_registry_asks_the_servers_the_file_names() {
    // Two servers on this machine, each expecting its own key.
    let laya = Upstream::answering("laya-english", 12);
    let jev = Upstream::start(|seen, _| {
        let request = seen.request();
        let mut response: Value = serde_json::from_str(&body(&request, "jev-1.13.0", 291)).unwrap();
        // Asked to choose a backend, it chooses the local one.
        if request.questions.contains_key("backend") {
            response["answers"]["backend"] =
                json!({"type": "choice", "choice": "local", "confidence": 0.97, "probabilities": {"hosted": 0.01, "local": 0.99}});
        }
        Reply::ok(response.to_string())
    });
    let registry = json!({
        "name": "routing-demo",
        "selector": "hosted",
        "backends": [
            {"id": "local", "base_url": format!("{}/", laya.url()), "model": "laya-english", "api_key_env": "LOCAL_API_KEY", "description": "Local Laya.",
             "limits": {"max_characters": 1200}, "capabilities": {"question_types": ["choice", "score", "noul"], "max_questions": 16}},
            // Plain HTTP is allowed here because it is this machine.
            {"id": "hosted", "base_url": jev.url(), "model": "jev-latest", "api_key_env": "TYPESAFE_API_KEY", "description": "Hosted Jev."},
        ],
    });
    let registry = Registry::from_json(&registry.to_string(), no_file).unwrap();
    let gateway = registry
        .gateway_with(|variable| match variable {
            "LOCAL_API_KEY" => Some("sk-local".to_string()),
            "TYPESAFE_API_KEY" => Some("sk-hosted".to_string()),
            _ => None,
        })
        .unwrap();
    // No key is anywhere a gateway can be printed from.
    let shown = format!("{gateway:?} {:?} {registry:?}", gateway.backends().collect::<Vec<_>>());
    assert!(!shown.contains("sk-local") && !shown.contains("sk-hosted"));

    let request =
        Request::new("Please refund the duplicate charge.").question("refund", Question::noul("Does the customer request a refund?"));
    let routed = gateway.route(&request.clone().model("routing-demo"));
    let response = routed.result.unwrap();
    assert_eq!(response.model, "laya-english");
    assert_eq!(response.usage.input_tokens, 303);
    assert_eq!(routed.route.picked, Some(one_system::Picked::Selected { confidence: 0.97 }));

    // The selector was asked under its own model's name and with its own key, and saw no state.
    let selection = &jev.seen()[0];
    assert_eq!(selection.header("authorization"), Some("Bearer sk-hosted"));
    assert_eq!(selection.json()["model"], "jev-latest");
    assert_eq!(
        selection.json()["questions"]["backend"]["criteria"],
        json!({"hosted": "Hosted Jev.", "local": "Local Laya."})
    );
    assert!(!String::from_utf8_lossy(&selection.body).contains("duplicate charge"));
    // The backend was asked with its own, and got the caller's request with only the model changed.
    let asked = &laya.seen()[0];
    assert_eq!(asked.header("authorization"), Some("Bearer sk-local"));
    assert_eq!(asked.request(), request.clone().model("laya-english"));
    assert_eq!((jev.hits(), laya.hits()), (1, 1));

    // A backend that fails is asked once: a gateway's clients do not retry.
    let down = Upstream::start(|_, _| Reply::new(503, r#"{"detail":{"error_type":"overloaded","message":"busy"}}"#));
    let solo = json!({"name": "solo", "selector": "only", "backends": [{"id": "only", "base_url": down.url(), "model": "m", "api_key_env": "K", "description": "Down."}]});
    let gateway = Registry::from_json(&solo.to_string(), no_file)
        .unwrap()
        .gateway_with(any_key)
        .unwrap();
    let error = gateway.decide(&request).unwrap_err();
    assert_eq!(common::api(&error), (503, "overloaded".into(), "busy".into()));
    assert_eq!(down.hits(), 1);
    // The same server through a client of one's own can be asked again.
    let patient = Client::new(down.url()).retry(one_system::Retry {
        initial: std::time::Duration::from_millis(1),
        ..Default::default()
    });
    assert!(patient.decide(&request).is_err());
    assert_eq!(down.hits(), 4);
}

#[test]
fn a_base_url_is_read_as_the_go_gateway_reads_one() {
    let with = |url: &str| {
        let mut registry = two();
        registry["backends"][0]["base_url"] = json!(url);
        load(&registry).map(|registry| registry.backends[0].base_url.clone())
    };
    // The scheme is the same in any case, and is kept in lower case.
    assert_eq!(with("HTTPS://api.typesafe.ai/").unwrap(), "https://api.typesafe.ai");
    assert_eq!(with("Http://127.0.0.1:8091").unwrap(), "http://127.0.0.1:8091");
    // This machine, however its address is written.
    assert_eq!(with("http://[::ffff:127.0.0.1]:8091").unwrap(), "http://[::ffff:127.0.0.1]:8091");
    assert_eq!(with("http://[::1]").unwrap(), "http://[::1]");
    assert_eq!(with("http://127.8.9.10").unwrap(), "http://127.8.9.10");
    assert_eq!(
        with("https://example.com:8443/tenant%2Fteam").unwrap(),
        "https://example.com:8443/tenant%2Fteam"
    );

    let shape = "backend base_url must use HTTP(S) without credentials, query, or fragment";
    for bad in [
        // A port that is no number.
        "http://localhost:notaport",
        "https://example.com:80a",
        "https://example.com:-1",
        "http://[::1]:x",
        "http://[::1]x",
        // A bracket that is not closed, or that holds no address.
        "https://[::1",
        "http://[::1",
        "https://[example.com]",
        "https://[]",
        // An escape that is none.
        "http://127.0.0.1:8091/%zz",
        "https://example.com/a%",
        "https://example.com/a%2",
        "https://exa%6dple.com",
    ] {
        assert_eq!(with(bad).unwrap_err().message(), shape, "{bad}");
    }
    let remote = "backend base_url requires HTTPS except for loopback hosts";
    for bad in ["http://[::ffff:10.0.0.5]:8091", "HTTP://example.com", "http://[2001:db8::1]"] {
        assert_eq!(with(bad).unwrap_err().message(), remote, "{bad}");
    }
}

#[test]
fn a_member_written_twice_is_refused() {
    // The Go gateway keeps the last of two. Here a file that says two things is no file.
    let twice = r#"{"name": "demo", "selector": "local", "fallback": "local", "fallback": "",
        "backends": [{"id": "local", "base_url": "http://127.0.0.1:8091", "model": "m", "api_key_env": "K", "description": "d"}]}"#;
    assert_eq!(
        Registry::from_json(twice, no_file).unwrap_err().message(),
        "backend configuration must contain name, selector, and backends"
    );
}
