//! Pictures: a request with pictures in its state goes only where they are read.
//!
//! Nothing here opens a connection: the backends are functions, and a client's server is a
//! transport that is one.

mod common;

use std::sync::{Arc, Mutex};

use common::{answers, body, model};
use one_system::{Backend, Capabilities, Client, Gateway, HttpRequest, HttpResponse, Limits, Picked, Registry, SELECTOR_QUESTION, Summary};
use serde_json::json;
use systemone::{Answer, DecisionModel, Error, Json, Question, Request, Scripted};

fn picture() -> Json {
    // Far longer as base64 than the limits below allow characters.
    systemone::image("image/png", &[9; 6000])
}

/// What a signature with a picture input sends: the shaped state, and the picture after it.
fn editor() -> Request {
    let state = json!([
        {"instructions": "Look.", "input_fields": "1. `window` (image):", "inputs": {"window": "<image: window>"}},
        picture(),
    ]);
    Request::new(state).question("editor", Question::noul("Is `inputs.window` a text editor?"))
}

fn plain() -> Request {
    Request::new(json!({"inputs": {"window": "a text editor"}})).question("editor", Question::noul("Is `inputs.window` a text editor?"))
}

fn sees() -> Capabilities {
    Capabilities::all().input_modalities(["text", "image"])
}

#[test]
fn the_gateway_sends_a_request_with_pictures_only_to_a_backend_that_takes_images() {
    let (laya, unknown, seeing) = (model("laya", 0), model("unknown", 0), model("pplx-decider", 0));
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("laya", laya.clone()).capabilities(Capabilities::all().input_modalities(["text"])))
        // Declaring nothing is no declaration that it reads pictures.
        .backend(Backend::new("unknown", unknown.clone()))
        // A limit on characters does not count a picture's base64.
        .backend(
            Backend::new("seeing", seeing.clone())
                .capabilities(sees())
                .limits(Limits::default().max_characters(200)),
        )
        .selector("laya")
        .build()
        .unwrap();

    let routed = gateway.route(&editor());
    assert_eq!(routed.route.eligible, ["seeing"]);
    assert_eq!(
        (routed.route.backend.as_deref(), routed.route.picked),
        (Some("seeing"), Some(Picked::Only))
    );
    assert_eq!(seeing.requests(), [editor()]);
    assert!(laya.requests().is_empty() && unknown.requests().is_empty());

    // Without pictures, every backend is able, as before.
    let plain = Gateway::builder("demo")
        .backend(Backend::new("laya", model("laya", 0)).capabilities(Capabilities::all()))
        .backend(Backend::new("seeing", model("pplx-decider", 0)).capabilities(sees()))
        .selector("seeing")
        .build()
        .unwrap();
    let mut eligible = plain.route(&self::plain()).route.eligible;
    eligible.sort();
    assert_eq!(eligible, ["laya", "seeing"]);

    // A backend named that cannot take them is not sent them.
    let named = gateway.route(&editor().model("laya"));
    assert_eq!(named.result.unwrap_err().status(), Some(422));
    assert!(laya.requests().is_empty());
}

#[test]
fn a_summary_counts_pictures_and_not_their_base64() {
    let summary = Summary::of(&editor().state);
    assert_eq!(summary.images, 1);
    let without = Summary::of(&editor().state.as_array().unwrap()[..1].to_vec().into());
    assert_eq!(
        (summary.characters, summary.non_ascii_letter_fraction),
        (without.characters, without.non_ascii_letter_fraction)
    );
    assert!(summary.characters < 200);
    // Written as it always was where there are none.
    assert_eq!(json!(Summary::of(&plain().state)).as_object().unwrap().len(), 2);
    assert_eq!(json!(summary)["images"], 1);
}

/// A selector that picks `choose`, sure of it.
fn picking(name: &str, choose: &'static str) -> Arc<Scripted> {
    Arc::new(
        Scripted::new(move |request: &Request| {
            let mut given = answers(request);
            if let Some(question) = request.questions.get(SELECTOR_QUESTION) {
                given.insert(
                    SELECTOR_QUESTION.into(),
                    Answer::chose(choose, question.labels().iter().map(String::as_str)),
                );
            }
            given
        })
        .named(name),
    )
}

#[test]
fn a_selector_that_is_the_fallback_and_cannot_see_is_given_the_summary_and_never_the_pictures() {
    let laya = picking("laya", "one");
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("laya", laya.clone()).capabilities(Capabilities::all()))
        .backend(Backend::new("one", model("one", 0)).capabilities(sees()).description("One."))
        .backend(Backend::new("two", model("two", 0)).capabilities(sees()).description("Two."))
        .selector("laya")
        .fallback("laya")
        .build()
        .unwrap();
    let routed = gateway.route(&editor());
    assert_eq!(routed.route.backend.as_deref(), Some("one"));
    let asked = &laya.requests()[0];
    assert_eq!(asked.state["input_summary"]["images"], 1);
    assert_eq!(asked.images(), 0);
    assert!(!asked.state.to_string().contains("CQkJ"), "no base64 of the picture");

    // One that can see is the fallback the state stays with: it sees the state, pictures and all.
    let seeing = picking("seeing", "one");
    let gateway = Gateway::builder("demo")
        .backend(Backend::new("seeing", seeing.clone()).capabilities(sees()))
        .backend(Backend::new("one", model("one", 0)).capabilities(sees()).description("One."))
        .selector("seeing")
        .fallback("seeing")
        .build()
        .unwrap();
    gateway.route(&editor()).result.unwrap();
    assert_eq!(seeing.requests()[0].state, editor().state);
}

#[test]
fn a_backend_whose_state_must_be_text_is_never_sent_pictures() {
    let text_only = sees().structured_state(false);
    assert!(!text_only.supports(&editor().state, &editor().questions));
    assert!(sees().supports(&editor().state, &editor().questions));
    assert!(!Capabilities::all().supports(&editor().state, &editor().questions));
}

#[test]
fn a_registry_entry_says_it_takes_images_by_its_input_modalities_and_one_that_does_not_is_text_only() {
    let registry = Registry::from_json(
        r#"{"name": "demo", "selector": "hosted", "backends": [
            {"id": "local", "base_url": "http://127.0.0.1:8080", "model": "laya-english", "api_key_env": "LOCAL_KEY",
             "description": "Short English.", "capabilities": {"question_types": ["noul"]}},
            {"id": "hosted", "base_url": "https://openrouter.ai/api", "model": "perplexity/pplx-decider-v1.2",
             "api_key_env": "OPENROUTER_API_KEY", "description": "Pictures.", "capabilities": {"question_types": ["choice", "score", "noul"], "input_modalities": ["text", "image"]}}
        ]}"#,
        |_| unreachable!(),
    )
    .unwrap();
    let takes = |id: &str| {
        let entry = registry.backends.iter().find(|entry| entry.id == id).unwrap();
        entry.capabilities.as_ref().unwrap().takes_images()
    };
    assert!(takes("hosted") && !takes("local"));
    let doubled = r#"{"name": "demo", "selector": "a", "backends": [{"id": "a", "base_url": "https://x.example", "model": "m",
        "api_key_env": "K", "description": "A.", "capabilities": {"question_types": ["noul"], "input_modalities": ["image", "image"]}}]}"#;
    let refused = Registry::from_json(doubled, |_| unreachable!()).unwrap_err();
    assert!(refused.to_string().contains("input_modalities"), "{refused}");
}

/// A server reached through a transport that is a function: it keeps every body it is sent.
fn server(carried: Arc<Mutex<Vec<Json>>>) -> impl Fn(HttpRequest) -> Result<HttpResponse, one_system::TransportError> + Send + Sync {
    move |request: HttpRequest| {
        let sent: Json = serde_json::from_slice(request.body.as_deref().unwrap_or_default()).unwrap();
        let asked: Request = serde_json::from_value(sent.clone()).unwrap();
        carried.lock().unwrap().push(sent);
        Ok(HttpResponse {
            status: 200,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body(&asked, "pplx-decider-v1.2", 1).into_bytes(),
        })
    }
}

#[test]
fn openrouter_is_sent_the_pictures_as_items_of_the_state() {
    let carried = Arc::new(Mutex::new(Vec::new()));
    let client = Client::openrouter().key("sk-test").transport(server(carried.clone()));
    client.decide(&editor()).unwrap();
    let sent = carried.lock().unwrap()[0].clone();
    assert_eq!(
        sent["state"][1],
        json!({"type": "image_url", "image_url": {"url": picture()["image_url"]["url"]}})
    );
    assert!(sent.get("images").is_none());
}

#[test]
fn the_typesafe_client_and_a_local_one_refuse_a_request_with_pictures_and_send_nothing() {
    // Checked 2026-10-17: TypeSafe's JSON endpoint answered a red picture and a green one the
    // same, at about 0.5, as if it had seen neither; OpenRouter's decisions told them apart.
    let carried = Arc::new(Mutex::new(Vec::new()));
    let clients = [
        Client::typesafe().key("sk-test"),
        Client::new("https://API.typesafe.ai/").key("sk-test"),
        Client::local("http://127.0.0.1:8080"),
    ];
    for client in clients {
        let client = client.transport(server(carried.clone()));
        let refused = client.decide(&editor()).unwrap_err();
        assert!(matches!(refused, Error::Invalid(_)), "{refused}");
        assert!(refused.to_string().contains("does not read pictures"), "{refused}");
        // A request with none is sent as ever.
        client.decide(&plain()).unwrap();
    }
    assert_eq!(carried.lock().unwrap().len(), 3);
    let typesafe = Client::typesafe().key("sk-test").decide(&editor()).unwrap_err().to_string();
    assert!(
        typesafe.contains("TypeSafe's JSON endpoint") && typesafe.contains("multipart"),
        "{typesafe}"
    );
    // Told it reads them, a local server is sent them.
    let read = Client::local("http://127.0.0.1:8080")
        .images(true)
        .transport(server(carried.clone()));
    read.decide(&editor()).unwrap();
}
