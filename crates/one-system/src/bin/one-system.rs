//! `one-system`: a gateway for decision models, served over HTTP.
//!
//! Configured from the environment, with the names the Go gateway uses:
//!
//! - `ONE_SYSTEM_API_KEY` (required): the bearer key callers present.
//! - `ONE_SYSTEM_CONFIG` (default `backends.json`): the registry file.
//! - `ONE_SYSTEM_ADDR` (default `127.0.0.1:8090`): where to listen.
//! - each backend's `api_key_env`: that backend's own key. All of them must be set.
//! - `ONE_SYSTEM_CACHE_ENTRIES` (default 0, which is off): how many answers to remember.
//! - `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`, `ALL_PROXY`: a proxy for the backends that are
//!   elsewhere. A backend on this machine is never asked through one.

#[cfg(not(target_family = "wasm"))]
use one_system::{Registry, Server};
#[cfg(not(target_family = "wasm"))]
use serde_json::json;

#[cfg(not(target_family = "wasm"))]
fn setting(name: &str, otherwise: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| otherwise.to_string())
}

/// Serve until the process ends, or say why it could not start. The reason is one of a fixed
/// set of sentences, with nothing in it from a file or from the environment.
#[cfg(not(target_family = "wasm"))]
fn run() -> Result<(), String> {
    let reason = |error: one_system::ConfigError| error.message().to_string();
    let key = setting("ONE_SYSTEM_API_KEY", "");
    let address = setting("ONE_SYSTEM_ADDR", "127.0.0.1:8090");
    let cache: usize = setting("ONE_SYSTEM_CACHE_ENTRIES", "0")
        .parse()
        .map_err(|_| "ONE_SYSTEM_CACHE_ENTRIES must be a whole number")?;
    // The key is checked before any file is opened.
    if key.is_empty() || key.contains(char::is_whitespace) {
        return Err("ONE_SYSTEM_API_KEY must contain a nonempty bearer key without whitespace".into());
    }
    let registry = Registry::load(setting("ONE_SYSTEM_CONFIG", "backends.json")).map_err(reason)?;
    let gateway = registry.builder().map_err(reason)?.cache(cache).build().map_err(reason)?;
    let server = Server::bind(address.as_str(), gateway, &key).map_err(reason)?.log_routes(true);
    eprintln!(
        "{}",
        json!({"level": "INFO", "msg": "listening", "addr": server.addr().to_string(), "name": registry.name})
    );
    server.run();
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    if let Err(reason) = run() {
        eprintln!("{}", json!({"level": "ERROR", "msg": "startup failed", "reason": reason}));
        std::process::exit(1);
    }
}

/// A WebAssembly module can listen on no port: the gateway is the library there, and a
/// server in front of it is the host's.
#[cfg(target_family = "wasm")]
fn main() {
    eprintln!(r#"{{"level": "ERROR", "msg": "startup failed", "reason": "a WebAssembly module cannot listen for requests"}}"#);
    std::process::exit(1);
}
