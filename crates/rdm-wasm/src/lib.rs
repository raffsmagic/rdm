//! An RDM as a WebAssembly module with nothing under it (`wasm32-unknown-unknown`): no system,
//! no clock, no network, no thread. It is the search of the README, and its decision model
//! is a server the host reaches: the module says what to ask, and the host carries it.
//!
//! The module is plain WebAssembly, with no `wasm-bindgen` and no glue but `host.mjs`, which
//! is all of it. Texts cross as JSON, in the module's memory:
//!
//! ```text
//! exports   rdm_alloc(len) -> ptr          memory the host writes a text into
//!           rdm_free(ptr, len)             and gives back
//!           rdm_search(ptr, len) -> ptr    {"text", "question", "server", "model", "key"} in, the result out
//! imports   host.http(ptr, len) -> ptr     one HTTP request out, its response back
//! ```
//!
//! What comes back from either is `[length: u32, little-endian][that many bytes of JSON]`, in
//! memory the other side frees. `rdm_search` answers `{"found": [...], "requests": n, "reason":
//! "..."}`, or `{"error": "..."}`.
//!
//! Everything in this crate waits for its answers, and a host written in JavaScript cannot be
//! waited on: `fetch` returns a promise. The engine closes that gap, not this module. With
//! JavaScript Promise Integration the host wraps `http` in `WebAssembly.Suspending` and
//! `rdm_search` in `WebAssembly.promising`: the module is suspended while the host's promise is
//! pending, and to the code here `http` is a function that returned. Where an engine has no
//! JSPI, the same module runs in a worker, with `http` a synchronous `XMLHttpRequest`.
//!
//! A panic ends a module: this target cannot unwind. A tool that fails here returns an error,
//! which is an observation, and the host's glue reports a trap as a failed call.
#![cfg(target_family = "wasm")]

use std::time::Duration;

use one_system::{Client, HttpRequest, HttpResponse, TransportError};
use rdm::{Json, Rdm, Signature, Tool, inputs};
use serde_json::json;

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// One HTTP request, as JSON: `{"method", "url", "headers": [[name, value]], "body",
    /// "timeout_ms", "limit"}`. The host answers `{"status", "headers", "body"}`, or
    /// `{"error": "...", "timed_out": bool}` for a request that got no response. It does not
    /// follow a redirect.
    fn http(request: *const u8, length: usize) -> *mut u8;
}

/// Memory for a text of `length` bytes, which the host writes and hands to `rdm_search`, or
/// returns from `http` with its length before it.
#[unsafe(no_mangle)]
pub extern "C" fn rdm_alloc(length: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; length].into_boxed_slice()).cast()
}

/// Give back what `rdm_alloc` gave, or what `rdm_search` returned (its length, and 4 for the length).
///
/// # Safety
/// `pointer` is one this module handed out, of this `length`, and is given back once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rdm_free(pointer: *mut u8, length: usize) {
    if !pointer.is_null() {
        // SAFETY: the caller gives back a box this module made of this length, once.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer, length)) });
    }
}

/// A text for the other side: its length, then its bytes.
fn handed(text: &str) -> *mut u8 {
    // The module is 32-bit WebAssembly: a length is a u32 already.
    let mut bytes = (text.len() as u32).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    Box::into_raw(bytes.into_boxed_slice()).cast()
}

/// What the host handed back from `http`: its length, then its bytes. The memory is taken.
///
/// # Safety
/// `pointer` is null, or what `http` returned: memory `rdm_alloc` gave the host, holding a
/// length and that many bytes, not yet taken.
unsafe fn taken(pointer: *mut u8) -> Vec<u8> {
    if pointer.is_null() {
        return Vec::new();
    }
    // SAFETY: the host wrote the length into the first 4 bytes of memory `rdm_alloc` gave it.
    let length = u32::from_le_bytes(unsafe { pointer.cast::<[u8; 4]>().read_unaligned() }) as usize;
    // SAFETY: that memory is one box, of the length and 4 for the length, and it is taken once.
    let whole = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer, length + 4)) };
    whole[4..].to_vec()
}

/// The host's network, as the client's transport.
fn carried(request: HttpRequest) -> Result<HttpResponse, TransportError> {
    let asked = json!({
        "method": request.method,
        "url": request.url,
        "headers": request.headers,
        "body": request.body.as_deref().map(String::from_utf8_lossy),
        "timeout_ms": request.timeout.as_millis() as u64,
        "limit": request.limit,
    })
    .to_string();
    // SAFETY: `http` reads `asked` while it is alive, and returns what `taken` is for.
    let answered: Json = serde_json::from_slice(&unsafe { taken(http(asked.as_ptr(), asked.len())) })
        .map_err(|_| TransportError::Failed("the host answered with something that is not JSON".to_owned()))?;
    if let Some(error) = answered.get("error") {
        return Err(match answered["timed_out"].as_bool() {
            Some(true) => TransportError::TimedOut,
            _ => TransportError::Failed(error.as_str().unwrap_or("the host could not carry the request").to_owned()),
        });
    }
    let headers = answered["headers"].as_array().into_iter().flatten();
    let headers = headers.filter_map(|pair| Some((pair[0].as_str()?.to_owned(), pair[1].as_str()?.to_owned())));
    Ok(HttpResponse {
        status: answered["status"].as_u64().unwrap_or(0) as u16,
        headers: headers.collect(),
        body: answered["body"].as_str().unwrap_or_default().as_bytes().to_vec(),
    })
}

/// Search a text for the answer to a question, as the README does: a yes about a text asks
/// the same question of each half, down to one line.
fn search(asked: &Json) -> rdm::Result<Json> {
    let text = |name: &str| asked[name].as_str().unwrap_or_default().to_owned();
    let mut model = Client::new(text("server"))
        .model(text("model"))
        .timeout(Duration::from_secs(60))
        .transport(carried);
    // A key is the host's to add where it can: a page has no place to keep one.
    if !text("key").is_empty() {
        model = model.key(text("key"));
    }
    let signature = Signature::new("Search a text for the answer to a question.")
        .input("text")
        .input("question")
        .noul("look", "Is the answer to `inputs.question` stated in `inputs.text`?");
    let look = Tool::new("look", "Look in each half of the text.", |call| {
        let (text, question): (String, String) = (call.arg("text")?, call.arg("question")?);
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() < 2 {
            return Ok(vec![text]);
        }
        let mut found = Vec::new();
        for half in lines.chunks(lines.len().div_ceil(2)) {
            let below = call.sub(inputs! { "text" => half.join("\n"), "question" => &question })?;
            found.extend(below.observed::<Vec<String>>("look").unwrap_or_default());
        }
        Ok(found)
    });
    let result = Rdm::new(signature)
        .tool(look)
        .max_depth(8)
        .model(model)
        .call(inputs! { "text" => text("text"), "question" => text("question") })?;
    Ok(json!({
        "found": result.observed::<Vec<String>>("look").unwrap_or_default(),
        "requests": result.trajectory.len(),
        "reason": result.termination_reason,
    }))
}

/// Run the search for the JSON at `pointer`, and hand back what it came to.
///
/// # Safety
/// `pointer` is `length` bytes the host wrote into memory `rdm_alloc` gave it, which it frees.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rdm_search(pointer: *const u8, length: usize) -> *mut u8 {
    let given: &[u8] = match pointer.is_null() {
        true => &[],
        // SAFETY: the host wrote `length` bytes at `pointer`, and they are not freed until after.
        false => unsafe { std::slice::from_raw_parts(pointer, length) },
    };
    let answered = match serde_json::from_slice::<Json>(given) {
        Ok(asked) => search(&asked).unwrap_or_else(|error| json!({"error": error.to_string()})),
        Err(_) => json!({"error": "What was given is not JSON."}),
    };
    handed(&answered.to_string())
}
