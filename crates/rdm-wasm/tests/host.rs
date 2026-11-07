//! The module, built and hosted: `cargo` builds it for `wasm32-unknown-unknown`, and `node`
//! runs `host.mjs`, which carries its requests to a stand-in server of its own and prints
//! what the search came to. It is the one place the crates are run with nothing under them.
//!
//! It needs the target and a Node with JavaScript Promise Integration (24 or later). Without
//! either there is nothing to run it with, and the test says so and passes.
// This test starts cargo and node, and a WebAssembly module can start no program.
#![cfg(not(target_family = "wasm"))]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The directory cargo builds into: the one above this test with cargo's tag in it.
fn target_directory() -> PathBuf {
    let test = std::env::current_exe().expect("where this test is");
    let found = test.ancestors().find(|directory| directory.join("CACHEDIR.TAG").exists());
    found.expect("this test is where cargo builds one").to_owned()
}

#[test]
fn the_module_searches_and_its_host_carries_each_request() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR"));
    let waits = "process.exit(typeof WebAssembly.Suspending === 'function' ? 0 : 1)";
    let node = Command::new("node").args(["-e", waits]).stdin(Stdio::null()).status();
    if !node.is_ok_and(|status| status.success()) {
        return eprintln!("not run: there is no node here with JavaScript Promise Integration.");
    }
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let target = target_directory();
    let built = Command::new(cargo)
        .current_dir(package)
        .args([
            "build",
            "--offline",
            "-p",
            "rdm-wasm",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
        ])
        .arg(&target)
        .stdin(Stdio::null())
        .output()
        .expect("cargo");
    let complained = String::from_utf8_lossy(&built.stderr);
    if !built.status.success() && complained.contains("target may not be installed") {
        return eprintln!("not run: the target wasm32-unknown-unknown is not installed.");
    }
    assert!(built.status.success(), "\nThe module does not build.\n\n{complained}\n");

    let module = target.join("wasm32-unknown-unknown/debug/rdm_wasm.wasm");
    let mut host = Command::new("node");
    host.arg(package.join("host.mjs")).arg(&module).stdin(Stdio::null());
    // Nothing of this environment names a server: the host's stand-in answers.
    for named in ["RDM_SERVER", "RDM_MODEL", "RDM_KEY_ENV"] {
        host.env_remove(named);
    }
    let ran = host.output().expect("node");
    let printed = String::from_utf8_lossy(&ran.stdout);
    assert!(ran.status.success(), "{printed}\n{}", String::from_utf8_lossy(&ran.stderr));
    // What the README's program comes to, and one request carried for each the run made.
    assert_eq!(
        printed,
        "{\"found\":[\"Flat is better than nested.\"],\"requests\":5,\"reason\":\"final\"}\n5 requests carried by the host\n"
    );
}
