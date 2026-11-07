// The host of the module: all the JavaScript there is. It instantiates the module, carries
// its HTTP requests, and asks it to search a text.
//
//   cargo build -p rdm-wasm --target wasm32-unknown-unknown --release
//   node crates/rdm-wasm/host.mjs                                   # a stand-in server, in this file
//   RDM_SERVER=http://127.0.0.1:8080 node crates/rdm-wasm/host.mjs  # a System One server, by fetch
//   RDM_SERVER=https://api.typesafe.ai RDM_MODEL=jev-latest RDM_KEY_ENV=TYPESAFE_API_KEY node crates/rdm-wasm/host.mjs
//
// The module waits for each answer, and `fetch` answers with a promise. JavaScript Promise
// Integration is what lets the one be the other: the engine suspends the module while the
// promise is pending. Node 24 and later has it, and so have the browsers built on Chromium;
// where an engine has none, run the module in a worker and carry each request with a
// synchronous XMLHttpRequest in place of `carry`.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const built = new URL('../../target/wasm32-unknown-unknown/release/rdm_wasm.wasm', import.meta.url);
const file = process.argv[2] ?? fileURLToPath(built);
const server = process.env.RDM_SERVER ?? '';
const model = process.env.RDM_MODEL ?? '';
// The key is read here, from the variable that holds it, and is given to the request as it
// leaves. The module never has it, and neither has anything it returns.
const key = process.env[process.env.RDM_KEY_ENV ?? ''] ?? '';

/// A stand-in for a System One server: a yes for a text with the word `nested` in it. It
/// answers after a moment, as a server does, so that the module really is waited for.
async function standIn(request) {
  await new Promise((resolve) => setTimeout(resolve, 2));
  const { state, questions } = JSON.parse(request.body);
  const nested = state.inputs.text.includes('nested');
  const answers = Object.fromEntries(Object.keys(questions).map((name) => [name, { type: 'noul', noul: nested ? 0.9 : 0.1 }]));
  return { status: 200, headers: [], body: JSON.stringify({ model: 'stand-in', answers }) };
}

/// Carry one request to the server it names, and bring back what was answered.
async function carry(request) {
  if (!server) return standIn(request);
  const headers = [...request.headers, ...(key ? [['Authorization', `Bearer ${key}`]] : [])];
  try {
    const response = await fetch(request.url, {
      method: request.method,
      headers,
      body: request.body ?? undefined,
      redirect: 'manual', // the request carries a key, and goes nowhere it was not sent
      signal: AbortSignal.timeout(request.timeout_ms),
    });
    return { status: response.status, headers: [...response.headers], body: await response.text() };
  } catch (error) {
    return { error: String(error.cause?.message ?? error.message), timed_out: error.name === 'TimeoutError' };
  }
}

let memory, exports;
const text = (pointer, length) => new TextDecoder().decode(new Uint8Array(memory.buffer, pointer, length));
/// Write a text into the module's memory, with its length before it where the module is to read that.
function write(string, lengthFirst) {
  const bytes = new TextEncoder().encode(string);
  const pointer = exports.rdm_alloc(bytes.length + (lengthFirst ? 4 : 0));
  if (lengthFirst) new DataView(memory.buffer).setUint32(pointer, bytes.length, true);
  new Uint8Array(memory.buffer, pointer + (lengthFirst ? 4 : 0), bytes.length).set(bytes);
  return [pointer, bytes.length];
}
/// Read what the module handed back, and free it.
function read(pointer) {
  const length = new DataView(memory.buffer).getUint32(pointer, true);
  const string = text(pointer + 4, length);
  exports.rdm_free(pointer, length + 4);
  return string;
}

let carried = 0;
const http = new WebAssembly.Suspending(async (pointer, length) => {
  carried += 1;
  const answered = await carry(JSON.parse(text(pointer, length)));
  return write(JSON.stringify(answered), true)[0];
});
const module = await WebAssembly.instantiate(await readFile(file), { host: { http } });
exports = module.instance.exports;
memory = exports.memory;
const search = WebAssembly.promising(exports.rdm_search);

const zen = 'Beautiful is better than ugly.\nSimple is better than complex.\nFlat is better than nested.\nSparse is better than dense.';
const asked = { text: zen, question: 'Is flat better than nested?', server: server || 'http://stand-in.invalid', model };
const [pointer, length] = write(JSON.stringify(asked), false);
let result;
try {
  result = JSON.parse(read(await search(pointer, length)));
  exports.rdm_free(pointer, length);
} catch (trap) {
  // A panic in the module ends it: this target cannot unwind. The instance is not used again.
  result = { error: `the module stopped: ${trap.message}` };
}
console.log(JSON.stringify(result));
console.log(`${carried} requests carried by the host`);
if (result.error) process.exitCode = 1;
