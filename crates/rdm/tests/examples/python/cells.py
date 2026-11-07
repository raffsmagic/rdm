"""The Python cells, run beside their Rust ports: where the transcripts in tests/examples/ come from.

A transcript is to say what the Python cell prints. This runs a cell of the site as plain Python, on a
stand-in decision model that answers by the same rules as the Rust example's, so that what the cell
prints can be recorded and what the port sends can be set beside what the cell sends.

    export RDM_CELLS=<the site's repository>/src/lib/examples       # the cells, *.py
    PY=<a checkout of the dspy fork with RDM>/.venv/bin/python      # a Python that has that dspy

    $PY cells.py run act task="Please shut the door."     # the cell, with an input set as the form sets it
    $PY cells.py record act                               # write tests/examples/act*.txt from what the cell prints
    $PY cells.py same act                                 # the port: does it print, and send, what the cell does?

The rules of a cell are in standins/<cell>.py, the twin of the `standin` at the foot of
examples/<cell>.rs:

    def decides(inputs, name, question)   True or False, a probability, the option chosen, or {option: probability}
    def writes(messages)                  what the language model writes, for a cell that has one
    CASES = [(case, arguments, inputs)]   a transcript for each: <cell>[.<case>].txt, the arguments of the Rust
                                          example after `--`, and the same inputs for the cell, by name

    RECORDED_FROM_RUST = "why"            only where the transcripts are the Rust example's own, and not the cell's:
                                          `record` then refuses the cell, and says why

`same` runs the Rust example (`cargo run -p rdm --example <cell>`, so cargo must be on the PATH) against a
System One server on this machine that answers by `decides` and writes down what it is sent. It asks nothing
of the network, and neither does anything else here: the taxonomy a cell fetches is read from examples/assets.

Requests that are asked at once reach the server in any order. Where the example sends the requests the cell
sends and only their order differs, `same` says so, and does not call it a difference.
"""

import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.dont_write_bytecode = True  # nothing is left beside the rules
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]  # crates/rdm
ASSETS = PACKAGE / "examples" / "assets"
sys.path.insert(0, str(HERE))


def rules(cell):
    return importlib.import_module(f"standins.{cell}")


def evidence(said, question):
    """What a rule said, as a System One server answers: the numbers of the site's mock."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": (0.93 if said else 0.04) if isinstance(said, bool) else said}
    labels = list(question["criteria"])
    if not isinstance(said, dict):
        assert said in labels, f"{said!r} is not one of {labels}"
        said = {label: 0.95 if label == said else 0.05 / max(len(labels) - 1, 1) for label in labels}
    odds = {label: said.get(label, 0.0) for label in labels}
    return {"type": "choice", "choice": max(odds, key=odds.get), "probabilities": odds, "confidence": 0.9}


def run(cell, inputs, sent=None):
    """Run a cell, with these inputs set as the page's form sets them. Appends what it sends to `sent`."""
    import urllib.request

    import dspy

    source = (Path(os.environ["RDM_CELLS"]) / f"{cell}.py").read_text()
    for name, value in inputs.items():
        found = re.search(rf"^{name} = (.*(?:\n(?:[ \t]+.*|[\]}})]))*)$", source, re.M)
        assert found, f"the cell assigns no {name}"
        source = source[: found.start()] + f"{name} = {value!r}" + source[found.end() :]
    twin = rules(cell)

    class BrowserDecision:
        supports_decision_requests = True
        cache = False
        model = "jev-stand-in"

        def __init__(self, model=None):
            self.history, self.callbacks = [], []

        def __call__(self, state, questions):
            if sent is not None:
                sent.append(json.dumps({"state": state, "questions": questions}))
            return {name: evidence(twin.decides(state["inputs"], name, question), question) for name, question in questions.items()}

    class BrowserLM(dspy.BaseLM):
        def __init__(self, model=None, **kwargs):
            super().__init__(model="lm-stand-in", cache=False)

        def forward(self, prompt=None, messages=None, **kwargs):
            text = twin.writes(messages or [{"role": "user", "content": prompt}])
            message = types.SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
            choice = types.SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
            return types.SimpleNamespace(choices=[choice], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, model=self.model)

    def fetched(url, *args, **kwargs):
        assert "product-taxonomy" in str(url), f"a cell asked the network for {url}"
        return open(ASSETS / "categories.txt", "rb")

    urllib.request.urlopen = fetched
    space = {"BrowserDecision": BrowserDecision, "BrowserLM": BrowserLM, "__name__": "__main__"}
    space.update({name: lambda *args, **kwargs: None for name in ("show_text", "show_house", "show_tree", "show_branch", "show_leaf")})
    # The files a cell opens are beside it on the site: here, in a directory of its own.
    with tempfile.TemporaryDirectory() as beside:
        for name in ("handbook.md", "starter_map.json"):
            (Path(beside) / name).write_bytes((ASSETS / name).read_bytes())
        os.chdir(beside)
        exec(compile(source, f"{cell}.py", "exec"), space)
        os.chdir(HERE)


def cell_prints(cell, inputs):
    """What the cell prints, and what it sends, in a Python of its own."""
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as log:
        settings = [f"{name}={json.dumps(value)}" for name, value in inputs.items()]
        ran = subprocess.run([sys.executable, __file__, "run", cell, *settings], capture_output=True, text=True, env={**os.environ, "RDM_SENT": log.name})
        assert ran.returncode == 0, ran.stderr
        return ran.stdout, log.read().splitlines()


def command(cell, arguments):
    return f"$ cargo run -p rdm --example {cell}" + (f" -- {arguments}" if arguments else "")


def transcript(cell, case):
    return PACKAGE / "tests" / "examples" / (f"{cell}.{case}.txt" if case else f"{cell}.txt")


def serve(cell, sent):
    """A System One server on this machine that answers by the cell's rules. Returns its address."""
    twin = rules(cell)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            sent.append(json.dumps({"state": body["state"], "questions": body["questions"]}))
            answers = {name: evidence(twin.decides(body["state"]["inputs"], name, question), question) for name, question in body["questions"].items()}
            reply = json.dumps({"model": "jev-stand-in", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def main():
    what, cell = sys.argv[1], sys.argv[2]
    if what == "run":
        sent = [] if os.environ.get("RDM_SENT") else None
        inputs = dict(setting.split("=", 1) for setting in sys.argv[3:])
        for name, value in inputs.items():
            try:
                inputs[name] = json.loads(value)
            except ValueError:
                pass
        try:
            run(cell, inputs, sent)
        finally:
            if sent is not None:
                Path(os.environ["RDM_SENT"]).write_text("".join(line + "\n" for line in sent))
    elif what == "record":
        kept = getattr(rules(cell), "RECORDED_FROM_RUST", None)
        if kept:
            sys.exit(f"{cell} is not recorded from its cell: {kept}\nIts transcripts are written by the Rust example: RDM_EXAMPLES_RECORD=1 cargo test -p rdm --test examples -- --exact {cell}")
        for case, arguments, inputs in rules(cell).CASES:
            printed, _ = cell_prints(cell, inputs)
            transcript(cell, case).write_text(f"{command(cell, arguments)}\n{printed}")
            print(transcript(cell, case).name, "recorded:", len(printed.splitlines()), "lines")
    elif what == "same":
        differs = False
        for case, arguments, inputs in rules(cell).CASES:
            printed, sent = cell_prints(cell, inputs)
            asked = []
            # The server is named as the cell's model is, so that an example that prints its model's name prints that one.
            environment = {**os.environ, "RDM_DECISION_MODEL": serve(cell, asked), "RDM_DECISION_MODEL_NAME": "jev-stand-in", "RDM_LANGUAGE_MODEL": "standin"}
            with tempfile.TemporaryDirectory() as scratch:
                cargo = ["cargo", "run", "--quiet", "--manifest-path", str(PACKAGE / "Cargo.toml"), "--example", cell, "--", *shlex.split(arguments)]
                ran = subprocess.run(cargo, capture_output=True, text=True, env=environment, cwd=scratch)
            # What is asked at once arrives in any order: the same requests in another order are the same requests.
            reordered = asked != sent and sorted(asked) == sorted(sent)
            same = ran.returncode == 0 and ran.stdout == printed and (asked == sent or reordered)
            differs |= not same
            requests = f"the same {len(sent)} requests" + (", in another order" if reordered else "")
            print(command(cell, arguments)[2:], "->", f"the same {len(printed.splitlines())} lines, {requests}" if same else "DIFFERS")
            if ran.returncode != 0:
                print("  the example failed:", ran.stderr.strip())
            elif ran.stdout != printed:
                print("  the cell prints:\n" + printed + "  the example prints:\n" + ran.stdout)
            elif not same:
                at = next((at for at, (one, other) in enumerate(zip(sent, asked)) if one != other), min(len(sent), len(asked)))
                print(f"  the cell sends {len(sent)} requests and the example {len(asked)}; they differ from request {at + 1}:")
                print("   cell:   ", (sent[at : at + 1] or ["nothing"])[0][:1200])
                print("   example:", (asked[at : at + 1] or ["nothing"])[0][:1200])
        sys.exit(1 if differs else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
