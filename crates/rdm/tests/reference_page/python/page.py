"""What the Python module asks for the reference page: where `requests.jsonl` comes from.

The blocks of `docs/docs/api/experimental/RDM.md` are run top to bottom, as
`tests/predict/test_rdm_docs.py` runs them and with its stand-ins, and every request a
stand-in decision model is put is written down as one line:

    {"block": 2, "module": "rdm", "state": {...}, "questions": {...}}

`block` is the python block of the page that was running, counted from 1, and `module` the
module whose stand-in was asked, named as the page names it. After the last block the four
thresholds of the page are tried: the four modules are called once more, and those requests
are block 16. `tests/reference_page.rs` sets what the Rust sends beside these lines.

The lines are in the order the requests arrived. Requests a module asks at once (a batch of
sub-calls, the parts `Fit` asks about, the texts `Near` labels) arrive in an order of their
own, so those lines change places from one run of this file to the next, and nothing else does.

    cd <the dspy checkout> && RDM_DSPY=$PWD .venv/bin/python <this file> [requests.jsonl]

It needs the worktree's own environment (dspy, dspy-interpreters) and touches no network.
"""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

WORKTREE = Path(os.environ["RDM_DSPY"])  # a checkout of github.com/rawwerks/dspy, branch feat/rdm-v0
sys.path[:0] = [str(WORKTREE), str(WORKTREE / "tests/predict")]

import dspy  # noqa: E402
import test_rdm_docs as docs  # noqa: E402
from dspy_interpreters import InProcessInterpreter  # noqa: E402

MODULES = ["act", "rdm", "search", "lines", "classify", "wide", "route", "cool", "covered", "guided"]
QUESTION = "How many obvious ways should there be to do something?"

# The page's last block sets four thresholds and asks nothing. This calls each module again.
# `lines.search` is a new predictor with no model of its own on the page, so it is given the
# module's, and the stand-in of `guided` is given its steps to take once more.
AFTER = """
lines.search.set_lm(lines.predict.lm)
guided.predict.lm.plans["Lock the front door."] = iter(["lock locked", "door closed", "lock locked", "finish"])
search(text=zen, question=QUESTION)
lines(question=QUESTION, state={"text": zen})
covered(text=zen, question=QUESTION)
house.update(door="open", lock="unlocked")
guided(task="Lock the front door.", house=house)
"""

block, written, made = 0, [], []


class Jev(docs.Jev):
    """The page's stand-in, writing down what it is asked."""

    def __init__(self, model=None):
        super().__init__(model)
        self.module = MODULES[len(made)]
        made.append(self)

    def __call__(self, state, questions):
        asked = json.loads(json.dumps({"state": state, "questions": questions}))  # as it is now: the house changes
        written.append({"block": block, "module": self.module, **asked})
        return super().__call__(state, questions)


def main(path):
    global block
    scope = {"_jev": Jev, "_llm": docs.llm, "__name__": "rdm_docs", "QUESTION": QUESTION}
    with contextlib.redirect_stdout(io.StringIO()), dspy.context(interpreter_factory=InProcessInterpreter):
        for source in [*docs.blocks(), AFTER]:
            block += 1
            source = docs.re.sub(r"\bTypeSafe\(", "_jev(", source).replace("dspy.LM(", "_llm(")
            exec(compile(source, str(docs.PAGE), "exec"), scope)
    Path(path).write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in written))
    print(f"{len(written)} requests in {block} blocks -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "requests.jsonl")
