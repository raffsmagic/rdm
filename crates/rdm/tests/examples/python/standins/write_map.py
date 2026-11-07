"""The rules of examples/write_map.rs, and the cases its transcripts record.

The cell's writer is a `dspy.RLM`, so `writes` replies with one step of an RLM: code that calls
`define` and submits what it returned. The Rust writer is one request whose reply is the function
as JSON. So `cells.py same write_map` finds the same lines printed and as many requests, and says
the requests differ from the second: the cell's observation after `write` is the RLM's whole
prediction, and the port's is what `define` returned.
"""

import json
import re
from pathlib import Path

MOVIE = "Movie time: turn the TV on and the lights off."
CASES = [
    ("", "", {}),
    ("movie-twice", f'"{MOVIE}" "{MOVIE}"', {"routines": [MOVIE, MOVIE]}),
    ("not-a-routine", f'"Party: turn everything on." "{MOVIE}"', {"routines": ["Party: turn everything on.", MOVIE]}),
]
SHIPPED = json.loads((Path(__file__).parents[4] / "examples" / "assets" / "starter_map.json").read_text())["written"]


def shipped(task):
    name = task.split(":")[0].strip().lower().replace(" ", "_")
    return (name, SHIPPED[name]) if name in SHIPPED else None


def decides(inputs, name, question):
    observation, routine = inputs["observation"], shipped(inputs["task"])
    written = re.match(r"`(\w+)` was written", observation) or re.search(r"wrote='`(\w+)` was written", observation)
    if written:
        return written[1] if written[1] in question["criteria"] else "finish"
    if observation or not routine:
        return "finish"
    return routine[0] if routine[0] in question["criteria"] else "write"


def writes(messages):
    # An RLM shows its writer each input as a variable, with the start of what it holds.
    task = re.search(r"Variable: `task`.*?Preview:\s*```\s*(.*?)\s*```", messages[-1]["content"], re.S)
    name, written = shipped(task[1] if task else "") or ("", {"description": "", "code": ""})
    step = f"SUBMIT(wrote=define(name={name!r}, description={written['description']!r}, code={written['code']!r}))"
    return f"[[ ## reasoning ## ]]\nWrite the function, and submit what `define` said.\n\n[[ ## code ## ]]\n```python\n{step}\n```\n\n[[ ## completed ## ]]"
