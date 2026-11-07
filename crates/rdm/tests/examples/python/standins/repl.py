"""The rules of examples/repl.rs, and the cases its transcripts record.

The cell's writer is a `dspy.RLM`, so `writes` replies with one step of an RLM: code that calls
`define` and submits what it returned. The Rust writer is one request whose reply is the function
as JSON. So `cells.py same repl` finds the same lines printed and the first request the same, and
says the requests differ from the second: the cell's observation after `write` is the RLM's whole
prediction, and the port's is what `define` returned.
"""

import re

CASES = [
    ("", "", {}),
    ("three-cooler", "--task 2", {"task": "Make it three degrees cooler."}),
    ("down-to-16", '"Bring the heat down to 16 degrees."', {"task": "Bring the heat down to 16 degrees."}),
    ("lovely-evening", '"What a lovely evening."', {"task": "What a lovely evening."}),
]
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
RETURNS = "    return f'it is {heat()} degrees'"


def function_for(task):
    task = task.lower()
    words = [word for word in re.split(r"[^0-9a-z]", task) if word]
    at = next((at for at, word in enumerate(words) if word.startswith("degree")), 0)
    if not at:
        return None
    said = words[at - 1]
    degrees = int(said) if said.isdigit() else NUMBERS.index(said) + 1 if said in NUMBERS else None
    if degrees is None:
        return None
    if words[at + 1 : at + 2] == ["cooler"]:
        name = f"cool_by_{said}_degrees"
        return name, f"Make it {said} degrees cooler.", f"def {name}():\n    for _ in range({degrees}):\n        cooler()\n{RETURNS}"
    if "down to" in task:
        name = f"cool_to_{degrees}"
        return name, f"Cool the house until it is {degrees} degrees.", f"def {name}():\n    while heat() > {degrees}:\n        cooler()\n{RETURNS}"
    return None


def decides(inputs, name, question):
    observation = inputs["observation"]
    written = re.match(r"`(\w+)` was written", observation) or re.search(r"wrote='`(\w+)` was written", observation)
    if written:
        return written[1] if written[1] in question["criteria"] else "finish"
    return "write" if not observation and function_for(inputs["task"]) else "finish"


def writes(messages):
    # An RLM shows its writer each input as a variable, with the start of what it holds.
    task = re.search(r"Variable: `task`.*?Preview:\s*```\s*(.*?)\s*```", messages[-1]["content"], re.S)
    name, description, code = function_for(task[1] if task else "") or ("", "", "")
    step = f"SUBMIT(wrote=define(name={name!r}, description={description!r}, code={code!r}))"
    return f"[[ ## reasoning ## ]]\nWrite the function, and submit what `define` said.\n\n[[ ## code ## ]]\n```python\n{step}\n```\n\n[[ ## completed ## ]]"
