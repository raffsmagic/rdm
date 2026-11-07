"""The rules of examples/route.rs, and the cases its transcripts record."""

CASES = [
    ("", "", {}),
    ("close-the-door", "--task 2", {"task": "Close the door."}),
    ("sky-blue", "--task 3", {"task": "Why is the sky blue?"}),
    ("lock-refused", '"Lock the front door, and turn the lights off."', {"task": "Lock the front door, and turn the lights off."}),
    ("other-words", '"What a lovely evening."', {"task": "What a lovely evening."}),
]
DEVICES = [("lock", "lock"), ("tv", "tv"), ("light", "lights"), ("door", "door")]
STATES = [("unlock", "unlocked"), ("lock", "locked"), ("shut", "closed"), ("close", "closed"), ("open", "open"), (" off", "off"), (" on", "on")]
REPLIES = [("rain", "I cannot see the weather from here."), ("sky", "Sunlight is scattered by the air, and blue light the most.")]


def decides(inputs, name, question):
    task = inputs["task"].lower()

    def named(words):
        return next((means for word, means in words if word in task), None)

    device, to = named(DEVICES), named(STATES)
    several = ", and " in task
    one = bool(device and to) and not several
    asks = task.endswith("?") or "tell me" in task
    return {"split": several, "act": one, "chat": asks and not several and not one, "switch": bool(device and to), "device": device or "tv", "to": to or "on"}[name]


def shown(prompt, name):
    """What a message shows under `[[ ## name ## ]]`."""
    return prompt.split(f"[[ ## {name} ## ]]\n", 1)[-1].split("\n\n[[ ## ", 1)[0].split("\n\nRespond with", 1)[0]


def writes(messages):
    import json

    task = shown(messages[-1]["content"], "task")
    if "`parts`" in messages[0]["content"]:  # the parts of a request: where it has ", and " or a comma
        pieces = [piece.strip() for half in task.split(", and ") for piece in half.split(", ")]
        parts = [piece[0].upper() + piece[1:] + ("" if piece[-1] in ".?!" else ".") for piece in pieces if piece]
        return f"[[ ## parts ## ]]\n{json.dumps(parts)}\n\n[[ ## completed ## ]]"
    reply = next((reply for word, reply in REPLIES if word in task.lower()), "I can only tell you about the house.")
    return f"[[ ## reply ## ]]\n{reply}\n\n[[ ## completed ## ]]"
