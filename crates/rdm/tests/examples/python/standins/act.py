"""The rules of examples/act.rs, and the cases its transcripts record."""

CLOSED = {"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}
CASES = [
    ("", "", {}),
    ("lights-off", "--task 2", {"task": "Switch the lights off."}),
    ("shut-the-door", "--task 3", {"task": "Please shut the door."}),
    ("lovely-evening", "--task 4", {"task": "What a lovely evening."}),
    ("lock-refused", '"Lock the front door."', {"task": "Lock the front door."}),
    ("lock", '"Lock the front door." --house \'{"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}\'', {"task": "Lock the front door.", "house": CLOSED}),
]
DEVICES = [("lock", "lock"), ("tv", "tv"), ("light", "lights"), ("door", "door")]
STATES = [("unlock", "unlocked"), ("lock", "locked"), ("shut", "closed"), ("close", "closed"), ("open", "open"), (" off", "off"), (" on", "on")]


def decides(inputs, name, question):
    task = inputs["task"].lower()

    def named(words):
        return next((means for word, means in words if word in task), None)

    device, to = named(DEVICES), named(STATES)
    if name == "switch":
        return bool(device and to)
    return (device or "tv") if name == "device" else (to or "on")
