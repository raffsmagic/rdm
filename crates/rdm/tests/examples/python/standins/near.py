"""The rules of examples/near.rs, and the cases its transcripts record."""

CLOSED = {"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}
CASES = [
    ("", "", {}),
    ("tv-on", "--task 2", {"task": "Put the TV on."}),
    ("too-bright", "--task 3", {"task": "It is too bright in here."}),
    ("door-closed", '--house \'{"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}\'', {"house": CLOSED}),
    ("other-words", '"What a lovely evening."', {"task": "What a lovely evening."}),
]
ABOUT = {"screen": ["tv"], "light": ["light", "bright"], "entry": ["door", "lock"]}
DEVICES = [("lock", "lock"), ("tv", "tv"), ("light", "lights"), ("bright", "lights"), ("door", "door")]
STATES = [("lock", "locked"), ("shut", "closed"), ("close", "closed"), (" off", "off"), ("bright", "off"), (" on", "on")]


def decides(inputs, name, question):
    task = inputs["task"].lower()
    if name in ABOUT:
        return any(word in task for word in ABOUT[name])  # a label: is the text about this?

    def named(words):
        return next((means for word, means in words if word in task), None)

    device, to, offered = named(DEVICES), named(STATES), question["criteria"]
    if "refused, the door is open" in inputs["observation"] and "door closed" in offered:
        return "door closed"  # the house said what is in the way
    if f"{device} {to}" in offered and inputs["house"].get(device) != to:
        return f"{device} {to}"
    return "finish"
