"""The rules of examples/home.rs, and the cases its transcripts record.

Each case is one request to the house as the cell starts it, which is what the Python cell runs.
`home.several.txt` is several requests in turn to one house, and is recorded from the example itself.
"""

import re

BLOCKED = {
    "entry": {"lights": "on", "door": "open", "lock": "unlocked", "alarm": "off"},
    "living room": {"lights": "on", "tv": "off", "fan": "on", "window": "closed"},
    "kitchen": {"lights": "on", "oven": "on", "coffee maker": "off", "dishwasher": "on"},
    "bedroom": {"lights": "off", "lamp": "on", "window": "open"},
    "bathroom": {"lights": "off", "fan": "off"},
    "garage": {"lights": "on", "door": "open", "doorway": "blocked", "washer": "off", "dryer": "on"},
    "thermostat": {"mode": "off", "target": 22, "temp": 28, "outside": 28},
}
PRESETS = [
    ("", "We are off to bed."),
    ("morning", "Morning!"),
    ("tv-on", "Turn the TV on."),
    ("garage", "Close the garage door."),
    ("weekend", "We are leaving for the weekend."),
    ("film", "Let's watch a film."),
    ("oven", "I left the oven on."),
    ("20-degrees", "Set the thermostat to 20 degrees."),
    ("lock", "Lock the front door."),
    ("too-hot", "It is too hot in here."),
    ("alarm", "Set the alarm."),
]
CASES = [(case, f"--task {at + 1}" if at else "", {"task": task} if at else {}) for at, (case, task) in enumerate(PRESETS)] + [
    ("doorway-blocked", f"--task 4 --house '{__import__('json').dumps(BLOCKED)}'", {"task": "Close the garage door.", "house": BLOCKED}),
    ("other-words", '"What a lovely evening."', {"task": "What a lovely evening."}),
]

# A routine the house knows, by a word of the request.
ROUTINES = [("bed", "good night"), ("morning", "good morning"), ("weekend", "leaving"), ("leaving", "leaving"), ("film", "movie time"), ("movie", "movie time")]
# One thing to set, by the words of the request: the room it is in, and the step there.
WANTS = [
    (["alarm"], "entry", "alarm armed"),
    (["hot"], "thermostat", "mode cool"),
    (["cold"], "thermostat", "mode heat"),
    (["garage"], "garage", "door closed"),
    (["oven"], "kitchen", "oven off"),
    (["tv", "off"], "living room", "tv off"),
    (["tv"], "living room", "tv on"),
    (["lock"], "entry", "lock locked"),
]
# What the house says is in the way, and the step that puts it right: the room, and the step there.
FIXES = [
    ("refused while the front door is not locked", "entry", "lock locked"),
    ("refused while the door is open", "entry", "door closed"),
    ("refused while the garage door is open", "garage", "door closed"),
    ("refused while the oven is on", "kitchen", "oven off"),
]


def decides(inputs, name, question):
    """It is shown no house, as a model is not: it reads what the last step did."""
    task, last, offered = inputs["task"].lower(), inputs["observation"].split("; ")[-1], list(question["criteria"])
    words = re.findall(r"[a-z0-9]+", task)
    want = next(((room, step) for needs, room, step in WANTS if all(word in words for word in needs)), None)
    at = words.index("degrees") if "degrees" in words else 0
    if not want and at:
        want = ("thermostat", f"target {words[at - 1]}")

    def fixing(said, rooms):
        """The room to go to and the step to take there, for what the house said is in the way."""
        fix = next(((room, step) for refusal, room, step in FIXES if refusal in said), None)
        return fix or next(((room, "window closed") for room in rooms if f"refused while the {room} window is open" in said), None)

    def done(said):
        thing, to = want[1].rsplit(" ", 1)
        return f"{thing}: now {to}" in said or f"{thing}: already {to}" in said

    if "before" not in inputs:
        # The house's map: a routine where the request is one, or the room to go to next.
        routine = next((name for word, name in ROUTINES if word in words and name in offered), None)
        if routine:
            return "finish" if last else routine
        fix = fixing(last, offered)
        if fix and fix[0] in offered:
            return fix[0]
        return want[0] if want and not done(last) and "refused" not in last and want[0] in offered else "finish"
    # A room's map. Its descriptions say which room it is: "Set the entry lights to on."
    room = re.match(r"^Set the (.*) \S+ to \S+\.$", question["criteria"][offered[0]])[1]
    fix = fixing(last or inputs["before"].split("; ")[-1], [room])
    if fix and fix[0] == room and fix[1] in offered:
        return fix[1]
    if "refused" in last or not want or want[0] != room or done(last) or want[1] not in offered:
        return "finish"
    return want[1]
