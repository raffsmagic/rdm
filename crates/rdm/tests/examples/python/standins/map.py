"""The rules of examples/map.rs, and the cases its transcripts record.

These cases are the notebook's cell, which is what the Python cell prints. `map.post.txt` is the map as
the post writes it (`RDM_POST=1`), and is recorded from the example itself: `decides` answers its
requests too, which show a `request` and no `task`.
"""

CASES = [
    ("", "", {}),
    ("everything-off", "--first 2", {"first": "Turn everything off."}),
    ("movie-night", "--routine 2", {"routine": "Movie night: TV on, lights off."}),
    ("time-for-bed", "--again 2", {"again": "Time for bed."}),
    (
        "door-closed",
        '--house \'{"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}\'',
        {"house": {"tv": "off", "lights": "on", "door": "closed", "lock": "unlocked"}},
    ),
    ("other-words", '"What a lovely evening."', {"first": "What a lovely evening."}),
]
NIGHT = ["tv off", "lights off", "lock locked"]
WANTS = [("movie", ["tv on", "lights off"]), ("film", ["tv on", "lights off"]), ("night", NIGHT), ("bed", NIGHT), ("everything off", ["tv off", "lights off"]), ("lock", ["lock locked"])]


def decides(inputs, name, question):
    asked, house = inputs.get("task", inputs.get("request")).lower(), inputs["house"]
    wanted = next((steps for word, steps in WANTS if word in asked), [])
    unmet = [step for step in wanted if house.get(step.split()[0]) != step.split()[1]]
    if "refused, the door is open" in inputs.get("observation", ""):
        return "door closed"  # the house said what is in the way
    for kept in ("good night", "movie time"):  # the cell's kept name, and the post's
        says = (question["criteria"].get(kept) or "").lower()
        if len([step for step in unmet if step in says]) > 1:
            return kept  # one name that does several of the steps still to take
    return unmet[0] if unmet else "stop" if "request" in inputs else "finish"
