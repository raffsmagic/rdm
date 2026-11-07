"""The rules of examples/starter.rs, and the cases its transcripts record."""

HOME = {"tv": "off", "lights": "off", "door": "closed", "lock": "locked", "heat": 18}
CASES = [
    ("", "", {}),
    ("other-words", '"Movie night!" "Good night, all." "What a lovely evening."', {"tasks": ["Movie night!", "Good night, all.", "What a lovely evening."]}),
    ("asleep", '"Morning!" --house \'{"tv": "off", "lights": "off", "door": "closed", "lock": "locked", "heat": 18}\'', {"tasks": ["Morning!"], "house": HOME}),
]
ROUTINES = [(("film", "movie"), "movie_time"), (("bed", "night"), "good_night"), (("morning",), "good_morning")]


def decides(inputs, name, question):
    task = inputs["task"].lower()
    routine = next((routine for words, routine in ROUTINES if any(word in task for word in words)), "finish")
    return "finish" if inputs["observation"] or routine not in question["criteria"] else routine
