"""The rules of examples/fit.rs, and the cases its transcripts record."""

PRESETS = [
    "What is the nightly hotel limit in London?",
    "How many days of annual leave are there?",
    "Who do I tell about a lost laptop?",
]
CASES = [
    ("", "", {}),
    ("annual-leave", "--question 2", {"question": PRESETS[1]}),
    ("lost-laptop", "--question=3", {"question": PRESETS[2]}),
    ("not-there", '"Is the sky blue?"', {"question": "Is the sky blue?"}),
]
# What a question is about: a word of it, and words of the line that states its answer.
ANSWERS = [
    ("hotel", "260 euros in London"),
    ("annual leave", "25 days of annual leave"),
    ("lost", "Report a lost device"),
]


def decides(inputs, name, question):
    # The question is `inputs.question` of the search, and of a part `Fit` asks about, among `inputs.questions`.
    asked = str(inputs.get("question", inputs.get("questions"))).lower()
    return any(word in asked and line in inputs["text"] for word, line in ANSWERS)
