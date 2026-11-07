"""The rules of examples/outline.rs, and the cases its transcripts record."""

PRESETS = [
    "Can I work from another country?",
    "What is the nightly hotel limit in London?",
    "Who do I tell about a lost laptop?",
    "How many days of annual leave are there?",
]
CASES = [
    ("", "", {}),
    ("london", "--question 2", {"question": PRESETS[1]}),
    ("lost-laptop", "--question=3", {"question": PRESETS[2]}),
    ("annual-leave", "4", {"question": PRESETS[3]}),
    ("not-there", '"Is the sky blue?"', {"question": "Is the sky blue?"}),
]
# What a question is about: a word of it, the heading its answer is under, and how the passage that states it begins.
ANSWERS = [
    ("another country", "Working from abroad", "Working from another country"),
    ("hotel", "Hotels", "The nightly limit for accommodation"),
    ("lost", "Reporting an incident", "Report a lost device"),
    ("annual leave", "Annual leave", "Full-time staff receive 25 days"),
]


def decides(inputs, name, question):
    asked, entry, contains = inputs["question"].lower(), inputs["entry"], inputs["contains"]
    return any(word in asked and (entry == heading or heading in contains or entry.startswith(passage)) for word, heading, passage in ANSWERS)
