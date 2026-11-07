"""The rules of examples/handbook.rs, and the cases its transcripts record."""

PRESETS = [
    "What is the nightly hotel limit in London?",
    "How much can I spend on meals per day?",
    "When can I take a taxi?",
    "How often can I claim home office furniture?",
]
CASES = [
    ("", "", {}),
    ("meals", "--question 2", {"question": PRESETS[1]}),
    ("taxi", "--question=3", {"question": PRESETS[2]}),
    ("furniture", "4", {"question": PRESETS[3]}),
    ("not-there", '"Is the sky blue?"', {"question": "Is the sky blue?"}),
]
# What a question is about: a word of it, words of the line that states its answer, and the answer as a
# language model might write it from that line. The first is what the site's own mock writes.
ANSWERS = [
    ("hotel", "260 euros in London", "260 euros a night in London."),
    ("meals", "45 euros per day", "Up to 45 euros per day when travelling."),
    ("taxi", "Taxis are reimbursed", "When travelling with equipment, or after 22:00."),
    ("furniture", "home office furniture", "Once every two years, up to 400 euros."),
]


def decides(inputs, name, question):
    asked = inputs["question"].lower()
    return any(word in asked and line in inputs["text"] for word, line, _ in ANSWERS)


def shown(prompt, name):
    """What a message shows under `[[ ## name ## ]]`."""
    return prompt.split(f"[[ ## {name} ## ]]\n", 1)[-1].split("\n\n[[ ## ", 1)[0].split("\n\nRespond with", 1)[0]


def writes(messages):
    prompt = messages[-1]["content"]
    asked, passage = shown(prompt, "question").lower(), shown(prompt, "passage")
    answer = next((written for word, line, written in ANSWERS if word in asked and line in passage), passage)
    return f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
