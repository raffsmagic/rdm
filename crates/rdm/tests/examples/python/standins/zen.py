"""The rules of examples/zen.rs, and the cases its transcripts record."""

CASES = [
    ("", "", {}),
    ("namespaces", "--question 2", {"question": "What does the text say about namespaces?"}),
    ("honking", "--question=3", {"question": "What is one honking great idea?"}),
    ("obvious", "4", {"question": "How many obvious ways should there be to do something?"}),
    ("not-there", '"Is the sky blue?"', {"question": "Is the sky blue?"}),
]
ANSWERS = [
    ("flat", "Flat is better than nested."),
    ("namespaces", "Namespaces are one honking great idea"),
    ("honking", "Namespaces are one honking great idea"),
    ("obvious", "preferably only one"),
]


def decides(inputs, name, question):
    asked = inputs["question"].lower()
    return any(word in asked and line in inputs["text"] for word, line in ANSWERS)
