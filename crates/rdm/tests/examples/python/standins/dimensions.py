"""The rules of examples/dimensions.rs, and the cases its transcripts record."""

CASES = [
    ("", "", {}),
    ("cant-log-in", "--email 2", {"email": "I can't log in, it says my password is wrong."}),
    ("export-is-broken", "--email 3", {"email": "The export button does nothing since yesterday."}),
    ("team-pricing", "--email 4", {"email": "What does the team plan cost for twelve people?"}),
    ("other-words", '"Thanks: the new export is great."', {"email": "Thanks: the new export is great."}),
]
# How far a text is about each dimension, by the words in it: the likeliest of those it has.
ABOUT = {
    "money": [("charge", 0.93), ("refund", 0.93), ("invoice", 0.93), ("price", 0.5), ("cost", 0.5)],
    "access": [("password", 0.93), ("log in", 0.93), ("login", 0.93)],
    "fault": [("fault", 0.93), ("bug", 0.93), ("does nothing", 0.93), ("outage", 0.8)],
    "buying": [("price", 0.93), ("plan", 0.93), ("cost", 0.93), ("sales", 0.93)],
}
# The step an email asks for, by a word in it.
STEPS = [
    ("charge", "refund"),
    ("invoice", "send invoice"),
    ("password", "reset password"),
    ("locked out", "unlock account"),
    ("does nothing", "file a bug"),
    ("outage", "check status page"),
    ("cost", "quote a plan"),
    ("price", "quote a plan"),
    ("demo", "book a demo"),
]


def decides(inputs, name, question):
    email = inputs["email"].lower()
    if name in ABOUT:
        return max([likely for word, likely in ABOUT[name] if word in email], default=0.04)
    step = next((step for word, step in STEPS if word in email), None)
    return step if step in question["criteria"] and not inputs["observation"] else "finish"
