"""The rules of examples/triage.rs, and the cases its transcripts record."""

PRESETS = [
    "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    "The app logs me out every time I open the settings page. Otherwise I love it!",
    "Could you tell me what the team plan costs for twelve people?",
    "Just wanted to say thanks, the new export is great.",
]
CASES = [
    ("", "", {}),
    ("logged-out", "--message 2", {"message": PRESETS[1]}),
    ("team-pricing", "--message 3", {"message": PRESETS[2]}),
    ("thanks", "--message 4", {"message": PRESETS[3]}),
    ("other-words", '"Please refund my last invoice."', {"message": "Please refund my last invoice."}),
]


def decides(inputs, name, question):
    message = inputs["message"].lower()

    def says(*words):
        return any(word in message for word in words)

    if name == "refund":
        return says("money back", "refund")
    if name == "upset":
        return says("nobody answers", "money back")
    if says("charged", "refund", "invoice"):
        return "billing"
    if says("logs me out", "error", "crash"):
        return "technical support"
    if says("cost", "price", "plan"):
        return "sales"
    return "account management"
