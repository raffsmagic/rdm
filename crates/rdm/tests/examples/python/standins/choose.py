"""The rules of examples/choose.rs, and the cases its transcripts record."""

PRESETS = [
    "A padded shelf that suction-cups to a window, so a cat can sun itself.",
    "A rope tug toy for large dogs.",
    "A chef's knife with a 20 cm blade.",
    "A three-seat sofa in grey fabric.",
]
CASES = [
    ("", "", {}),
    ("dog-tug-toy", "--listing 2", {"listing": PRESETS[1]}),
    ("chefs-knife", "--listing=3", {"listing": PRESETS[2]}),
    ("sofa", "4", {"listing": PRESETS[3]}),
    ("other-words", '"A cast-iron frying pan."', {"listing": "A cast-iron frying pan."}),
]
# The words of a listing that put it in a category.
WORDS = {
    "Animals & Pet Supplies": ["cat", "dog"],
    "Furniture": ["sofa", "shelf", "bed"],
    "Kitchen": ["knife", "pan", "plate"],
    "Cat Supplies": ["cat"],
    "Dog Supplies": ["dog"],
    "Cat Beds": ["bed"],
    "Cat Toys": ["toy"],
    "Cat Window Perches": ["window"],
    "Dog Beds": ["bed"],
    "Dog Toys": ["toy"],
    "Dog Leashes": ["leash"],
    "Shelves": ["shelf"],
    "Beds": ["bed"],
    "Sofas": ["sofa"],
    "Cookware": ["pan"],
    "Tableware": ["plate"],
    "Kitchen Knives": ["knife"],
}


def decides(inputs, name, question):
    listing, options = inputs["listing"].lower(), list(question["criteria"])
    return next((option for option in options if any(word in listing for word in WORDS.get(option, []))), options[0])
