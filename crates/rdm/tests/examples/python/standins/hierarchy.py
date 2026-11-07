"""The rules of examples/hierarchy.rs, and the cases its transcripts record.

The Rust example also prints each step of the walk, as the post's version of this search does, where the cell
draws it (`show_branch`). So `same` says its lines differ, by those lines and no others; the requests are the same.
"""

# `record` would write the cell's five lines over transcripts that hold the steps as well.
RECORDED_FROM_RUST = "the Rust example also prints each step of the walk, as the post does, and the cell does not."

PRESETS = [
    "Laptop backpack with a padded sleeve, 20 litres.",
    "A board game for families: a strategy game for 2 to 4 players.",
    "Hiking backpack, 40 litres, with a rain cover.",
    "Furniture listing: a wall-mounted window shelf bed. This padded floating shelf uses suction cups and a washable cushion as a sunny perch for one cat.",
    "Wireless over-ear headphones with active noise cancelling, 30 hours of battery and a folding case.",
]
CASES = [
    ("", "", {}),
    ("board-game", "--listing 2", {"listing": PRESETS[1]}),
    ("hiking-backpack", "--listing=3", {"listing": PRESETS[2]}),
    ("cat-window-perch", "4", {"listing": PRESETS[3]}),
    ("headphones", "5", {"listing": PRESETS[4]}),
    ("other-words", '"A cast-iron frying pan."', {"listing": "A cast-iron frying pan."}),
]
# Where a listing belongs, by a word of it.
LEAVES = [
    ("laptop", "Luggage & Bags > Backpacks > Laptop Backpacks"),
    ("hiking", "Luggage & Bags > Backpacks > Hiking Backpacks"),
    ("board game", "Toys & Games > Games > Board Games"),
    ("cat", "Animals & Pet Supplies > Pet Supplies > Cat Supplies > Cat Furniture > Cat Window Beds & Perches"),
    ("headphones", "Electronics > Audio > Audio Components > Headphones & Headsets > Headphones > Over-Ear Headphones"),
]
# What a listing that says "bed" could be as well: where the post's own run branched.
BEDS = "Animals & Pet Supplies > Pet Supplies > Pet Beds"
ALSO = [BEDS] + [f"{BEDS} > {bed}" for bed in ("Hammocks", "Pet Cots", "Pillow Beds", "Radiator Beds")]


def decides(inputs, name, question):
    listing, options = inputs["listing"].lower(), list(question["criteria"])
    leaf = next((leaf for word, leaf in LEAVES if word in listing), "")
    # The option on the way to the leaf, and the options it could be as well.
    way = next((option for option in options if leaf == option or leaf.startswith(f"{option} > ")), None)
    also = [option for option in options if option in ALSO] if "bed" in listing else []
    if also:
        odds = {option: (1.0 if way is None else 0.4) / len(also) for option in also}
        return odds if way is None else {way: 0.6, **odds}
    return way or ("Uncategorized" if "Uncategorized" in options else options[0])
