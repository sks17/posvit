"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
IN-9 Backgrounds-Challenge data: acquisition, verification, loaders, and masks.
"""

from __future__ import annotations
import json
from posvit import paths

# The nine IN-9 class folders, in canonical (index) order. Classes 02 and 06 really do
# contain spaces in the released dataset, so the names are not normalized.
CANONICAL_CLASSES = (
    "00_dog",
    "01_bird",
    "02_wheeled vehicle",
    "03_reptile",
    "04_carnivore",
    "05_insect",
    "06_musical instrument",
    "07_primate",
    "08_fish",
)

def load_in9_map() -> dict[str, int]:
    """
    Loads the ImageNet-to-IN9 class map.
    - Pre: `in9_map.json` exists and contains valid JSON object data.
    - Post: Returns a dictionary mapping string class ids to integer IN9 classes.
    """
    with open(paths.in9_map_path(), encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def map_to_in9(pred1000, in9_map: dict[str, int]) -> int:
    """
    Maps a 1000-way ImageNet prediction into the nine IN-9 classes.

    Only 370 of the 1000 ImageNet classes belong to an IN-9 group; the remaining 630
    map to -1 and are therefore always scored as incorrect. Callers record how often
    that happens, because "wrong IN-9 class" and "left the benchmark entirely" are
    different failures.
    - Pre: `pred1000` is an integer-valued ImageNet class index and `in9_map` has
        string keys.
    - Post: Returns the IN-9 class in 0..8, or -1 when the class is outside IN-9.
    """
    return in9_map.get(str(int(pred1000)), -1)