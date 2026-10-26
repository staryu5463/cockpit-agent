"""Vehicle state: a flat dict of public fields.

Flat keys ("ac.temp.driver") make diffs, goals and side-effect checks trivial.
"""
from __future__ import annotations

import copy

FIELDS = (
    "ac.power",            # bool
    "ac.temp.driver",      # int, 16..32
    "ac.temp.passenger",   # int, 16..32
    "media.playing",       # item id or None
    "media.volume",        # int, 0..30
    "nav.destination",     # poi id or None
)


def default_state() -> dict:
    return {
        "ac.power": False,
        "ac.temp.driver": 24,
        "ac.temp.passenger": 24,
        "media.playing": None,
        "media.volume": 10,
        "nav.destination": None,
    }


def random_state(rng) -> dict:
    s = default_state()
    s["ac.power"] = rng.random() < 0.6
    s["ac.temp.driver"] = rng.randint(18, 28)
    s["ac.temp.passenger"] = rng.randint(18, 28)
    s["media.volume"] = rng.randint(4, 24)
    return s


def diff(before: dict, after: dict) -> dict:
    """Fields whose value changed: {field: (before, after)}."""
    return {k: (before[k], after[k]) for k in FIELDS if before.get(k) != after.get(k)}


def clone(state: dict) -> dict:
    return copy.deepcopy(state)
