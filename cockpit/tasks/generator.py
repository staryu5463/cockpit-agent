"""Task generator.

Order matters (see design doc):
  1. sample the *semantic* task (ops + params) and an initial state/catalog,
  2. compute the goal with code, never with an LLM,
  3. only then render natural language.
Paraphrasing (optional, cockpit/tasks/paraphrase.py) may only change wording.
"""
from __future__ import annotations

import copy
import random

from cockpit import spec
from cockpit.env import state as st
from cockpit.env.catalog import POI_CATEGORIES, make_catalog
from cockpit.tasks.templates import (CHAT_TEXTS, CONNECTORS, NUM_CN, TEMPLATES,
                                     ZONE_WORDS)

TRAIN_FAMILIES = ("A", "B", "C")
EXPR_OOD_FAMILY = "D"

OP_FIELDS = {
    "AC_ON": ["ac.power"], "AC_OFF": ["ac.power"],
    "TEMP_ABS": None, "TEMP_REL": None,                 # depends on zone
    "VOL_ABS": ["media.volume"], "VOL_REL": ["media.volume"],
    "NAV_NEAREST": ["nav.destination"], "NAV_BRAND": ["nav.destination"],
    "PLAY_SONG": ["media.playing"], "PLAY_ARTIST": ["media.playing"],
}
ABS_OPS = ["AC_ON", "AC_OFF", "TEMP_ABS", "VOL_ABS"]
REL_OPS = ["TEMP_REL", "VOL_REL"]
ENTITY_OPS = ["NAV_NEAREST", "NAV_BRAND", "PLAY_SONG", "PLAY_ARTIST"]

HELDOUT_COMBOS = [frozenset({"NAV_NEAREST", "VOL_REL"}),
                  frozenset({"PLAY_ARTIST", "TEMP_REL"}),
                  frozenset({"NAV_BRAND", "AC_OFF"})]

TIER_WEIGHTS = {"L1": .15, "L2": .15, "L3": .20, "L4": .15, "L5": .20,
                "clarify": .05, "unsupported": .05, "chat": .05}


def op_fields(op: dict) -> list[str]:
    if op["op"] in ("TEMP_ABS", "TEMP_REL"):
        z = op["zone"]
        return [f"ac.temp.{x}" for x in (spec.ZONES if z == "all" else (z,))]
    return OP_FIELDS[op["op"]]


# --------------------------------------------------------------------------
# sampling ops
# --------------------------------------------------------------------------
def _sample_op(rng, name, state, catalog):
    if name == "AC_ON":
        state["ac.power"] = False
        return {"op": "AC_ON"}
    if name == "AC_OFF":
        state["ac.power"] = True
        return {"op": "AC_OFF"}
    if name == "TEMP_ABS":
        return {"op": "TEMP_ABS", "zone": rng.choice(["driver", "passenger", "all"]),
                "value": rng.randint(spec.TEMP_MIN, spec.TEMP_MAX)}
    if name == "TEMP_REL":
        zone = rng.choice(["driver", "passenger", "all"])
        dflt = rng.random() < 0.3
        delta = (spec.DEFAULT_TEMP_STEP if dflt else rng.randint(1, 4)) * rng.choice([-1, 1])
        return {"op": "TEMP_REL", "zone": zone, "delta": delta, "dflt": dflt}
    if name == "VOL_ABS":
        return {"op": "VOL_ABS", "value": rng.randint(spec.VOLUME_MIN, spec.VOLUME_MAX)}
    if name == "VOL_REL":
        dflt = rng.random() < 0.35
        delta = (spec.DEFAULT_VOLUME_STEP if dflt else rng.randint(1, 5)) * rng.choice([-1, 1])
        return {"op": "VOL_REL", "delta": delta, "dflt": dflt}
    if name == "NAV_NEAREST":
        cats = [c for c in POI_CATEGORIES
                if any(p["category"] == c and p["open"] for p in catalog["pois"])]
        return {"op": "NAV_NEAREST", "category": rng.choice(cats)} if cats else None
    if name == "NAV_BRAND":
        brands = sorted({b for c, (_, bs) in POI_CATEGORIES.items() for b in bs
                         if any(b in p["name"] and p["open"] for p in catalog["pois"])})
        return {"op": "NAV_BRAND", "brand": rng.choice(brands)} if brands else None
    if name == "PLAY_SONG":
        m = rng.choice(catalog["media"])
        return {"op": "PLAY_SONG", "title": m["title"], "artist": m["artist"]}
    if name == "PLAY_ARTIST":
        return {"op": "PLAY_ARTIST", "artist": rng.choice(sorted({m["artist"] for m in catalog["media"]}))}
    raise ValueError(name)


# --------------------------------------------------------------------------
# goal computation (pure code)
# --------------------------------------------------------------------------
def _apply(op, s, catalog):
    """Apply op to state dict `s`. Returns {field: expected} or None if impossible."""
    o = op["op"]
    if o == "AC_ON":
        return {"ac.power": True}
    if o == "AC_OFF":
        return {"ac.power": False}
    if o == "TEMP_ABS":
        return {f: op["value"] for f in op_fields(op)}
    if o == "TEMP_REL":
        out = {f: s[f] + op["delta"] for f in op_fields(op)}
        return out if all(spec.TEMP_MIN <= v <= spec.TEMP_MAX for v in out.values()) else None
    if o == "VOL_ABS":
        return {"media.volume": op["value"]}
    if o == "VOL_REL":
        v = s["media.volume"] + op["delta"]
        return {"media.volume": v} if spec.VOLUME_MIN <= v <= spec.VOLUME_MAX else None
    if o in ("NAV_NEAREST", "NAV_BRAND"):
        if o == "NAV_NEAREST":
            cands = [p for p in catalog["pois"] if p["category"] == op["category"] and p["open"]]
        else:
            cands = [p for p in catalog["pois"] if op["brand"] in p["name"] and p["open"]]
        if not cands:
            return None
        d = min(p["distance_km"] for p in cands)
        best = [p for p in cands if p["distance_km"] == d]
        if len(best) > 1 and len({p["name"] for p in best}) == 1:
            return None                      # ambiguous: only the clarify tier may do this
        return {"nav.destination": {"any_of": [p["id"] for p in best]}}
    if o == "PLAY_SONG":
        ids = [m["id"] for m in catalog["media"] if m["title"] == op["title"]]
        return {"media.playing": {"any_of": ids}}
    if o == "PLAY_ARTIST":
        ids = [m["id"] for m in catalog["media"] if m["artist"] == op["artist"]]
        return {"media.playing": {"any_of": ids}}
    raise ValueError(o)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _render_op(rng, op, family):
    o = op["op"]
    if o in ("TEMP_REL", "VOL_REL") and op.get("dflt"):
        tpl = rng.choice(TEMPLATES[o + "_dflt"][family])
    else:
        tpl = rng.choice(TEMPLATES[o][family])
    zone = rng.choice(ZONE_WORDS[op["zone"]]) if "zone" in op else ""
    kw = {"zone": zone}
    if o in ("TEMP_ABS", "VOL_ABS"):
        kw["v"] = op["value"]
    if o == "TEMP_REL":
        kw.update(dir="低" if op["delta"] < 0 else "高", n=NUM_CN[abs(op["delta"])],
                  feel="热" if op["delta"] < 0 else "冷")
    if o == "VOL_REL":
        kw.update(dir="小" if op["delta"] < 0 else "大", n=NUM_CN.get(abs(op["delta"]), str(abs(op["delta"]))),
                  feel="太吵了" if op["delta"] < 0 else "听不清",
                  dirfeel="大" if op["delta"] < 0 else "小")
    if o == "NAV_NEAREST":
        kw["cat"] = POI_CATEGORIES[op["category"]][0]
    if o == "NAV_BRAND":
        kw["brand"] = op["brand"]
    if o in ("PLAY_SONG", "PLAY_ARTIST"):
        kw["artist"] = op["artist"]
        kw["title"] = op.get("title", "")
    text = tpl.format(**kw)
    return text.replace("  ", " ").strip()


def _render(rng, ops, family):
    parts = [_render_op(rng, op, family) for op in ops]
    text = parts[0]
    for p in parts[1:]:
        text += rng.choice(CONNECTORS[family]) + p
    return text


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def make_task(rng: random.Random, tier: str, family: str, *, heldout_entities=False, hard=False,
              allow_combo=lambda names: True, task_id="t", max_tries=50) -> dict:
    for _ in range(max_tries):
        seed = rng.randint(0, 2**31 - 1)
        trng = random.Random(seed)
        catalog = make_catalog(trng, heldout=heldout_entities, hard=hard)
        init = st.random_state(trng)
        t = _try_make(trng, tier, family, init, catalog, allow_combo)
        if t is not None:
            t.update(id=task_id, group_id=task_id, seed=seed, tier=tier, family=family,
                     heldout_entities=heldout_entities)
            return t
    raise RuntimeError(f"could not sample tier={tier}")


def _pick_names(trng, tier):
    if tier == "L1":
        return [trng.choice(ABS_OPS)]
    if tier == "L2":
        return trng.sample(ABS_OPS, 2)
    if tier == "L3":
        return [trng.choice(ENTITY_OPS)]
    if tier == "L4":
        return [trng.choice(REL_OPS)]
    if tier == "L5":
        k = trng.choice([2, 3])
        names = [trng.choice(ENTITY_OPS), trng.choice(REL_OPS)]
        if k == 3:
            names.append(trng.choice(ABS_OPS + ENTITY_OPS))
        trng.shuffle(names)
        return names
    raise ValueError(tier)


def _try_make(trng, tier, family, init, catalog, allow_combo):
    if tier == "chat":
        text = trng.choice(CHAT_TEXTS[family])
        return _pack(text, init, catalog, [{"op": "CHAT"}], {}, spec.FINISH_CHAT)
    if tier == "unsupported":
        if trng.random() < 0.6:
            op = {"op": "TEMP_ABS", "zone": trng.choice(["driver", "passenger", "all"]),
                  "value": trng.choice([5, 10, 12, 35, 38, 40])}
        else:
            op = {"op": "VOL_ABS", "value": trng.choice([35, 40, 50, 99])}
        text = _render_op(trng, op, family)
        return _pack(text, init, catalog, [dict(op, unsupported=True)], {}, spec.FINISH_UNSUPPORTED)
    if tier == "clarify":
        return _make_clarify(trng, family, init, catalog)
    if tier == "manual":
        from cockpit.rag.manual import MANUAL_QUESTIONS
        mid = trng.choice(sorted(MANUAL_QUESTIONS))
        text = trng.choice(MANUAL_QUESTIONS[mid][family])
        t = _pack(text, init, catalog, [{"op": "MANUAL_QA", "query": text}], {}, spec.FINISH_DONE)
        t["goal"]["must_cite"] = mid
        t["enable_manual"] = True
        return t

    names = _pick_names(trng, tier)
    if len(set(names)) < len(names) or not allow_combo(frozenset(names)):
        return None
    state = st.clone(init)
    ops = []
    for n in names:
        op = _sample_op(trng, n, state, catalog)
        if op is None:
            return None
        ops.append(op)
    fields = [f for op in ops for f in op_fields(op)]
    if len(fields) != len(set(fields)):
        return None                                  # two ops touch one field
    init = state                                     # AC_ON/OFF may have adjusted power
    goal, s = {}, st.clone(init)
    for op in ops:
        g = _apply(op, s, catalog)
        if g is None:
            return None
        goal.update(g)
    faults = {}
    if tier in ("L3", "L5") and trng.random() < 0.15:
        tool = next(("nav_start" if o["op"].startswith("NAV") else "media_play")
                    for o in ops if o["op"] in ENTITY_OPS)
        faults = {tool: 1}
    text = _render(trng, ops, family)
    t = _pack(text, init, catalog, ops, goal, spec.FINISH_DONE)
    t["faults"] = faults
    return t


def _make_clarify(trng, family, init, catalog):
    """Two open POIs with the same name and distance -> the agent must ask."""
    opens = [p for p in catalog["pois"] if p["open"] and p["category"] != "parking"]
    if not opens:
        return None
    p = trng.choice(opens)
    brand = next(b for _, (_, bs) in POI_CATEGORIES.items() for b in bs if b in p["name"])
    same_brand = [q for q in catalog["pois"] if brand in q["name"] and q["open"]]
    if min(q["distance_km"] for q in same_brand) != p["distance_km"]:
        return None                       # p must be the nearest so the tie matters
    twin = copy.deepcopy(p)
    twin["id"] = f"p{trng.randint(1000, 9999)}"
    if any(q["id"] == twin["id"] for q in catalog["pois"]):
        return None
    catalog["pois"].insert(trng.randrange(len(catalog["pois"]) + 1), twin)
    op = {"op": "NAV_BRAND", "brand": brand}
    text = _render_op(trng, op, family)
    return _pack(text, init, catalog, [dict(op, clarify=True)], {}, spec.FINISH_CLARIFY)


def _pack(text, init, catalog, ops, goal_fields, status):
    return {"text": text, "init_state": init, "catalog": catalog, "semantic": ops,
            "goal": {"fields": goal_fields, "finish_status": status}, "faults": {}}
