"""Build train / dev / test splits.

Splits (reported separately, never averaged together):
  train, dev, test_id   same distribution: families A-C, training entities
  test_expr             unseen phrasing: family D only
  test_entity           unseen place names and artists
  test_combo            unseen combinations of skills that were each trained
Splitting happens on semantic tasks, before any paraphrasing, so a sentence
and its rewrite can never land on different sides.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cockpit.tasks.generator import (HELDOUT_COMBOS, TIER_WEIGHTS, TRAIN_FAMILIES,
                                     EXPR_OOD_FAMILY, make_task)


def _no_heldout_combo(names: frozenset) -> bool:
    return not any(c <= names for c in HELDOUT_COMBOS)


def _only_heldout_combo(names: frozenset) -> bool:
    return any(c <= names for c in HELDOUT_COMBOS)


def _tiers(rng, n):
    tiers, w = zip(*TIER_WEIGHTS.items())
    return rng.choices(tiers, weights=w, k=n)


def build_split(name: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(f"{name}-{seed}")
    out = []
    for i in range(n):
        tid = f"{name}-{i:05d}"
        if name in ("train", "dev", "test_id"):
            tier = _tiers(rng, 1)[0]
            t = make_task(rng, tier, rng.choice(TRAIN_FAMILIES),
                          allow_combo=_no_heldout_combo, task_id=tid)
        elif name == "test_expr":
            tier = _tiers(rng, 1)[0]
            t = make_task(rng, tier, EXPR_OOD_FAMILY, allow_combo=_no_heldout_combo, task_id=tid)
        elif name == "test_entity":
            tier = rng.choice(["L3", "L5"])
            t = make_task(rng, tier, rng.choice(TRAIN_FAMILIES), heldout_entities=True,
                          allow_combo=_no_heldout_combo, task_id=tid)
        elif name == "train_hard":                   # RL curriculum: dense, near-tie POIs
            tier = rng.choice(["L3", "L5", "L5"])
            t = make_task(rng, tier, rng.choice(TRAIN_FAMILIES), hard=True,
                          allow_combo=_no_heldout_combo, task_id=tid)
        elif name in ("train_manual", "test_manual"):
            fam = rng.choice(TRAIN_FAMILIES) if name == "train_manual" else EXPR_OOD_FAMILY
            t = make_task(rng, "manual", fam, task_id=tid)
        elif name == "test_combo":
            t = make_task(rng, "L5", rng.choice(TRAIN_FAMILIES),
                          allow_combo=_only_heldout_combo, task_id=tid, max_tries=2000)
        else:
            raise ValueError(name)
        out.append(t)
    return out


DEFAULT_SIZES = {"train": 3000, "dev": 300, "test_id": 300, "test_expr": 300,
                 "test_entity": 300, "test_combo": 300,
                 "train_manual": 300, "test_manual": 100}     # RAG module (extension)


def build_all(out_dir: str | Path, sizes=None, seed: int = 0) -> dict:
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, n in sizes.items():
        tasks = build_split(name, n, seed)
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        stats[name] = n
    return stats


def load_split(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
