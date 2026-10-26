"""Synthetic POI and media catalogs.

Each task gets its own catalog drawn from a seeded RNG, so IDs, ordering and
which candidate is nearest all change between tasks. A policy cannot memorise
"the answer is p3"; it has to read the search results.
"""
from __future__ import annotations

import random

POI_CATEGORIES = {
    "charging": ("充电站", ["特来电", "星星充电", "国家电网", "小桔充电", "蔚来超充", "云快充"]),
    "gas":      ("加油站", ["中石化", "中石油", "壳牌", "道达尔", "延长壳牌"]),
    "parking":  ("停车场", ["万达广场停车场", "人民公园停车场", "高铁站停车场", "龙湖天街停车场"]),
    "food":     ("餐厅",   ["海底捞", "西贝莜面村", "外婆家", "麦当劳", "肯德基", "太二酸菜鱼"]),
}
AREAS_TRAIN = ["徐汇", "静安", "浦东", "闵行", "长宁", "普陀", "杨浦", "虹口"]
AREAS_HELDOUT = ["西湖", "滨江", "拱墅", "萧山"]          # entity-OOD split only

ARTISTS_TRAIN = {
    "周杰伦": ["晴天", "稻香", "七里香", "夜曲", "告白气球"],
    "陈奕迅": ["十年", "浮夸", "好久不见", "富士山下"],
    "林俊杰": ["江南", "修炼爱情", "不为谁而作的歌"],
    "邓紫棋": ["光年之外", "泡沫", "倒数"],
    "五月天": ["倔强", "突然好想你", "温柔"],
    "孙燕姿": ["遇见", "绿光", "天黑黑"],
}
ARTISTS_HELDOUT = {                                       # entity-OOD split only
    "李健": ["贝加尔湖畔", "传奇"],
    "毛不易": ["消愁", "像我这样的人"],
    "朴树": ["平凡之路", "那些花儿"],
}


def make_catalog(rng: random.Random, heldout: bool = False, hard: bool = False) -> dict:
    """Return {"pois": [...], "media": [...]} with fresh random IDs.

    hard=True (training curriculum only): more candidates per category, the same
    brand in several areas, distances packed within ~2 km, more closed places.
    Picking "the nearest open one" then needs careful comparison, which is the
    main failure mode of the SFT model on held-out combinations.
    """
    areas = AREAS_HELDOUT if heldout else AREAS_TRAIN
    artists = ARTISTS_HELDOUT if heldout else ARTISTS_TRAIN

    pois = []
    for cat, (cat_cn, brands) in POI_CATEGORIES.items():
        if hard:
            picks = [b for b in brands for _ in range(rng.choice([1, 2]))][:8]
            base = rng.uniform(0.5, 6.0)
        else:
            picks = rng.sample(brands, k=min(rng.randint(3, 5), len(brands)))
        used_names = set()
        for brand in picks:
            area = rng.choice([a for a in areas if f"{brand}({a}{cat_cn})" not in used_names] or areas)
            name = f"{brand}({area}{cat_cn})" if cat != "parking" else f"{brand}"
            used_names.add(name)
            pois.append({
                "id": f"p{rng.randint(1000, 9999)}",
                "name": name,
                "category": cat,
                "distance_km": round(base + rng.uniform(0.0, 2.0), 1) if hard else round(rng.uniform(0.3, 15.0), 1),
                "open": rng.random() < (0.65 if hard else 0.8),
            })
    _dedupe_ids(pois, rng, "p")

    media = []
    for artist, songs in artists.items():
        for title in songs:
            media.append({"id": f"m{rng.randint(1000, 9999)}", "title": title,
                          "artist": artist, "type": "song"})
    _dedupe_ids(media, rng, "m")
    rng.shuffle(pois)
    rng.shuffle(media)
    return {"pois": pois, "media": media}


def _dedupe_ids(items, rng, prefix):
    seen = set()
    for it in items:
        while it["id"] in seen:
            it["id"] = f"{prefix}{rng.randint(1000, 9999)}"
        seen.add(it["id"])
