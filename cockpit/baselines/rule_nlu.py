"""Strong traditional baseline (B1): regex NLU + the same scripted planner.

This is roughly what a production NLU + rules stack does. Its rules were
written against families A-C only; how it degrades on family D and on
unseen entities is part of the result.
"""
from __future__ import annotations

import re

from cockpit import spec
from cockpit.baselines.planner import ScriptedPolicy

CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
SPLIT_RE = re.compile(r"，然后|，再|，顺便|；|，另外|，同时|，对了，|，")
CAT_WORDS = {"充电站": "charging", "加油站": "gas", "停车场": "parking", "餐厅": "food"}


def _num(s):
    s = re.sub(r"一点|一些|一下|点", "", s)          # "小一点" carries no number
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    for ch in s:
        if ch in CN_NUM:
            return CN_NUM[ch]
    return None


def _zone(s):
    if re.search(r"主驾|驾驶位|我这边", s):
        return "driver"
    if "副驾" in s:
        return "passenger"
    return "all"


def parse_clause(c: str):
    c = c.strip()
    if not c:
        return None
    # --- climate power
    if re.search(r"(打开|开一下|开|帮我开)空调|空调(打开|开起来|开了)", c) and "度" not in c:
        return {"op": "AC_ON"}
    if re.search(r"(关闭|关掉|关一下|帮我关|关)空调|空调(关掉|关了)|不用开空调", c):
        return {"op": "AC_OFF"}
    # --- temperature
    if re.search(r"温度|空调|度", c) or re.search(r"热|冷", c):
        zone = _zone(c)
        if re.search(r"低|热", c):
            n = _num(c)
            return {"op": "TEMP_REL", "zone": zone, "delta": -(n or spec.DEFAULT_TEMP_STEP)}
        if re.search(r"高|冷", c):
            n = _num(c)
            return {"op": "TEMP_REL", "zone": zone, "delta": n or spec.DEFAULT_TEMP_STEP}
        n = _num(c)
        if n is not None:
            return {"op": "TEMP_ABS", "zone": zone, "value": n}
    # --- volume
    if re.search(r"音量|声音|太吵|听不清", c):
        if re.search(r"小|太吵", c):
            return {"op": "VOL_REL", "delta": -(_num(c) or spec.DEFAULT_VOLUME_STEP)}
        if re.search(r"大|听不清", c):
            return {"op": "VOL_REL", "delta": _num(c) or spec.DEFAULT_VOLUME_STEP}
        n = _num(c)
        if n is not None:
            return {"op": "VOL_ABS", "value": n}
    # --- navigation
    m = re.search(r"(最近|附近)的?(充电站|加油站|停车场|餐厅)", c)
    if m:
        return {"op": "NAV_NEAREST", "category": CAT_WORDS[m.group(2)]}
    m = re.search(r"(?:导航到|导航去|带我去|我要去|开去|去)(.+)", c)
    if m:
        return {"op": "NAV_BRAND", "brand": m.group(1).strip("吧呀啊。 ")}
    # --- media
    m = re.search(r"(?:播放|放一首|来一首|来首|放首|来点|放)(.+?)的(.+)", c)
    if m and m.group(2) not in ("歌", "歌曲", ""):
        return {"op": "PLAY_SONG", "artist": m.group(1), "title": m.group(2)}
    m = re.search(r"(?:播放|放首|来点|放)(.+?)的(?:歌曲|歌)?$", c)
    if m:
        return {"op": "PLAY_ARTIST", "artist": m.group(1)}
    m = re.search(r"(?:放一首|来一首|播放)(.+)", c)
    if m:
        return {"op": "PLAY_SONG", "title": m.group(1)}
    return None


QUESTION_RE = re.compile(r"怎么|如何|什么意思|多久|在哪|哪里|咋|要不要|能.*吗|还能")


def parse(text: str):
    ops = [o for o in (parse_clause(c) for c in SPLIT_RE.split(text)) if o]
    if ops:
        return ops                                   # vehicle commands win
    if QUESTION_RE.search(text):
        return [{"op": "MANUAL_QA", "query": text}]
    return [{"op": "CHAT"}]


class RulePolicy(ScriptedPolicy):
    def ops(self, task):
        return parse(task["text"])
