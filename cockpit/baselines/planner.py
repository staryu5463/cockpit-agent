"""Scripted planner: turns a list of normalized ops into tool calls.

It only learns things through tools (reads state, searches, retries), exactly
like a model would. Two policies share it:
  ExpertPolicy  ops come from the task's private semantic intent (oracle NLU)
  RulePolicy    ops come from a regex NLU over the raw text (strong baseline B1)
The expert's trajectories are the SFT data.
"""
from __future__ import annotations

import json

from cockpit import spec
from cockpit.agent import format as F
from cockpit.agent.format import format_tool_call, format_turn
from cockpit.env.catalog import POI_CATEGORIES


def plan(ops):
    """Generator: yields (tool, args), receives the tool result envelope."""
    def call(name, args):
        res = yield (name, args)
        tries = 0
        while not res["ok"] and res.get("retryable") and tries < 2:
            tries += 1
            res = yield (name, args)
        return res

    if not ops or ops[0]["op"] == "CHAT":
        yield from call("finish", {"status": spec.FINISH_CHAT, "message": "我在呢，有需要随时叫我。"})
        return

    def failed(res):
        return not res["ok"]

    def give_up():
        return call("finish", {"status": spec.FINISH_UNSUPPORTED, "message": "这个操作暂时无法完成。"})

    done_msgs = []
    for op in ops:
        o = op["op"]
        if o == "AC_ON":
            yield from call("ac_power", {"on": True}); done_msgs.append("空调已打开")
        elif o == "AC_OFF":
            yield from call("ac_power", {"on": False}); done_msgs.append("空调已关闭")
        elif o == "TEMP_ABS":
            if not spec.TEMP_MIN <= op["value"] <= spec.TEMP_MAX:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED,
                                           "message": f"温度只能设在 {spec.TEMP_MIN}-{spec.TEMP_MAX} 度之间。"})
                return
            yield from call("ac_set_temperature", {"zone": op["zone"], "value": op["value"]})
            done_msgs.append(f"温度已调到 {op['value']} 度")
        elif o == "TEMP_REL":
            zones = spec.ZONES if op["zone"] == "all" else (op["zone"],)
            fields = [f"ac.temp.{z}" for z in zones]
            res = yield from call("get_vehicle_status", {"fields": fields})
            if failed(res):
                yield from give_up()
                return
            cur = res["data"]
            targets = {z: cur[f"ac.temp.{z}"] + op["delta"] for z in zones}
            if not all(spec.TEMP_MIN <= v <= spec.TEMP_MAX for v in targets.values()):
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED,
                                           "message": "调整后会超出可设置的温度范围。"})
                return
            if len(set(targets.values())) == 1 and op["zone"] == "all":
                yield from call("ac_set_temperature", {"zone": "all", "value": targets["driver"]})
            else:
                for z, v in targets.items():
                    yield from call("ac_set_temperature", {"zone": z, "value": v})
            done_msgs.append("温度已调整")
        elif o == "VOL_ABS":
            if not spec.VOLUME_MIN <= op["value"] <= spec.VOLUME_MAX:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED,
                                           "message": f"音量只能设在 {spec.VOLUME_MIN}-{spec.VOLUME_MAX} 之间。"})
                return
            yield from call("volume_set", {"value": op["value"]}); done_msgs.append("音量已调整")
        elif o == "VOL_REL":
            res = yield from call("get_vehicle_status", {"fields": ["media.volume"]})
            if failed(res):
                yield from give_up()
                return
            v = res["data"]["media.volume"] + op["delta"]
            if not spec.VOLUME_MIN <= v <= spec.VOLUME_MAX:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED,
                                           "message": "调整后会超出音量范围。"})
                return
            yield from call("volume_set", {"value": v}); done_msgs.append("音量已调整")
        elif o in ("NAV_NEAREST", "NAV_BRAND"):
            q = POI_CATEGORIES[op["category"]][0] if o == "NAV_NEAREST" else op["brand"]
            res = yield from call("search_poi", {"query": q})
            if failed(res):
                yield from give_up()
                return
            hits = res["data"] or []
            if o == "NAV_BRAND":                      # search also matches by category
                hits = [p for p in hits if op["brand"] in p["name"]] or hits
            opens = [p for p in hits if p["open"]]
            if not opens:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED,
                                           "message": f"附近没有营业中的{q}。"})
                return
            d = min(p["distance_km"] for p in opens)
            best = [p for p in opens if p["distance_km"] == d]
            if len(best) > 1 and len({p["name"] for p in best}) == 1:   # same name, same distance
                yield from call("finish", {"status": spec.FINISH_CLARIFY,
                                           "message": f"找到 {len(best)} 个同样近的{q}，您要去哪一个？"})
                return
            yield from call("nav_start", {"poi_id": best[0]["id"]})
            done_msgs.append(f"已开始导航到{best[0]['name']}")
        elif o in ("PLAY_SONG", "PLAY_ARTIST"):
            q = op.get("title") or op["artist"]
            res = yield from call("media_search", {"query": q})
            if failed(res):
                yield from give_up()
                return
            hits = res["data"] or []
            if o == "PLAY_SONG":
                hits = [m for m in hits if m["title"] == op["title"]] or hits
            else:
                hits = [m for m in hits if m["artist"] == op["artist"]] or hits
            if not hits:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED, "message": "没有找到这首歌。"})
                return
            yield from call("media_play", {"item_id": hits[0]["id"]})
            done_msgs.append(f"正在播放{hits[0]['title']}")
        elif o == "MANUAL_QA":
            res = yield from call("search_manual", {"query": op["query"]})
            if failed(res):
                yield from give_up()
                return
            hits = res["data"] or []
            if not hits:
                yield from call("finish", {"status": spec.FINISH_UNSUPPORTED, "message": "说明书里没有找到相关内容。"})
                return
            h = hits[0]
            yield from call("finish", {"status": spec.FINISH_DONE,
                                       "message": f"{h['text'][:60]}……（来源：说明书 {h['id']}）"})
            return
        else:
            raise ValueError(o)
    yield from call("finish", {"status": spec.FINISH_DONE, "message": "，".join(done_msgs) + "。"})


class ScriptedPolicy:
    """Drives `plan()` from the conversation. Subclasses provide `ops(task)`."""

    def reset(self, task):
        self._gen = plan(self.ops(task))
        self._started = False

    def act(self, messages):
        try:
            if not self._started:
                self._started = True
                name, args = next(self._gen)
            else:
                last = json.loads(messages[-1]["content"])
                name, args = self._gen.send(last)
        except StopIteration:
            name, args = "finish", {"status": spec.FINISH_DONE, "message": "好的。"}
        return {"text": format_tool_call(name, args), "token_ids": None}

    def ops(self, task):
        raise NotImplementedError


class ExpertPolicy(ScriptedPolicy):
    """Oracle NLU (reads the private semantic intent) + fair planner.

    parallel=True (default: COCKPIT_PARALLEL) groups calls that do not depend on each
    other into the same turn; see `parallel_schedule`.
    """

    def __init__(self, parallel=None):
        self.parallel = F.PARALLEL if parallel is None else parallel

    def reset(self, task):
        if not self.parallel:
            return super().reset(task)
        self._turns, self._k = parallel_schedule(task, self.ops(task)), 0

    def act(self, messages):
        if not self.parallel:
            return super().act(messages)
        turn = self._turns[min(self._k, len(self._turns) - 1)]
        self._k += 1
        return {"text": format_turn(turn), "token_ids": None}

    def ops(self, task):
        ops = task["semantic"]
        if ops and ops[0]["op"] == "CHAT":
            return ops
        return [{k: v for k, v in op.items() if k not in ("unsupported", "clarify", "dflt")} for op in ops]


# --------------------------------------------------------------- parallel schedule
_NEEDS = {"nav_start": "search_poi", "media_play": "media_search"}
_NEEDS_STATUS = {"ac_set_temperature", "volume_set"}


def _sequential_calls(task, ops):
    """Run the sequential plan against a scratch episode: [(name, args)] and the episode."""
    from cockpit.env.episode import Episode
    ep = Episode(task, enable_manual=task.get("enable_manual", False))
    gen, calls = plan(ops), []
    try:
        name, args = next(gen)
        while True:
            calls.append((name, args))
            res = ep.call(name, args)
            if ep.finished or ep.budget_left <= 0:
                break
            name, args = gen.send(res)
    except StopIteration:
        pass
    return calls, ep


def _levels(calls):
    """Turn index for each call: one after everything it depends on."""
    lv = []
    for i, (name, args) in enumerate(calls):
        deps = []
        for j in range(i - 1, -1, -1):                     # a retry waits for the failed attempt
            if calls[j] == (name, args):
                deps.append(j); break
        if name in _NEEDS:
            deps += [j for j in range(i) if calls[j][0] == _NEEDS[name]][-1:]
        if name in _NEEDS_STATUS:
            deps += [j for j in range(i) if calls[j][0] == "get_vehicle_status"][-1:]
        if name == "finish":
            deps = list(range(i))
        lv.append(1 + max((lv[j] for j in deps), default=-1))
    return lv


def parallel_schedule(task, ops):
    """Group the expert's calls into turns by dependency level, finish alone in the last turn.

    The schedule is replayed on a fresh episode and kept only if it reaches exactly the
    same final state and finish payload as the sequential plan; otherwise (rare order
    effects) the sequential one-call-per-turn schedule is used.
    """
    from cockpit.env.episode import Episode
    calls, seq_ep = _sequential_calls(task, ops)
    lv = _levels(calls)
    turns = []
    for level in sorted(set(lv)):
        group = [c for c, l in zip(calls, lv) if l == level]
        group_nf = [c for c in group if c[0] != "finish"]
        for k in range(0, len(group_nf), F.MAX_CALLS_PER_TURN):
            turns.append(group_nf[k:k + F.MAX_CALLS_PER_TURN])
        turns += [[c] for c in group if c[0] == "finish"]
    ep = Episode(task, enable_manual=task.get("enable_manual", False))
    for turn in turns:
        for name, args in turn:
            ep.call(name, args)
    same = ep.state == seq_ep.state and ep.finish_payload == seq_ep.finish_payload and ep.finished == seq_ep.finished
    return turns if same else [[c] for c in calls]
