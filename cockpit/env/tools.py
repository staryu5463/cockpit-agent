"""Cockpit tools: JSON-schema specs (MCP-style) plus implementations.

Every tool returns the same envelope:
    {"ok": bool, "code": str, "data": any, "retryable": bool}
A failed call never changes the vehicle state.
"""
from __future__ import annotations

from cockpit import spec

# --------------------------------------------------------------------------
# Specs. `inputSchema` follows the MCP tools/list format, so the same list is
# served by the MCP server and rendered into the model's system prompt.
# --------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "get_vehicle_status",
        "description": "读取车辆当前状态。fields 为空时返回全部公开字段。",
        "inputSchema": {
            "type": "object",
            "properties": {"fields": {"type": "array", "items": {"type": "string"},
                                      "description": "如 ac.temp.driver, media.volume"}},
        },
    },
    {
        "name": "ac_power",
        "description": "打开或关闭空调。",
        "inputSchema": {"type": "object", "properties": {"on": {"type": "boolean"}},
                        "required": ["on"]},
    },
    {
        "name": "ac_set_temperature",
        "description": f"设定空调绝对温度（{spec.TEMP_MIN}-{spec.TEMP_MAX} 度）。"
                       "zone 为 driver / passenger / all。",
        "inputSchema": {
            "type": "object",
            "properties": {"zone": {"type": "string", "enum": ["driver", "passenger", "all"]},
                           "value": {"type": "integer"}},
            "required": ["zone", "value"],
        },
    },
    {
        "name": "volume_set",
        "description": f"设定绝对音量（{spec.VOLUME_MIN}-{spec.VOLUME_MAX}）。",
        "inputSchema": {"type": "object", "properties": {"value": {"type": "integer"}},
                        "required": ["value"]},
    },
    {
        "name": "search_poi",
        "description": "按关键词搜索地点，返回候选的 id、名称、距离（公里）和是否营业。",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
    },
    {
        "name": "nav_start",
        "description": "导航到某个地点。poi_id 必须来自本次对话中 search_poi 的返回。",
        "inputSchema": {"type": "object", "properties": {"poi_id": {"type": "string"}},
                        "required": ["poi_id"]},
    },
    {
        "name": "media_search",
        "description": "按歌手或歌名搜索歌曲，返回候选的 id、歌名和歌手。",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
    },
    {
        "name": "media_play",
        "description": "播放某首歌。item_id 必须来自本次对话中 media_search 的返回。",
        "inputSchema": {"type": "object", "properties": {"item_id": {"type": "string"}},
                        "required": ["item_id"]},
    },
    {
        "name": "finish",
        "description": "结束本次请求。status: done=已完成, clarify=需要用户补充信息, "
                       "unsupported=无法执行, chat=不是车控指令。message 是对用户说的话。",
        "inputSchema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": list(spec.FINISH_STATUSES)},
                           "message": {"type": "string"}},
            "required": ["status", "message"],
        },
    },
]

MANUAL_TOOL_SPEC = {
    "name": "search_manual",
    "description": "检索车辆使用说明书，返回最相关的段落及其 id。",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]},
}

CORE_TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def ok(data=None):
    return {"ok": True, "code": "OK", "data": data, "retryable": False}


def err(code, msg="", retryable=False):
    return {"ok": False, "code": code, "data": msg, "retryable": retryable}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------
# Implementations. `ep` is the running Episode (see episode.py).
# --------------------------------------------------------------------------
def t_get_vehicle_status(ep, fields=None):
    from cockpit.env.state import FIELDS
    if fields is None or fields == []:
        fields = list(FIELDS)
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        return err("INVALID_ARGUMENT", "fields must be a list of strings")
    unknown = [f for f in fields if f not in FIELDS]
    if unknown:
        return err("INVALID_ARGUMENT", f"unknown fields: {unknown}")
    return ok({f: ep.state[f] for f in fields})


def t_ac_power(ep, on=None):
    if not isinstance(on, bool):
        return err("INVALID_ARGUMENT", "on must be boolean")
    ep.state["ac.power"] = on
    return ok({"ac.power": on})


def t_ac_set_temperature(ep, zone=None, value=None):
    if zone not in ("driver", "passenger", "all"):
        return err("INVALID_ARGUMENT", "zone must be driver/passenger/all")
    if not _is_int(value):
        return err("INVALID_ARGUMENT", "value must be an integer")
    if not spec.TEMP_MIN <= value <= spec.TEMP_MAX:
        return err("OUT_OF_RANGE", f"temperature must be {spec.TEMP_MIN}-{spec.TEMP_MAX}")
    zones = spec.ZONES if zone == "all" else (zone,)
    for z in zones:
        ep.state[f"ac.temp.{z}"] = value
    return ok({f"ac.temp.{z}": value for z in zones})


def t_volume_set(ep, value=None):
    if not _is_int(value):
        return err("INVALID_ARGUMENT", "value must be an integer")
    if not spec.VOLUME_MIN <= value <= spec.VOLUME_MAX:
        return err("OUT_OF_RANGE", f"volume must be {spec.VOLUME_MIN}-{spec.VOLUME_MAX}")
    ep.state["media.volume"] = value
    return ok({"media.volume": value})


def _poi_matches(poi, q):
    from cockpit.env.catalog import POI_CATEGORIES
    cat_cn = POI_CATEGORIES[poi["category"]][0]
    return q in poi["name"] or q in cat_cn or cat_cn in q or poi["category"] == q


def t_search_poi(ep, query=None):
    if not isinstance(query, str) or not query.strip():
        return err("INVALID_ARGUMENT", "query must be a non-empty string")
    q = query.strip()
    hits = [p for p in ep.catalog["pois"] if _poi_matches(p, q)][:8]
    for p in hits:
        ep.surfaced.add(p["id"])
    return ok([{k: p[k] for k in ("id", "name", "distance_km", "open")} for p in hits])


def t_nav_start(ep, poi_id=None):
    if not isinstance(poi_id, str):
        return err("INVALID_ARGUMENT", "poi_id must be a string")
    if poi_id not in ep.surfaced:
        return err("UNKNOWN_ID", "poi_id was not returned by search_poi in this episode")
    poi = next((p for p in ep.catalog["pois"] if p["id"] == poi_id), None)
    if poi is None:
        return err("NOT_FOUND", "no such poi")
    if not poi["open"]:
        return err("POI_CLOSED", f"{poi['name']} is closed")
    ep.state["nav.destination"] = poi_id
    return ok({"nav.destination": poi_id, "name": poi["name"]})


def t_media_search(ep, query=None):
    if not isinstance(query, str) or not query.strip():
        return err("INVALID_ARGUMENT", "query must be a non-empty string")
    q = query.strip()
    hits = [m for m in ep.catalog["media"] if q in m["artist"] or q in m["title"]
            or m["artist"] in q or m["title"] in q][:8]
    for m in hits:
        ep.surfaced.add(m["id"])
    return ok([{k: m[k] for k in ("id", "title", "artist")} for m in hits])


def t_media_play(ep, item_id=None):
    if not isinstance(item_id, str):
        return err("INVALID_ARGUMENT", "item_id must be a string")
    if item_id not in ep.surfaced:
        return err("UNKNOWN_ID", "item_id was not returned by media_search in this episode")
    if not any(m["id"] == item_id for m in ep.catalog["media"]):
        return err("NOT_FOUND", "no such item")
    ep.state["media.playing"] = item_id
    return ok({"media.playing": item_id})


def t_finish(ep, status=None, message=None):
    if status not in spec.FINISH_STATUSES:
        return err("INVALID_ARGUMENT", f"status must be one of {spec.FINISH_STATUSES}")
    if not isinstance(message, str):
        return err("INVALID_ARGUMENT", "message must be a string")
    ep.finished = True
    ep.finish_payload = {"status": status, "message": message}
    return ok({"status": status})


def t_search_manual(ep, query=None):
    from cockpit.rag.manual import search_manual
    if not isinstance(query, str) or not query.strip():
        return err("INVALID_ARGUMENT", "query must be a non-empty string")
    hits = search_manual(query, k=3)
    for h in hits:
        ep.surfaced.add(h["id"])
    return ok(hits)


IMPLS = {
    "get_vehicle_status": t_get_vehicle_status,
    "ac_power": t_ac_power,
    "ac_set_temperature": t_ac_set_temperature,
    "volume_set": t_volume_set,
    "search_poi": t_search_poi,
    "nav_start": t_nav_start,
    "media_search": t_media_search,
    "media_play": t_media_play,
    "finish": t_finish,
    "search_manual": t_search_manual,
}
