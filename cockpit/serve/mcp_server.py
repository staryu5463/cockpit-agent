"""Minimal MCP server (JSON-RPC 2.0 over stdio) exposing the cockpit tools.

Implements the three methods a client needs: initialize, tools/list,
tools/call. Any MCP-capable client (or an agent framework) can drive the
simulated cockpit through it. One process = one simulated vehicle; send
{"method": "cockpit/reset", "params": {"task": {...}}} to load a scenario.

    python -m cockpit.serve.mcp_server < requests.jsonl
"""
from __future__ import annotations

import json
import random
import sys

from cockpit.env.episode import Episode
from cockpit.env.tools import MANUAL_TOOL_SPEC, TOOL_SPECS
from cockpit.tasks.generator import make_task

PROTOCOL_VERSION = "2025-06-18"


class Server:
    def __init__(self):
        self.ep = Episode(make_task(random.Random(0), "L1", "A", task_id="mcp-demo"), enable_manual=True)

    def handle(self, req):
        m, p = req.get("method"), req.get("params") or {}
        if m == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "cockpit-bench", "version": "0.1.0"}}
        if m == "tools/list":
            return {"tools": TOOL_SPECS + [MANUAL_TOOL_SPEC]}
        if m == "tools/call":
            res = self.ep.call(p.get("name"), p.get("arguments") or {})
            return {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}],
                    "isError": not res["ok"]}
        if m == "cockpit/reset":
            self.ep = Episode(p["task"], enable_manual=True)
            return {"ok": True}
        raise KeyError(m)

    def serve(self, fin=sys.stdin, fout=sys.stdout):
        for line in fin:
            if not line.strip():
                continue
            req = json.loads(line)
            if "id" not in req:          # notification, e.g. notifications/initialized
                continue
            try:
                resp = {"jsonrpc": "2.0", "id": req["id"], "result": self.handle(req)}
            except KeyError as e:
                resp = {"jsonrpc": "2.0", "id": req["id"],
                        "error": {"code": -32601, "message": f"method not found: {e}"}}
            fout.write(json.dumps(resp, ensure_ascii=False) + "\n"); fout.flush()


if __name__ == "__main__":
    Server().serve()
