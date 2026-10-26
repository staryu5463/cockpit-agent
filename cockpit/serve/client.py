"""vLLM (OpenAI-compatible) policy over HTTP, standard library only.

Uses the /v1/completions endpoint with our own rendered prompt, so the server
sees exactly the format the model was trained on.

guided=True: the prompt is forced to end with "<tool_call>" and the server is
asked to emit JSON matching the tool-call schema (constrained decoding). This
removes format errors by construction; the experiment is to measure how much
that buys compared with fixing format through RL, and what it costs in latency.
The request field name for constrained decoding changed across vLLM versions
("guided_json" in older releases, "structured_outputs" in newer ones); check
the docs of the version you install and set `guided_field` accordingly.
"""
from __future__ import annotations

import json
import time
import urllib.request

from cockpit.agent.format import IM_END
from cockpit.env.tools import MANUAL_TOOL_SPEC, TOOL_SPECS


def tool_call_schema(tool_specs):
    return {"oneOf": [{"type": "object",
                       "properties": {"name": {"const": t["name"]}, "arguments": t["inputSchema"]},
                       "required": ["name", "arguments"]} for t in tool_specs]}


class _TextFormat:
    """render_text without needing a tokenizer locally."""

    def __init__(self, assistant_prefix=""):
        self.assistant_prefix = assistant_prefix

    def render_text(self, messages, add_generation_prompt=False):
        from cockpit.agent.format import ChatFormat
        return ChatFormat.render_text(self, messages, add_generation_prompt)


class VLLMPolicy:
    def __init__(self, base_url, model, tokenizer_path=None, guided=False, guided_field="guided_json",
                 max_new_tokens=256, temperature=0.0, enable_manual=False, timeout=60):
        from cockpit.train.common import assistant_prefix_for
        self.url = base_url.rstrip("/") + "/completions"
        self.model, self.guided, self.guided_field = model, guided, guided_field
        self.max_new_tokens, self.temperature, self.timeout = max_new_tokens, temperature, timeout
        self.fmt = _TextFormat(assistant_prefix_for(tokenizer_path or model))
        specs = TOOL_SPECS + ([MANUAL_TOOL_SPEC] if enable_manual else [])
        self.schema = tool_call_schema(specs)
        self.last_latency = []

    def reset(self, task):
        self.last_latency = []

    def act(self, messages):
        prompt = self.fmt.render_text(messages, add_generation_prompt=True)
        body = {"model": self.model, "prompt": prompt, "max_tokens": self.max_new_tokens,
                "temperature": self.temperature, "stop": [IM_END]}
        if self.guided:
            body["prompt"] += "<tool_call>"
            body[self.guided_field] = self.schema
            body["stop"] = [IM_END, "</tool_call>"]
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        self.last_latency.append(time.perf_counter() - t0)
        text = out["choices"][0]["text"]
        if self.guided:
            text = "<tool_call>" + text.strip() + "</tool_call>"
        return {"text": text, "token_ids": None, "usage": out.get("usage")}
