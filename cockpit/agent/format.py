"""ChatML rendering at the token level, plus tool-call parsing.

Why render by hand instead of tokenizer.apply_chat_template?
  RL trains on the exact tokens the policy sampled. If we re-render the whole
  conversation as a string and tokenize it again, tokens can merge differently
  at turn boundaries and the loss mask drifts off the generated tokens
  (the failure mode in TRL issue #7503). Here every assistant turn keeps its
  sampled token ids, and prompts are built by concatenating token ids.

Layout (Qwen / ChatML):
  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n{instruction}<|im_end|>\n
  <|im_start|>assistant\n{prefix}{CALL}<|im_end|>\n        <- only {CALL}<|im_end|> is trained
  <|im_start|>user\n<tool_response>\n{json}\n</tool_response><|im_end|>\n
  ...

Two call styles (env COCKPIT_CALL_STYLE, default json):
  json  <tool_call>{"name": "volume_set", "arguments": {"value": 8}}</tool_call>   ~25 tokens / call
  fn    <tool_call>volume_set(value=8)</tool_call>                                ~14 tokens / call
With COCKPIT_PARALLEL=1 one turn may hold several calls that do not depend on each
other (all results come back in one tool message); finish must stay alone.
"""
from __future__ import annotations

import json
import os
import re

CALL_STYLE = os.environ.get("COCKPIT_CALL_STYLE", "json")
PARALLEL = os.environ.get("COCKPIT_PARALLEL", "0") == "1"
MAX_CALLS_PER_TURN = 4

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

SYSTEM_TEMPLATE = """你是车载智能座舱的语音助手，通过调用工具帮车主完成操作。

# 规则
1. {call_rule}
2. 相对调整（如“再低两度”“声音小一点”）要先用 get_vehicle_status 读取当前值。
3. 地点和歌曲必须先搜索，只能使用搜索结果里的 id；“最近”指营业中且距离最小的地点。
4. 工具返回 retryable=true 时可以重试；参数超出范围时不要擅自改成别的值。
5. 信息不足（如两个同名同距离的地点）时用 finish(status="clarify") 询问；不是车控指令时用 finish(status="chat")。
6. 不要修改用户没有要求的任何设置。最后必须调用 finish。
7. 默认值：没说位置的温度指全车（zone=all）；“一点”指温度 {temp_step} 度、音量 {vol_step} 格。

# 工具
<tools>
{tools}
</tools>"""


def call_rule(style=None, parallel=None) -> str:
    style = style or CALL_STYLE
    parallel = PARALLEL if parallel is None else parallel
    if style == "fn":
        fmt = '格式：<tool_call>工具名(参数名=值, ...)</tool_call>，值用 JSON 写法，如 volume_set(value=8)、search_poi(query="加油站")'
    else:
        fmt = '格式：<tool_call>{"name": 工具名, "arguments": {参数}}</tool_call>'
    if parallel:
        return (f"每个调用{fmt}。互不依赖的调用可以放在同一轮（最多 {MAX_CALLS_PER_TURN} 个，结果按顺序一起返回）；"
                "依赖上一步结果的调用要等结果回来；finish 必须单独一轮。")
    return f"每一轮只调用一个工具，{fmt}"


def build_system_prompt(tool_specs) -> str:
    from cockpit import spec
    tools = "\n".join(json.dumps({"name": t["name"], "description": t["description"],
                                  "parameters": t["inputSchema"]}, ensure_ascii=False)
                      for t in tool_specs)
    return SYSTEM_TEMPLATE.format(tools=tools, temp_step=spec.DEFAULT_TEMP_STEP,
                                  vol_step=spec.DEFAULT_VOLUME_STEP, call_rule=call_rule())


def _val(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def format_tool_call(name: str, args: dict, style=None) -> str:
    if (style or CALL_STYLE) == "fn":
        return "<tool_call>" + name + "(" + ",".join(f"{k}={_val(v)}" for k, v in args.items()) + ")</tool_call>"
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


def format_turn(calls, style=None) -> str:
    """Several (name, args) calls in one assistant turn."""
    return "".join(format_tool_call(n, a, style) for n, a in calls)


_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FN_RE = re.compile(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.S)
_KEY_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*")
_PY_LIT_RE = re.compile(r"True|False|None")
_PY_LIT = {"True": True, "False": False, "None": None}
_DEC = json.JSONDecoder()


def _parse_fn_args(s: str) -> dict:
    """'zone="driver",value=22' -> dict. Values are JSON literals, so strings may hold , ( ) =."""
    args, i, n = {}, 0, len(s)
    while True:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            return args
        m = _KEY_RE.match(s, i)
        if not m:
            raise ValueError(f"expected name=value at {s[i:i + 15]!r}")
        key, i = m.group(1), m.end()
        try:
            val, i = _DEC.raw_decode(s, i)
        except json.JSONDecodeError:
            lit = _PY_LIT_RE.match(s, i)
            if not lit:
                raise ValueError(f"bad value for {key}")
            val, i = _PY_LIT[lit.group(0)], lit.end()
        args[key] = val
        while i < n and s[i].isspace():
            i += 1
        if i < n:
            if s[i] != ",":
                raise ValueError(f"expected ',' after {key}")
            i += 1


def _parse_block(body: str):
    body = body.strip()
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            return None, None, f"invalid JSON: {e.msg}"
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return None, None, "tool call needs a string 'name'"
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return None, None, "'arguments' must be an object"
        return obj["name"], args, None
    m = _FN_RE.match(body)
    if not m:
        return None, None, "tool call must look like name(arg=value, ...)"
    try:
        return m.group(1), _parse_fn_args(m.group(2)), None
    except ValueError as e:
        return None, None, str(e)


def parse_tool_calls(text: str):
    """Every call in one turn, as a list of (name, args, error). Both styles are accepted."""
    return [_parse_block(b) for b in _BLOCK_RE.findall(text) if b.strip()]


def parse_tool_call(text: str):
    """First call only: (name, args, None) or (None, None, error_message)."""
    calls = parse_tool_calls(text)
    if not calls:
        return None, None, "no <tool_call> block"
    return calls[0]


class ChatFormat:
    """Token-level renderer. `assistant_prefix` is e.g. '<think>\\n\\n</think>\\n\\n' for Qwen3."""

    def __init__(self, tokenizer, assistant_prefix: str = ""):
        self.tok = tokenizer
        self.assistant_prefix = assistant_prefix
        self.im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
        assert self.im_end_id is not None and self.im_end_id != tokenizer.unk_token_id, \
            "tokenizer must know <|im_end|> as a single token"

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def _block(self, role, content):
        return self._enc(f"{IM_START}{role}\n{content}{IM_END}\n")

    def encode(self, messages, add_generation_prompt=False):
        """messages: [{role, content, token_ids?}] -> (input_ids, loss_mask, turn_spans).

        Assistant turns contribute their stored `token_ids` (sampled tokens,
        ending with <|im_end|>) or, for SFT targets, the encoded content.
        loss_mask is 1 exactly on those assistant tokens.
        """
        ids, mask, spans = [], [], []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                head = self._enc(f"{IM_START}assistant\n{self.assistant_prefix}")
                body = list(m.get("token_ids") or (self._enc(m["content"]) + [self.im_end_id]))
                if body[-1] != self.im_end_id:      # truncated generation: close the turn
                    body = body + [self.im_end_id]
                tail = self._enc("\n")
                ids += head; mask += [0] * len(head)
                spans.append((len(ids), len(ids) + len(body)))
                ids += body; mask += [1] * len(body)
                ids += tail; mask += [0] * len(tail)
            elif role == "tool":
                blk = self._block("user", f"<tool_response>\n{m['content']}\n</tool_response>")
                ids += blk; mask += [0] * len(blk)
            else:
                blk = self._block(role, m["content"])
                ids += blk; mask += [0] * len(blk)
        if add_generation_prompt:
            head = self._enc(f"{IM_START}assistant\n{self.assistant_prefix}")
            ids += head; mask += [0] * len(head)
        return ids, mask, spans

    def render_text(self, messages, add_generation_prompt=False):
        """String form, for HTTP backends (vLLM completions) and for reading logs."""
        out = []
        for m in messages:
            if m["role"] == "assistant":
                out.append(f"{IM_START}assistant\n{self.assistant_prefix}{m['content']}{IM_END}\n")
            elif m["role"] == "tool":
                out.append(f"{IM_START}user\n<tool_response>\n{m['content']}\n</tool_response>{IM_END}\n")
            else:
                out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
        if add_generation_prompt:
            out.append(f"{IM_START}assistant\n{self.assistant_prefix}")
        return "".join(out)
