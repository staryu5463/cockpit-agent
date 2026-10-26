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
"""
from __future__ import annotations

import json
import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

SYSTEM_TEMPLATE = """你是车载智能座舱的语音助手，通过调用工具帮车主完成操作。

# 规则
1. 每一轮只调用一个工具，格式：<tool_call>{{"name": 工具名, "arguments": {{参数}}}}</tool_call>
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


def build_system_prompt(tool_specs) -> str:
    from cockpit import spec
    tools = "\n".join(json.dumps({"name": t["name"], "description": t["description"],
                                  "parameters": t["inputSchema"]}, ensure_ascii=False)
                      for t in tool_specs)
    return SYSTEM_TEMPLATE.format(tools=tools, temp_step=spec.DEFAULT_TEMP_STEP,
                                  vol_step=spec.DEFAULT_VOLUME_STEP)


def format_tool_call(name: str, args: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


def parse_tool_call(text: str):
    """Return (name, args, None) or (None, None, error_message)."""
    m = _CALL_RE.search(text)
    if not m:
        return None, None, "no <tool_call> block"
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return None, None, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None, None, "tool call needs a string 'name'"
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return None, None, "'arguments' must be an object"
    return obj["name"], args, None


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
