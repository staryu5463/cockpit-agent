"""Token-level format tests: the loss mask must sit exactly on the assistant tokens.

Needs the tiny smoke-test model:  python scripts/make_tiny_model.py --tokenizer <gpt2 dir>
"""
import json
import os
import unittest

from cockpit.agent.format import ChatFormat, format_tool_call, parse_tool_call

TINY = "runs/tiny-gpt2"


@unittest.skipUnless(os.path.isdir(TINY), "tiny model not built")
class FormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TINY)
        cls.fmt = ChatFormat(cls.tok)

    def test_mask_covers_exactly_assistant_tokens(self):
        call = format_tool_call("volume_set", {"value": 8})
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "音量调到8"},
                {"role": "assistant", "content": call},
                {"role": "tool", "content": json.dumps({"ok": True})}]
        ids, mask, spans = self.fmt.encode(msgs)
        self.assertEqual(len(spans), 1)
        a, b = spans[0]
        trained = [i for i, m in zip(ids, mask) if m]
        self.assertEqual(trained, ids[a:b])
        self.assertEqual(self.tok.decode(ids[a:b - 1]), call)          # last token is <|im_end|>
        self.assertEqual(ids[b - 1], self.fmt.im_end_id)

    def test_generated_ids_are_kept_verbatim(self):
        """Prompt for turn k+1 = prompt for turn k + the sampled ids: no re-tokenisation."""
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        p1, _, _ = self.fmt.encode(msgs, add_generation_prompt=True)
        sampled = self.tok.encode("<tool_call>{\"name\": \"finish\"", add_special_tokens=False) + [self.fmt.im_end_id]
        msgs.append({"role": "assistant", "content": "ignored", "token_ids": sampled})
        full, mask, spans = self.fmt.encode(msgs)
        self.assertEqual(full[:len(p1)], p1)
        a, b = spans[0]
        self.assertEqual(full[a:b], sampled)

    def test_text_and_token_rendering_agree(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "打开空调"},
                {"role": "assistant", "content": format_tool_call("ac_power", {"on": True})},
                {"role": "tool", "content": "{}"}]
        ids, _, _ = self.fmt.encode(msgs, add_generation_prompt=True)
        self.assertEqual(self.tok.decode(ids), self.fmt.render_text(msgs, add_generation_prompt=True))


class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_tool_call('<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'),
                         ("a", {"x": 1}, None))
        self.assertIsNotNone(parse_tool_call("我来帮你打开空调")[2])
        self.assertIsNotNone(parse_tool_call('<tool_call>{"name": 1}</tool_call>')[2])
        # unterminated tag (generation cut by the stop token) still parses
        self.assertEqual(parse_tool_call('<tool_call>{"name": "a", "arguments": {}}')[0], "a")


class McpTest(unittest.TestCase):
    def test_list_and_call(self):
        import io
        from cockpit.serve.mcp_server import Server
        reqs = "\n".join(json.dumps(r) for r in [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "volume_set", "arguments": {"value": 99}}},
        ])
        out = io.StringIO()
        Server().serve(io.StringIO(reqs), out)
        lines = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertIn("search_manual", [t["name"] for t in lines[1]["result"]["tools"]])
        self.assertTrue(lines[2]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
