"""SSE 流式生成器测试（不调用真实模型）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from web_app import _sse, _stream_answer


class FakeGraph:
    def stream(self, payload, config, stream_mode):
        yield SimpleNamespace(content="你"), {"langgraph_node": "chatbot"}
        yield SimpleNamespace(content=""), {"langgraph_node": "chatbot"}
        yield SimpleNamespace(content="不应出现"), {"langgraph_node": "tools"}
        yield SimpleNamespace(content="好。"), {"langgraph_node": "chatbot"}


class BrokenGraph:
    def stream(self, payload, config, stream_mode):
        raise RuntimeError("boom")


def _decode(chunks):
    events = []
    for chunk in chunks:
        assert chunk.startswith("data: ")
        events.append(json.loads(chunk[6:].strip()))
    return events


def test_stream_emits_tokens_and_done():
    events = _decode(_stream_answer(FakeGraph(), "问题", {}))

    assert [event["type"] for event in events] == ["token", "token", "done"]
    assert "".join(event["content"] for event in events if event["type"] == "token") == "你好。"
    assert events[-1]["content"] == "你好。"


def test_stream_reports_error():
    events = _decode(_stream_answer(BrokenGraph(), "问题", {}))

    assert events == [{"type": "error", "message": "回答生成失败，请稍后重试"}]


def test_sse_payload_is_utf8_json():
    payload = _sse({"type": "token", "content": "中文"})
    assert payload.startswith("data: ")
    assert payload.endswith("\n\n")
    assert "\\u" not in payload
