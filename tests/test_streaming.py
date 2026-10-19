"""SSE 流式生成器测试（不调用真实模型）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from langgraph.errors import GraphRecursionError

from web_app import _is_paper_request, _sse, _stream_answer


class FakeGraph:
    def stream(self, payload, config, stream_mode):
        yield SimpleNamespace(content="你"), {"langgraph_node": "chatbot"}
        yield SimpleNamespace(content=""), {"langgraph_node": "chatbot"}
        yield SimpleNamespace(content="不应出现"), {"langgraph_node": "tools"}
        yield SimpleNamespace(content="好。"), {"langgraph_node": "chatbot"}


class BrokenGraph:
    def stream(self, payload, config, stream_mode):
        raise RuntimeError("boom")


class RecursiveGraph:
    def stream(self, payload, config, stream_mode):
        raise GraphRecursionError("too many steps")


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


def test_stream_reports_recursion_limit():
    events = _decode(_stream_answer(RecursiveGraph(), "问题", {}))

    assert events[0]["type"] == "error"
    assert "步骤过多" in events[0]["message"]


def test_paper_request_detection():
    assert _is_paper_request("把 1.pdf 生成 1.doc")
    assert _is_paper_request("基于知识库生成一篇论文 Word 文档")
    assert _is_paper_request("export this paper as docx")
    assert not _is_paper_request("这篇论文讲了什么？")


def test_sse_payload_is_utf8_json():
    payload = _sse({"type": "token", "content": "中文"})
    assert payload.startswith("data: ")
    assert payload.endswith("\n\n")
    assert "\\u" not in payload
