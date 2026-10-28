"""SSE 流式生成器测试（不调用真实模型）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import HTTPException
from langgraph.errors import GraphRecursionError

from web_app import _is_paper_request, _remove_repeated_answer, _sse, _stream_answer


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

    assert [event["type"] for event in events] == [
        "reset",
        "token",
        "status",
        "reset",
        "token",
        "done",
    ]
    assert "".join(event["content"] for event in events if event["type"] == "token") == "你好。"
    assert events[-1]["content"] == "好。"


def test_stream_reports_error():
    events = _decode(_stream_answer(BrokenGraph(), "问题", {}))

    assert events == [{"type": "error", "message": "回答生成失败，请稍后重试"}]


def test_stream_reports_model_service_failure_without_exposing_exception():
    class FailedModelGraph:
        def stream(self, payload, config, stream_mode):
            raise HTTPException(
                status_code=503,
                detail="模型服务暂时不可用，请稍后重试",
            )

    events = _decode(_stream_answer(FailedModelGraph(), "问题", {}))

    assert events == [
        {
            "type": "error",
            "message": "模型服务暂时不可用，请稍后重试；服务端日志已记录详细原因。",
        }
    ]


def test_stream_reports_recursion_limit():
    events = _decode(_stream_answer(RecursiveGraph(), "问题", {}))

    assert events[0]["type"] == "error"
    assert "步骤过多" in events[0]["message"]


def test_repeated_complete_answer_is_removed_from_final_event():
    answer = "# 对比结论\n两篇论文关注点不同。"

    class DuplicateGraph:
        def stream(self, payload, config, stream_mode):
            yield SimpleNamespace(content=f"{answer}\n\n{answer}"), {
                "langgraph_node": "replanner"
            }

    events = _decode(_stream_answer(DuplicateGraph(), "比较两篇论文", {}))

    assert events[-1] == {"type": "done", "content": answer}


def test_repeated_answer_cleanup_does_not_remove_partial_overlap():
    text = "共同结论。\n\n共同结论与差异。"

    assert _remove_repeated_answer(text) == text


def test_repeated_answer_cleanup_preserves_prefix_before_duplicate():
    answer = "# 对比结论\n\n" + "\n".join(
        f"第 {index} 节：论文讨论训练阶段与推理数据的关系。"
        for index in range(8)
    )
    text = f"用户问题\n\n{answer}\n\n{answer}"

    assert _remove_repeated_answer(text) == f"用户问题\n\n{answer}"


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
