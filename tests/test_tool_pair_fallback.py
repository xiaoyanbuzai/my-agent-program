"""工具消息不配对时的降级重试测试。

上游偶发 400：`assistant message with tool_calls must be followed by tool
messages`。这类请求不能整轮 503，这里锁定「降级为纯文本 + 证据并入」的兜底。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from web_app import _is_tool_pair_error, _strip_tool_history


def test_detects_tool_pair_error():
    exc = RuntimeError(
        "An assistant message with 'tool_calls' must be followed by tool messages "
        "responding to each 'tool_call_id'. (insufficient tool messages following "
        "tool_calls message)"
    )
    assert _is_tool_pair_error(exc)


def test_other_errors_are_not_tool_pair_errors():
    assert not _is_tool_pair_error(RuntimeError("connection reset"))
    assert not _is_tool_pair_error(RuntimeError("rate limit exceeded"))


def test_strip_tool_history_keeps_text_and_inlines_evidence():
    messages = [
        HumanMessage(content="TP53 与肺癌预后？"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "search_pubmed_tool", "args": {"query": "tp53"}}],
        ),
        ToolMessage(content="OS HR=1.74（95%CI 1.12–2.71）", tool_call_id="c1"),
    ]
    cleaned = _strip_tool_history(messages)
    assert len(cleaned) == 1
    assert isinstance(cleaned[0], HumanMessage)
    assert "TP53 与肺癌预后" in cleaned[0].content
    assert "HR=1.74" in cleaned[0].content
    assert "【已检索到的证据】" in cleaned[0].content


def test_strip_tool_history_survives_empty_conversation():
    cleaned = _strip_tool_history([])
    assert len(cleaned) == 1
    assert isinstance(cleaned[0], HumanMessage)
