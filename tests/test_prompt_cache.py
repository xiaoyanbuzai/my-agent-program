from __future__ import annotations

import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import web_app


def test_prompt_messages_keep_only_recent_compact_history_and_current_tool_cycle():
    messages = []
    for turn in range(4):
        messages.extend(
            [
                HumanMessage(content=f"旧问题 {turn} " + "问" * 500),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "kb_search_tool",
                            "args": {"query": "old"},
                            "id": f"call-{turn}",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="旧检索结果 " + "资料" * 500, tool_call_id=f"call-{turn}"),
                AIMessage(content=f"旧回答 {turn} " + "答" * 500),
            ]
        )
    messages.extend(
        [
            HumanMessage(content="当前问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "kb_search_tool",
                        "args": {"query": "current"},
                        "id": "current-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="当前检索结果", tool_call_id="current-call"),
        ]
    )

    prepared = web_app._prompt_messages(
        messages, "当前记忆", paper_mode=False
    )

    assert len(prepared) == 7
    assert [message.type for message in prepared] == [
        "human",
        "ai",
        "human",
        "ai",
        "human",
        "ai",
        "tool",
    ]
    assert "旧问题 0" not in str(prepared)
    assert "旧检索结果" not in str(prepared)
    assert len(prepared[0].content) <= 1000
    assert "当前记忆" in prepared[-3].content
    assert prepared[-3].content.endswith("【用户问题】\n当前问题")


def test_prompt_system_is_stable_and_dynamic_context_stays_in_user_message():
    assert web_app._system_prompt() == web_app._system_prompt()
    assert "【该用户的历史对话摘要" not in web_app._system_prompt()

    prepared = web_app._prompt_messages(
        [HumanMessage(content="请写论文 Word")],
        "用户历史",
        paper_mode=True,
    )

    assert "用户历史" in prepared[-1].content
    assert "论文生成模式" in prepared[-1].content
    assert "请写论文 Word" in prepared[-1].content


def test_prompt_cache_options_are_only_set_for_moonshot(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")
    assert web_app._prompt_cache_model_kwargs("1h") == {
        "prompt_cache_options": {"mode": "implicit", "ttl": "1h"}
    }

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    assert web_app._prompt_cache_model_kwargs("1h") == {}


def test_prompt_usage_logs_cache_hits_and_misses(caplog):
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 120,
            "input_token_details": {"cached_tokens": 80},
        },
        response_metadata={},
    )

    with caplog.at_level(logging.INFO, logger=web_app.logger.name):
        web_app._log_prompt_usage(response, 7)

    assert "prompt_cache_hit_tokens=80" in caplog.text
    assert "prompt_cache_miss_tokens=40" in caplog.text
