"""DeepSeek 工具调用的 reasoning_content 往返测试。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from openai.types.chat import ChatCompletion

from web_app import _ReasoningContentChatOpenAI


def _model(base_url: str) -> _ReasoningContentChatOpenAI:
    return _ReasoningContentChatOpenAI(
        model="deepseek-chat",
        api_key="test-key",
        base_url=base_url,
    )


@pytest.mark.parametrize("as_openai_model", [False, True])
def test_deepseek_response_preserves_reasoning_content(as_openai_model: bool):
    response = {
        "id": "response-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Need search recent LLM papers.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_papers_tool",
                                "arguments": '{"topic":"LLM"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    if as_openai_model:
        response = ChatCompletion.model_validate(response)

    result = _model("https://api.deepseek.com/v1")._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs["reasoning_content"] == (
        "Need search recent LLM papers."
    )


def test_deepseek_request_sends_reasoning_content_with_tool_call():
    model = _model("https://api.deepseek.com/v1")
    messages = [
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "Need search recent LLM papers."},
            tool_calls=[
                {
                    "name": "search_papers_tool",
                    "args": {"topic": "LLM"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="Found two papers.", tool_call_id="call-1"),
    ]

    payload = model._get_request_payload(messages)

    assert payload["messages"][0]["reasoning_content"] == (
        "Need search recent LLM papers."
    )
    assert payload["messages"][1]["role"] == "tool"


def test_other_openai_compatible_endpoints_do_not_receive_reasoning_field():
    model = _model("https://api.openai.com/v1")
    message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "provider-specific"},
        tool_calls=[
            {
                "name": "search_papers_tool",
                "args": {"topic": "LLM"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    payload = model._get_request_payload([message])

    assert "reasoning_content" not in payload["messages"][0]
