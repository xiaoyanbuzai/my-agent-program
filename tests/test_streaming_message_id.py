"""流式问答的用户消息 ID 传递测试。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import web_app
from web_app import AskRequest


def test_ask_stream_sends_persisted_user_message_id(monkeypatch):
    class FakeGraph:
        def stream(self, payload, config, stream_mode):
            yield SimpleNamespace(content="answer"), {"langgraph_node": "chatbot"}

    class CapturedResponse:
        def __init__(self, content, **kwargs):
            self.content = content

    conversation = SimpleNamespace(id=3, thread_id="thread-3")
    monkeypatch.setattr(
        web_app,
        "get_settings",
        lambda: SimpleNamespace(max_question_chars=100, recursion_limit=20),
    )
    monkeypatch.setattr(web_app, "_resolve_conversation", lambda *args: conversation)
    monkeypatch.setattr(
        web_app, "append_message", lambda *args: SimpleNamespace(id=42)
    )
    monkeypatch.setattr(web_app, "StreamingResponse", CapturedResponse)
    monkeypatch.setattr(web_app, "_persist_assistant_message", lambda *args: None)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(graph=FakeGraph(), checkpointer=None)
        )
    )
    response = asyncio.run(
        web_app.ask_stream(
            AskRequest(question="Find two LLM papers", conversation_id=3),
            request,
            user_id=7,
            db=object(),
        )
    )

    events = [
        json.loads(chunk[6:].strip())
        for chunk in response.content
    ]

    assert events[0] == {"type": "user_msg_id", "id": 42}


def test_edit_rebuilds_history_and_clears_the_scoped_checkpoint(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.updated = None

        def update_state(self, config, state):
            self.updated = (config, state)

        def stream(self, payload, config, stream_mode):
            yield SimpleNamespace(content="revised answer"), {"langgraph_node": "chatbot"}

    class FakeCheckpointer:
        deleted_thread = None

        def delete_thread(self, thread_id):
            self.deleted_thread = thread_id

    class CapturedResponse:
        def __init__(self, content, **kwargs):
            self.content = list(content)

    graph = FakeGraph()
    checkpointer = FakeCheckpointer()
    conversation = SimpleNamespace(id=3, thread_id="thread-3")
    messages = [
        {"id": 40, "role": "user", "content": "Earlier question"},
        {"id": 41, "role": "assistant", "content": "Earlier answer"},
        {"id": 42, "role": "user", "content": "Original question"},
        {"id": 43, "role": "assistant", "content": "Old answer"},
    ]
    deleted = []

    monkeypatch.setattr(
        web_app,
        "get_settings",
        lambda: SimpleNamespace(max_question_chars=100, recursion_limit=20),
    )
    monkeypatch.setattr(web_app, "_resolve_conversation", lambda *args: conversation)
    monkeypatch.setattr(web_app, "list_conversation_messages", lambda *args: messages)
    monkeypatch.setattr(
        web_app, "delete_messages_after", lambda *args: deleted.append(args[-1])
    )
    monkeypatch.setattr(
        web_app, "append_message", lambda *args: SimpleNamespace(id=44)
    )
    monkeypatch.setattr(web_app, "StreamingResponse", CapturedResponse)
    monkeypatch.setattr(web_app, "_persist_assistant_message", lambda *args: None)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(graph=graph, checkpointer=checkpointer)
        )
    )
    response = asyncio.run(
        web_app.ask_stream(
            AskRequest(
                question="Revised question",
                conversation_id=3,
                edit_message_id=42,
            ),
            request,
            user_id=7,
            db=object(),
        )
    )
    events = [json.loads(chunk[6:].strip()) for chunk in response.content]
    config, state = graph.updated

    assert checkpointer.deleted_thread == "u7-thread-3"
    assert deleted == [42]
    assert config["configurable"]["thread_id"] == "u7-thread-3"
    assert [(message.type, message.content) for message in state["messages"]] == [
        ("human", "Earlier question"),
        ("ai", "Earlier answer"),
    ]
    assert events[0] == {"type": "user_msg_id", "id": 44}
