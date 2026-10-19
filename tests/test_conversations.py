"""会话持久化测试：创建、消息、排序、标题、租户隔离与删除。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from conversations import (
    ConversationNotFound,
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
)
from db import Base
from models import ChatMessage, User


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    db.add_all(
        [
            User(id=1, username="alice", password_hash="x"),
            User(id=2, username="bob", password_hash="x"),
        ]
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


def test_conversation_messages_and_title(session):
    conversation = create_conversation(session, 1)
    assert conversation.title == "新对话"

    append_message(session, conversation, "user", "这篇论文的核心方法是什么？")
    append_message(session, conversation, "assistant", "它使用了混合检索。")

    messages = list_messages(session, conversation)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert conversation.title == "这篇论文的核心方法是什么？"

    listed = list_conversations(session, 1)
    assert listed[0]["message_count"] == 2


def test_conversation_is_tenant_scoped(session):
    conversation = create_conversation(session, 1)

    with pytest.raises(ConversationNotFound):
        get_conversation(session, 2, conversation.id)

    assert list_conversations(session, 2) == []


def test_delete_conversation_cascades_messages(session):
    conversation = create_conversation(session, 1)
    append_message(session, conversation, "user", "你好")
    conversation_id = conversation.id

    delete_conversation(session, conversation)

    assert list_conversations(session, 1) == []
    assert session.query(User).filter_by(id=1).one() is not None
    assert session.query(ChatMessage).filter_by(conversation_id=conversation_id).count() == 0
    with pytest.raises(ConversationNotFound):
        get_conversation(session, 1, conversation_id)
