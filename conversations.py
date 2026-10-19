"""会话与消息持久化：列表、创建、读取、追加、删除。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import ChatMessage, Conversation


class ConversationNotFound(LookupError):
    """会话不存在或不属于当前用户。"""


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.replace(tzinfo=timezone.utc).isoformat()


def conversation_to_dict(conversation: Conversation, message_count: int = 0) -> dict:
    return {
        "id": conversation.id,
        "thread_id": conversation.thread_id,
        "title": conversation.title,
        "created_at": _iso(conversation.created_at),
        "updated_at": _iso(conversation.updated_at),
        "message_count": message_count,
    }


def message_to_dict(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": _iso(message.created_at),
    }


def create_conversation(db: Session, user_id: int, title: str = "新对话") -> Conversation:
    conversation = Conversation(
        user_id=user_id,
        thread_id=uuid.uuid4().hex,
        title=(title or "新对话").strip()[:120] or "新对话",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db: Session, user_id: int) -> list[dict]:
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .all()
    )
    if not conversations:
        return []

    conversation_ids = [conversation.id for conversation in conversations]
    counts = dict(
        db.query(ChatMessage.conversation_id, func.count(ChatMessage.id))
        .filter(ChatMessage.conversation_id.in_(conversation_ids))
        .group_by(ChatMessage.conversation_id)
        .all()
    )
    return [
        conversation_to_dict(conversation, counts.get(conversation.id, 0))
        for conversation in conversations
    ]


def get_conversation(db: Session, user_id: int, conversation_id: int) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if conversation is None:
        raise ConversationNotFound("会话不存在")
    return conversation


def list_messages(db: Session, conversation: Conversation) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return [message_to_dict(message) for message in messages]


def append_message(
    db: Session, conversation: Conversation, role: str, content: str
) -> ChatMessage:
    text = (content or "").strip()
    if not text:
        raise ValueError("消息内容不能为空")

    if role == "user" and conversation.title in {"", "新对话"}:
        title = " ".join(text.split())[:40]
        if title:
            conversation.title = title

    conversation.updated_at = datetime.utcnow()
    message = ChatMessage(
        conversation_id=conversation.id,
        role=role,
        content=text,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def delete_conversation(db: Session, conversation: Conversation) -> None:
    db.delete(conversation)
    db.commit()


def thread_key(user_id: int, conversation: Conversation) -> str:
    """与 web_app._thread_id 保持一致的 LangGraph 线程键。"""
    return f"u{user_id}-{conversation.thread_id}"
