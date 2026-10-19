"""ORM 模型：用户、文档、长期记忆。"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    doc_name = Column(String(255), nullable=False)
    doc_type = Column(String(16))              # pdf / docx / md / txt
    file_path = Column(String(512))            # 对象存储 key，本地是相对路径
    file_hash = Column(String(64), index=True) # MD5，去重用
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")

    # 同一用户下，同一文件名不能重复
    __table_args__ = (
        Index("ix_user_docname", "user_id", "doc_name", unique=True),
    )


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key = Column(String(128), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("ix_user_memkey", "user_id", "key", unique=True),
    )