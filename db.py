"""数据库连接层：本地 SQLite,上线切 PostgreSQL。"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import get_settings

# 本地开发默认 SQLite；上线在 .env 里设 DATABASE_URL=postgresql://...
DB_URL = os.getenv("DATABASE_URL") or get_settings().database_url

if DB_URL.startswith("sqlite"):
    # SQLite 允许多线程访问（FastAPI 默认多线程）
    engine = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(
        DB_URL,
        pool_size=20,
        max_overflow=80,
        pool_pre_ping=True,
        echo=False,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    """FastAPI 依赖：每个请求一个 session,用完自动关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """启动时调用：建表（表存在则跳过）。"""
    import models  # noqa: F401  确保模型已注册到 Base
    Base.metadata.create_all(bind=engine)