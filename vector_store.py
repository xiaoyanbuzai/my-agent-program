"""向量库封装:按租户(user_id)隔离,并校验嵌入模型版本。

隔离策略：每个 user_id 一个独立持久化目录(data/chroma_db/<user_id>),
从物理层面避免跨租户检索:collection 内仍保留 user_id 元数据便于审计。

版本策略:collection 创建时写入 ``embedding_model``。换模型后首次访问会抛出
``EmbeddingModelMismatch``，提示运行 ``scripts/reindex.py --rebuild``，避免"换了模型但没重建索引"
导致的静默错检。
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from config import configure_hf_environment, get_settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kb_documents"

_LOCK = threading.RLock()
_embeddings: Embeddings | None = None
_embedding_model_name: str = ""
_cache: dict[int, Chroma] = {}


class EmbeddingModelMismatch(RuntimeError):
    """索引与当前嵌入模型不一致。"""


class PrefixedEmbeddings(Embeddings):
    """给查询加指令前缀(BGE 中文模型需要)，文档侧不加前缀。"""

    def __init__(self, base: Embeddings, query_prefix: str = "") -> None:
        self._base = base
        self._prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(f"{self._prefix}{text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._base.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._base.aembed_query(f"{self._prefix}{text}")


def _build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    configure_hf_environment()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_embeddings() -> tuple[Embeddings, str]:
    """加载嵌入模型；离线且缓存缺失时回退到备用模型并显著告警。"""
    settings = get_settings()
    configured = settings.embedding_model

    try:
        base = _build_embeddings(configured)
        actual = configured
    except Exception as exc:
        fallback = settings.embedding_fallback_model
        if not fallback or fallback == configured:
            raise
        logger.error(
            "嵌入模型 %s 加载失败(%s):已回退到 %s。"
            "请运行 scripts/download_models.py 下载模型后重建索引。",
            configured,
            exc,
            fallback,
        )
        base = _build_embeddings(fallback)
        actual = fallback

    prefix = settings.embedding_query_prefix if "bge" in actual.lower() else ""
    return PrefixedEmbeddings(base, prefix), actual


def get_embeddings() -> Embeddings:
    global _embeddings, _embedding_model_name
    with _LOCK:
        if _embeddings is None:
            _embeddings, _embedding_model_name = _load_embeddings()
            logger.info("嵌入模型：%s", _embedding_model_name)
        return _embeddings


def get_embedding_model_name() -> str:
    get_embeddings()
    return _embedding_model_name


def _chroma_path(user_id: int) -> Path:
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise ValueError("user_id 必须是正整数")
    return get_settings().chroma_dir / str(user_id)


def _open_chroma(user_id: int) -> Chroma:
    path = _chroma_path(user_id)
    path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    model_name = get_embedding_model_name()

    db = Chroma(
        persist_directory=str(path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    stored: dict[str, Any] = dict((db._collection.metadata or {}))
    stored_model = stored.get("embedding_model")
    if stored_model and stored_model != model_name:
        raise EmbeddingModelMismatch(
            f"用户 {user_id} 的索引由 {stored_model} 构建，当前模型为 {model_name}。"
            "请先运行 `uv run python scripts/reindex.py --rebuild` 重建索引。"
        )
    if not stored_model:
        if db._collection.count() > 0:
            raise EmbeddingModelMismatch(
                f"用户 {user_id} 的索引缺少模型标记（可能是旧版本数据）。"
                "请先运行 `uv run python scripts/reindex.py --rebuild` 重建索引。"
            )
        db._collection.modify(
            metadata={
                "embedding_model": model_name,
                "collection": COLLECTION_NAME,
            }
        )

    return db


def get_user_collection(user_id: int) -> Chroma:
    """获取（并缓存）某租户的向量库。"""
    with _LOCK:
        if user_id not in _cache:
            _cache[user_id] = _open_chroma(user_id)
        return _cache[user_id]


def count_user_chunks(user_id: int) -> int:
    """快速统计租户的块数量（走 Chroma 原生 count,不拉全量数据）。"""
    try:
        return int(get_user_collection(user_id)._collection.count())
    except EmbeddingModelMismatch:
        raise
    except Exception:
        logger.exception("统计块数量失败(user_id=%s)", user_id)
        return 0


def delete_user_collection(user_id: int) -> None:
    """删除租户的整个向量库（不可恢复）。"""
    with _LOCK:
        _cache.pop(user_id, None)
        path = _chroma_path(user_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        logger.warning("已删除用户 %s 的向量库：%s", user_id, path)


def reset_cache() -> None:
    """测试 / 重载用：清空进程内缓存（不删数据）。"""
    global _embeddings, _embedding_model_name
    with _LOCK:
        _cache.clear()
        _embeddings = None
        _embedding_model_name = ""
