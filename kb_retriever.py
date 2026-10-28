"""租户级混合检索：向量 + BM25(jieba)+ CrossEncoder 精排，带分数阈值与引用元数据。

与旧实现的区别：
- BM25 保留原始 metadata(旧实现丢失 source/page,导致引用不可用);
- 精排分数显式保留并支持阈值，检索不到时返回空列表，让上层拒答而不是硬答；
- 按 parent_id 去重，返回父块内容和页码。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from config import configure_hf_environment, get_settings
import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

from chunking import truncate_text
from vector_store import count_user_chunks, get_user_collection

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_context_cache: dict[int, "RetrievalContext"] = {}
_reranker: HuggingFaceCrossEncoder | None = None


@dataclass
class RetrievalContext:
    user_id: int
    db: Any
    ensemble: Any | None
    chunk_total: int


def _tokenize(text: str) -> list[str]:
    return [token for token in jieba.cut(text) if token.strip()]


def _get_reranker() -> HuggingFaceCrossEncoder:
    global _reranker
    with _LOCK:
        if _reranker is None:
            settings = get_settings()
            configure_hf_environment()
            _reranker = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
            logger.info("精排模型：%s", settings.reranker_model)
        return _reranker


def _build_context(user_id: int) -> RetrievalContext:
    settings = get_settings()
    db = get_user_collection(user_id)

    data = db.get(
        limit=settings.bm25_max_docs,
        include=["documents", "metadatas"],
    )
    raw_documents = list(data.get("documents") or [])
    raw_metadatas = list(data.get("metadatas") or [])
    pairs = [
        (text, metadata or {})
        for text, metadata in zip(raw_documents, raw_metadatas)
        if text
    ]
    texts = [text for text, _ in pairs]
    metadatas = [metadata for _, metadata in pairs]

    if not texts:
        return RetrievalContext(user_id=user_id, db=db, ensemble=None, chunk_total=0)

    if len(texts) >= settings.bm25_max_docs:
        logger.warning(
            "用户 %s 的块数达到 BM25 上限 %s,关键词召回只覆盖部分数据;"
            "建议调大 BM25_MAX_DOCS 或引入增量索引。",
            user_id,
            settings.bm25_max_docs,
        )

    vector_retriever = db.as_retriever(
        search_kwargs={"k": settings.retrieve_candidates}
    )
    bm25_retriever = BM25Retriever.from_texts(
        texts,
        metadatas=metadatas,
        preprocess_func=_tokenize,
    )
    bm25_retriever.k = settings.retrieve_candidates

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[
            settings.hybrid_weight_vector,
            1.0 - settings.hybrid_weight_vector,
        ],
    )
    return RetrievalContext(
        user_id=user_id,
        db=db,
        ensemble=ensemble,
        chunk_total=len(texts),
    )


def _get_context(user_id: int) -> RetrievalContext:
    with _LOCK:
        if user_id not in _context_cache:
            _context_cache[user_id] = _build_context(user_id)
        return _context_cache[user_id]


def search_kb(
    query: str,
    user_id: int,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """混合检索，返回按精排分数降序的父块列表。

    返回项:source / doc_type / title / page / modality / parent_id / score / content。
    无相关结果时返回 ``[]``（调用方应据此拒答）。
    """
    settings = get_settings()
    query = (query or "").strip()
    if not query:
        return []

    context = _get_context(user_id)
    if context.ensemble is None:
        return []

    try:
        candidates = context.ensemble.invoke(query)
    except Exception:
        logger.exception("混合检索失败(user_id=%s)", user_id)
        return []

    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, document.page_content) for document in candidates]
    try:
        scores = reranker.score(pairs)
    except Exception:
        logger.exception("精排失败,降级为融合排序(user_id=%s)", user_id)
        scores = [0.0] * len(candidates)

    threshold = settings.rerank_min_score if min_score is None else min_score
    limit = top_k or settings.retrieve_top_k

    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    results: list[dict] = []
    seen_parents: set[str] = set()

    for document, score in ranked:
        score_value = float(score)
        if score_value < threshold:
            continue

        metadata = document.metadata or {}
        parent_id = str(metadata.get("parent_id") or "")
        if not parent_id:
            parent_id = f"legacy-{hash(document.page_content) & 0xFFFFFFFF}"
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        parent_content = metadata.get("parent_content") or document.page_content
        results.append(
            {
                "source": metadata.get("source", "未知来源"),
                "doc_type": metadata.get("doc_type", ""),
                "title": metadata.get("title", ""),
                "page": int(metadata.get("page") or 0),
                "page_end": int(metadata.get("page_end") or metadata.get("page") or 0),
                "modality": metadata.get("modality", "text"),
                "parent_id": parent_id,
                "score": score_value,
                "content": truncate_text(parent_content, settings.max_parent_chars),
            }
        )
        if len(results) >= limit:
            break

    return results


def count_documents(user_id: int) -> int:
    """租户知识库块数量。"""
    return count_user_chunks(user_id)


def invalidate_cache(user_id: int | None = None) -> None:
    """文档变更后重建 BM25 上下文。"""
    with _LOCK:
        if user_id is None:
            _context_cache.clear()
        else:
            _context_cache.pop(user_id, None)


def reset_state() -> None:
    """测试用：清空缓存与精排模型。"""
    global _reranker
    with _LOCK:
        _context_cache.clear()
        _reranker = None
