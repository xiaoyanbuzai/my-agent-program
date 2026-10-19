"""早期“每篇论文一个 collection”检索器（已退役，仅保留给旧演示脚本）。

当前运行时请使用 ``kb_retriever.search_kb``。此文件不再参与 Web 应用，
模型改为懒加载，避免导入时就下载/占用显存。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jieba
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings: HuggingFaceEmbeddings | None = None
_reranker: CrossEncoderReranker | None = None
_paper_context_cache: dict[str, dict] = {}


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        _reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
    return _reranker


def _get_paper_context(paper_name: str) -> dict:
    """懒加载并缓存单篇论文的检索上下文。"""
    if paper_name not in _paper_context_cache:
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=_get_embeddings(),
            collection_name=paper_name,
        )
        base_retriever = db.as_retriever(search_kwargs={"k": 20})

        all_docs = db.get(limit=10000)["documents"]
        bm25_retriever = BM25Retriever.from_texts(
            all_docs,
            preprocess_func=lambda text: list(jieba.cut(text)),
        )
        bm25_retriever.k = 20

        ensemble = EnsembleRetriever(
            retrievers=[base_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_get_reranker(),
            base_retriever=ensemble,
        )
        _paper_context_cache[paper_name] = {
            "db": db,
            "bm25": bm25_retriever,
            "compression": compression_retriever,
        }
    return _paper_context_cache[paper_name]


def search_paper(query: str, paper_name: str) -> str:
    """搜索单篇论文（旧接口）。新代码请用 ``kb_retriever.search_kb``。"""
    ctx = _get_paper_context(paper_name)
    docs = ctx["compression"].invoke(query)
    if not docs:
        return "未找到相关内容"
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
