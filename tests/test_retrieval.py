"""混合检索阈值、去重与降级测试（不加载真实模型）。"""

from __future__ import annotations

from langchain_core.documents import Document

import kb_retriever


class FakeEnsemble:
    def __init__(self, documents):
        self._documents = documents

    def invoke(self, query):
        return self._documents


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, pairs):
        return self._scores[: len(pairs)]


def _document(parent_id: str, source: str, page: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "parent_id": parent_id,
            "source": source,
            "doc_type": "pdf",
            "title": "第一章",
            "page": page,
            "modality": "text",
            "parent_content": content,
        },
    )


def test_search_applies_threshold_and_dedupes_parents(monkeypatch):
    documents = [
        _document("p1", "a.pdf", 1, "内容一" * 100),
        _document("p1", "a.pdf", 1, "内容一" * 100),
        _document("p2", "b.pdf", 3, "内容二" * 100),
    ]
    context = kb_retriever.RetrievalContext(
        user_id=1, db=None, ensemble=FakeEnsemble(documents), chunk_total=3
    )
    monkeypatch.setattr(kb_retriever, "_get_context", lambda user_id: context)
    monkeypatch.setattr(
        kb_retriever, "_get_reranker", lambda: FakeReranker([2.0, 1.5, -3.0])
    )

    results = kb_retriever.search_kb("查询", 1, min_score=0.0)

    assert [item["parent_id"] for item in results] == ["p1"]
    assert results[0]["score"] == 2.0


def test_search_returns_empty_when_no_candidates(monkeypatch):
    context = kb_retriever.RetrievalContext(
        user_id=1, db=None, ensemble=None, chunk_total=0
    )
    monkeypatch.setattr(kb_retriever, "_get_context", lambda user_id: context)
    assert kb_retriever.search_kb("查询", 1) == []


def test_search_returns_empty_for_blank_query(monkeypatch):
    assert kb_retriever.search_kb("   ", 1) == []
