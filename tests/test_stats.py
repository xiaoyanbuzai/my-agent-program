"""统计接口测试。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import web_app
from vector_store import EmbeddingModelMismatch


@pytest.mark.parametrize(
    "failing_operation",
    ["count_documents", "list_documents", "get_embedding_model_name"],
)
def test_stats_reports_embedding_model_mismatch(
    monkeypatch, failing_operation: str
):
    mismatch = EmbeddingModelMismatch("索引需要重建")

    def raise_mismatch(*args):
        raise mismatch

    monkeypatch.setattr(web_app, "count_documents", lambda user_id: 12)
    monkeypatch.setattr(web_app, "list_documents", lambda user_id: [{"source": "a.pdf"}])
    monkeypatch.setattr(web_app, "get_embedding_model_name", lambda: "model")
    monkeypatch.setattr(web_app, failing_operation, raise_mismatch)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_app.stats(user_id=1))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == str(mismatch)


def test_stats_returns_existing_success_payload(monkeypatch):
    monkeypatch.setattr(web_app, "count_documents", lambda user_id: 12)
    monkeypatch.setattr(
        web_app,
        "list_documents",
        lambda user_id: [{"source": "a.pdf"}, {"source": "b.pdf"}],
    )
    monkeypatch.setattr(web_app, "get_embedding_model_name", lambda: "model")

    result = asyncio.run(web_app.stats(user_id=1))

    assert result == {
        "total_chunks": 12,
        "documents": 2,
        "embedding_model": "model",
        "retrieval": "hybrid(vector+bm25)+cross-encoder",
    }
