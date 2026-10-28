"""向量库模型版本校验测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vector_store
from vector_store import EmbeddingModelMismatch


def test_model_mismatch_recommends_full_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "_chroma_path", lambda user_id: tmp_path / str(user_id))
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: object())
    monkeypatch.setattr(vector_store, "get_embedding_model_name", lambda: "new-model")

    class FakeChroma:
        def __init__(self, **kwargs):
            self._collection = SimpleNamespace(
                metadata={"embedding_model": "old-model"}
            )

    monkeypatch.setattr(vector_store, "Chroma", FakeChroma)

    with pytest.raises(EmbeddingModelMismatch) as exc_info:
        vector_store._open_chroma(1)

    assert "uv run python scripts/reindex.py --rebuild" in str(exc_info.value)
