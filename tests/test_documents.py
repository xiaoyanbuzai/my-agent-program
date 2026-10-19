"""文档管理测试：跨盘 knowledge 目录删除与列表聚合。"""

from __future__ import annotations

from pathlib import Path

import documents
from config import get_settings


class FakeDB:
    def __init__(self, metadatas: list[dict]) -> None:
        self._metadatas = metadatas

    def get(self, where=None, include=None, limit=None):
        return {"ids": [f"id-{index}" for index in range(len(self._metadatas))],
                "metadatas": self._metadatas}


def test_list_documents_groups_chunks(tmp_path, monkeypatch):
    metadatas = [
        {"source": "a.pdf", "doc_type": "pdf", "modality": "text", "page": 1, "ingested_at": 10},
        {"source": "a.pdf", "doc_type": "pdf", "modality": "text", "page": 2, "ingested_at": 11},
        {"source": "b.png", "doc_type": "png", "modality": "image", "page": 0, "ingested_at": 5},
    ]
    monkeypatch.setattr(documents, "get_user_collection", lambda user_id: FakeDB(metadatas))

    result = documents.list_documents(1)

    assert [doc["source"] for doc in result] == ["a.pdf", "b.png"]
    assert result[0]["chunks"] == 2
    assert (result[0]["page_start"], result[0]["page_end"]) == (1, 2)
    assert result[1]["modality"] == ["image"]


def test_delete_document_when_knowledge_dir_outside_base(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    target = knowledge / "documents" / "1" / "a.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.4")

    monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge))
    get_settings.cache_clear()
    try:
        metadata = {
            "source": "a.pdf",
            "file_path": str(target),
            "page": 1,
            "doc_type": "pdf",
        }
        monkeypatch.setattr(
            documents, "get_user_collection", lambda user_id: FakeDB([metadata])
        )
        monkeypatch.setattr(documents, "delete_vectors", lambda user_id, source: 1)
        monkeypatch.setattr(documents, "BASE_DIR", Path("E:/nonexistent-base"))

        result = documents.delete_document(1, "a.pdf")

        assert result["deleted_chunks"] == 1
        assert not target.exists()
        assert result["deleted_files"] == [str(target)]
    finally:
        get_settings.cache_clear()
