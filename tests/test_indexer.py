"""索引元数据与替换语义测试。"""

from __future__ import annotations

import indexer


class FakeDB:
    def __init__(self) -> None:
        self.added_texts: list[str] = []
        self.added_metadatas: list[dict] = []
        self.added_ids: list[str] = []
        self.deleted: list[str] = []

    def get(self, where=None, include=None):
        return {"ids": ["old-1", "old-2"]}

    def add_texts(self, texts, metadatas, ids):
        self.added_texts.extend(texts)
        self.added_metadatas.extend(metadatas)
        self.added_ids.extend(ids)
        return ids

    def delete(self, ids):
        self.deleted.extend(ids)


def test_build_index_writes_tenant_metadata_and_replaces_old(monkeypatch, tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4")

    blocks = [
        {
            "title": "第一章",
            "content": "这是正文内容。" * 400,
            "page": 2,
            "modality": "text",
        }
    ]
    monkeypatch.setattr(indexer, "_load_blocks", lambda *args: (blocks, False))
    monkeypatch.setattr(indexer, "sha256_file", lambda _: "deadbeef")

    db = FakeDB()
    monkeypatch.setattr(indexer, "get_user_collection", lambda user_id: db)

    ok, message = indexer.build_index(str(path), "paper.pdf", "pdf", 7)

    assert ok, message
    assert db.added_texts
    assert db.deleted == ["old-1", "old-2"]

    metadata = db.added_metadatas[0]
    assert metadata["user_id"] == 7
    assert metadata["source"] == "paper.pdf"
    assert metadata["page"] == 2
    assert metadata["modality"] == "text"
    assert len(metadata["parent_content"]) <= 1502
    assert metadata["file_hash"] == "deadbeef"
    assert all(meta["user_id"] == 7 for meta in db.added_metadatas)


def test_build_index_rejects_unknown_extension(monkeypatch, tmp_path):
    path = tmp_path / "script.exe"
    path.write_bytes(b"MZ")
    ok, message = indexer.build_index(str(path), "script.exe", "exe", 1)
    assert not ok
    assert "不支持" in message


def test_build_index_rejects_bad_user(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("内容", encoding="utf-8")
    ok, message = indexer.build_index(str(path), "a.txt", "txt", 0)
    assert not ok
    assert "用户" in message
