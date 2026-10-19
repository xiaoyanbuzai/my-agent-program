"""多租户隔离测试：不同 user_id 必须落到不同目录。"""

from __future__ import annotations

from config import get_settings, user_dir
from vector_store import _chroma_path


def test_chroma_paths_are_per_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    try:
        first = _chroma_path(1)
        second = _chroma_path(2)
        assert first != second
        assert first.parent.name == "chroma_db"
        assert first.name == "1"
    finally:
        get_settings.cache_clear()


def test_user_directories_are_per_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "knowledge"))
    get_settings.cache_clear()
    try:
        first = user_dir("documents", 1)
        second = user_dir("documents", 2)
        assert first != second
        assert first.is_dir() and second.is_dir()
    finally:
        get_settings.cache_clear()
