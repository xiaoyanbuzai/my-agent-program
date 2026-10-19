"""文档管理：按租户列出 / 删除已入库文档，并清理原始文件。

文档列表直接由向量库 metadata 聚合，避免再维护一份容易失真的注册表。
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import BASE_DIR, get_settings
from indexer import delete_document as delete_vectors
from vector_store import get_user_collection

logger = logging.getLogger(__name__)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def list_documents(user_id: int, limit: int = 100000) -> list[dict]:
    """返回租户已入库文档的聚合信息（按最近入库时间倒序）。"""
    db = get_user_collection(user_id)
    result = db.get(limit=limit, include=["metadatas"])
    metadatas = [meta or {} for meta in (result.get("metadatas") or [])]

    aggregated: dict[str, dict] = {}
    for meta in metadatas:
        source = str(meta.get("source") or "未知来源")
        item = aggregated.setdefault(
            source,
            {
                "source": source,
                "doc_type": meta.get("doc_type", ""),
                "modality": set(),
                "chunks": 0,
                "pages": set(),
                "ingested_at": 0,
                "file_path": "",
            },
        )
        item["chunks"] += 1
        item["modality"].add(str(meta.get("modality") or "text"))
        page = int(meta.get("page") or 0)
        if page > 0:
            item["pages"].add(page)
        item["ingested_at"] = max(item["ingested_at"], int(meta.get("ingested_at") or 0))
        if meta.get("file_path") and not item["file_path"]:
            item["file_path"] = str(meta["file_path"])

    documents = []
    for item in aggregated.values():
        pages = sorted(item["pages"])
        documents.append(
            {
                "source": item["source"],
                "doc_type": item["doc_type"],
                "modality": sorted(item["modality"]),
                "chunks": item["chunks"],
                "page_start": pages[0] if pages else 0,
                "page_end": pages[-1] if pages else 0,
                "ingested_at": item["ingested_at"],
                "file_path": item["file_path"],
            }
        )

    documents.sort(key=lambda doc: (doc["ingested_at"], doc["source"]), reverse=True)
    return documents


def delete_document(user_id: int, source: str) -> dict:
    """删除文档的全部块与原始文件（仅限 knowledge 目录内）。"""
    if not source or not source.strip():
        raise ValueError("文档名不能为空")

    settings = get_settings()
    db = get_user_collection(user_id)
    existing = db.get(where={"source": source}, include=["metadatas"])
    metadatas = list(existing.get("metadatas") or [])
    chunk_count = delete_vectors(user_id, source)

    removed_files: list[str] = []
    for meta in metadatas:
        relative = (meta or {}).get("file_path")
        if not relative:
            continue
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if not _is_within(candidate, settings.knowledge_dir):
            logger.warning("拒绝删除 knowledge 目录之外的文件：%s", candidate)
            continue
        if candidate.is_file():
            try:
                candidate.unlink()
                try:
                    display_path = str(candidate.relative_to(BASE_DIR))
                except ValueError:
                    display_path = str(candidate)
                removed_files.append(display_path)
            except OSError:
                logger.exception("删除文件失败：%s", candidate)

    return {
        "source": source,
        "deleted_chunks": chunk_count,
        "deleted_files": sorted(set(removed_files)),
    }
