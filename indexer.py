"""统一索引构建：解析 → 父块/子块 → 写入租户隔离的向量库。

关键设计：
- 父子块：子块 ≤ CHILD_CHUNK_SIZE参与向量/BM25 检索，父块截断后存入 metadata,
  命中后按 parent_id 去重，避免把整章原文全部塞进上下文；
- 溯源：每个块带 source / page / title / modality;
- 一致替换：先写新块、成功后再删旧块，写失败时尽量回滚新块。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking import is_valid_chunk, split_by_chars, split_by_toc, truncate_text
from config import BASE_DIR, get_settings
from multimodal import MultimodalError, describe_image_bytes
from uploads import sha256_file
from vector_store import get_user_collection

logger = logging.getLogger(__name__)

_INGEST_LOCKS: dict[int, threading.Lock] = {}
_INGEST_LOCKS_GUARD = threading.Lock()
_BATCH_SIZE = 500


def _lock_for(user_id: int) -> threading.Lock:
    with _INGEST_LOCKS_GUARD:
        return _INGEST_LOCKS.setdefault(user_id, threading.Lock())


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def _child_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )


def _load_blocks(
    file_path: Path, doc_type: str, display_name: str
) -> tuple[list[dict], bool]:
    """返回 (blocks, is_image)。"""
    settings = get_settings()
    if doc_type in settings.allowed_image_exts:
        data = file_path.read_bytes()
        text = describe_image_bytes(data)
        return (
            [
                {
                    "title": f"图片：{display_name}",
                    "content": text,
                    "page": 0,
                    "modality": "image",
                }
            ],
            True,
        )

    from kb_loader import load_document

    blocks = load_document(str(file_path))
    for block in blocks:
        block.setdefault("modality", "text")
        block.setdefault("page", 0)
    return blocks, False


def _add_in_batches(db, texts, metadatas, ids) -> None:
    for start in range(0, len(texts), _BATCH_SIZE):
        db.add_texts(
            texts=texts[start : start + _BATCH_SIZE],
            metadatas=metadatas[start : start + _BATCH_SIZE],
            ids=ids[start : start + _BATCH_SIZE],
        )


def build_index(
    file_path: str,
    doc_name: str,
    doc_type: str,
    user_id: int,
) -> tuple[bool, str]:
    """把文档建成父子分块索引。

    Args:
        file_path: 待入库文件路径(pdf/docx/md/txt/png/jpg/jpeg/webp)
        doc_name: 文档显示名（通常取原始文件名）
        doc_type: 文档类型标签（空则按扩展名推断）
        user_id: 租户 ID,用于物理隔离

    Returns:
        (是否成功, 面向用户的提示)
    """
    settings = get_settings()
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        return False, "用户信息无效"

    path = Path(file_path)
    if not path.is_file():
        return False, f"文件不存在：{path.name}"

    ext = (doc_type or path.suffix.lstrip(".")).lower().lstrip(".")
    if ext not in settings.allowed_upload_exts:
        return False, f"不支持的文件类型：{ext or '（无扩展名）'}"

    display_name = (doc_name or path.name).strip() or path.name
    source = display_name

    try:
        blocks, is_image = _load_blocks(path, ext, display_name)
    except MultimodalError as exc:
        return False, str(exc)
    except Exception:
        logger.exception("文档解析失败：%s", path.name)
        return False, "文档解析失败，请检查文件是否损坏或格式是否受支持"

    if not blocks:
        return False, "文档为空或无法解析"

    file_hash = sha256_file(path)
    child_splitter = _child_splitter()
    ingest_id = uuid.uuid4().hex[:10]
    ingested_at = int(time.time())

    child_texts: list[str] = []
    child_metadatas: list[dict] = []
    child_ids: list[str] = []
    parent_count = 0

    for block_index, block in enumerate(blocks):
        parent_content = truncate_text(block["content"], settings.max_parent_chars)
        if not parent_content:
            continue
        parent_count += 1
        title = (block.get("title") or "正文").strip()
        page = int(block.get("page") or 0)
        modality = block.get("modality") or "text"
        parent_id = hashlib.sha1(
            f"{user_id}:{source}:{block_index}:{page}:{title}".encode("utf-8")
        ).hexdigest()[:16]

        children = [
            child.strip()
            for child in child_splitter.split_text(block["content"])
            if child.strip()
        ]
        if not children:
            children = [block["content"].strip()]

        accepted = 0
        for child_index, child in enumerate(children):
            # 图片解析结果可能很短（例如图上只有一行字），不做长度门槛
            if not is_image and not is_valid_chunk(child):
                continue

            child_texts.append(child)
            child_metadatas.append(
                {
                    "user_id": int(user_id),
                    "source": source,
                    "doc_type": ext,
                    "title": title,
                    "page": page,
                    "page_end": int(block.get("page_end") or page),
                    "modality": modality,
                    "parent_id": parent_id,
                    "child_index": f"{block_index}_{child_index}",
                    "parent_content": parent_content,
                    "file_path": _relative_path(path),
                    "file_hash": file_hash,
                    "ingested_at": ingested_at,
                }
            )
            child_ids.append(
                f"{user_id}-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:10]}"
                f"-{ingest_id}-{block_index}-{child_index}"
            )
            accepted += 1

        if accepted == 0:
            # 极短但非空的块也要保留，否则整篇可能被丢光
            child_texts.append(block["content"].strip())
            child_metadatas.append(
                {
                    "user_id": int(user_id),
                    "source": source,
                    "doc_type": ext,
                    "title": title,
                    "page": page,
                    "page_end": int(block.get("page_end") or page),
                    "modality": modality,
                    "parent_id": parent_id,
                    "child_index": f"{block_index}_0",
                    "parent_content": parent_content,
                    "file_path": _relative_path(path),
                    "file_hash": file_hash,
                    "ingested_at": ingested_at,
                }
            )
            child_ids.append(
                f"{user_id}-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:10]}"
                f"-{ingest_id}-{block_index}-0"
            )

    if not child_texts:
        return False, "没有提取到有效文本块"

    db = get_user_collection(user_id)

    with _lock_for(user_id):
        try:
            existing = db.get(where={"source": source}, include=["metadatas"])
            old_ids = list(existing.get("ids") or [])
        except Exception:
            logger.exception("查询旧块失败(user_id=%s, source=%s)", user_id, source)
            old_ids = []

        try:
            _add_in_batches(db, child_texts, child_metadatas, child_ids)
        except Exception:
            logger.exception("写入向量库失败(user_id=%s, source=%s)", user_id, source)
            # 尽量回滚本次已写入的块，避免与旧版本重复
            try:
                db.delete(ids=child_ids)
            except Exception:
                logger.exception("回滚新块失败(user_id=%s)", user_id)
            return False, "写入向量库失败，请稍后重试或联系管理员"

        if old_ids:
            try:
                db.delete(ids=old_ids)
            except Exception:
                logger.exception("删除旧块失败(user_id=%s, source=%s)", user_id, source)

    return (
        True,
        f"《{source}》已入库：{parent_count} 个父块，{len(child_texts)} 个子块({ext})",
    )


def delete_document(user_id: int, source: str) -> int:
    """删除某文档的全部块，返回删除数量。"""
    db = get_user_collection(user_id)
    with _lock_for(user_id):
        existing = db.get(where={"source": source}, include=["metadatas"])
        ids = list(existing.get("ids") or [])
        if ids:
            db.delete(ids=ids)
    return len(ids)


# 兼容旧代码的导出（新代码应直接从 chunking 导入）
__all__ = ["build_index", "delete_document", "split_by_chars", "split_by_toc"]
