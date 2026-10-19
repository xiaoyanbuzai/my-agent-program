"""文档切块：按 PDF 书签 / 页切父块，并保留页码用于引用溯源。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

_WORD_CHAR = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]")


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )


def is_valid_chunk(text: str, min_chars: int | None = None) -> bool:
    """过滤过短、乱码、纯符号的碎片。"""
    settings = get_settings()
    cleaned = (text or "").strip()
    if len(cleaned) < (min_chars if min_chars is not None else settings.min_child_chars):
        return False
    word_chars = len(_WORD_CHAR.findall(cleaned))
    if word_chars < 10:
        return False
    return (word_chars / len(cleaned)) >= 0.4


def truncate_text(text: str, max_chars: int) -> str:
    """按句号边界截断，避免把父块整段塞进上下文。"""
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正数")
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned

    window = cleaned[:max_chars]
    boundary = max(window.rfind(mark) for mark in ("。", "！", "？", "\n"))
    if boundary >= max_chars // 2:
        return window[: boundary + 1].rstrip() + "……"
    return window[: max_chars - 1].rstrip() + "……"


def split_by_chars(doc: Any) -> list[dict]:
    """无书签 PDF：逐页切块，保证每个块都带页码。"""
    settings = get_settings()
    splitter = _splitter(settings.parent_chunk_size, settings.parent_chunk_overlap)
    chunks: list[dict] = []

    for page_index, page in enumerate(doc, start=1):
        text = (page.get_text() or "").strip()
        if not text:
            continue
        for part in splitter.split_text(text):
            if part.strip():
                chunks.append(
                    {
                        "title": f"第 {page_index} 页",
                        "content": part.strip(),
                        "page": page_index,
                        "modality": "text",
                    }
                )
    return chunks


def split_by_toc(doc: Any, toc: Iterable[Any]) -> list[dict]:
    """按一级章节切父块；章节过长时内部再切，并保留起始页码。"""
    settings = get_settings()
    splitter = _splitter(settings.parent_chunk_size, settings.parent_chunk_overlap)

    level1 = [
        (str(title).strip(), int(page))
        for level, title, page in toc
        if int(level) == 1 and str(title).strip()
    ]
    if not level1:
        return split_by_chars(doc)

    chunks: list[dict] = []
    total_pages = doc.page_count

    for index, (title, page_number) in enumerate(level1):
        start_page = max(0, min(page_number - 1, total_pages - 1))
        if index + 1 < len(level1):
            end_page = max(start_page + 1, min(level1[index + 1][1] - 1, total_pages))
        else:
            end_page = total_pages

        text = "\n".join(
            (doc[page].get_text() or "") for page in range(start_page, end_page)
        ).strip()
        if not text:
            continue

        parts = splitter.split_text(text) if len(text) > settings.parent_chunk_size else [text]
        for part_index, part in enumerate(parts, start=1):
            part = part.strip()
            if not part:
                continue
            chunks.append(
                {
                    "title": f"{title} (part {part_index})" if len(parts) > 1 else title,
                    "content": part,
                    "page": start_page + 1,
                    "page_end": end_page,
                    "modality": "text",
                }
            )
    return chunks or split_by_chars(doc)


def split_text_blocks(
    text: str, *, title: str, page: int = 0, modality: str = "text"
) -> list[dict]:
    """把一段纯文本切成父块（docx / txt / markdown 兜底用）。"""
    settings = get_settings()
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    splitter = _splitter(settings.parent_chunk_size, settings.parent_chunk_overlap)
    parts = splitter.split_text(cleaned) if len(cleaned) > settings.parent_chunk_size else [cleaned]
    return [
        {
            "title": f"{title} (part {index})" if len(parts) > 1 else title,
            "content": part.strip(),
            "page": page,
            "modality": modality,
        }
        for index, part in enumerate(parts, start=1)
        if part.strip()
    ]
