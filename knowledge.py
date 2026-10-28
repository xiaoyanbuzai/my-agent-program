"""文档智能分析：生成摘要、关键词、主题分类(结构化 JSON)。

失败时返回降级结果，不阻断上传流程；所有异常写入日志而不是静默吞掉。
"""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI

from config import get_settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None
_EMPTY = {"summary": "", "keywords": [], "category": ""}


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            timeout=settings.llm_timeout,
            max_retries=1,
        )
    return _llm


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("模型输出中没有 JSON 对象")
    return text[start : end + 1]


def analyze_document(title: str, content: str, max_len: int = 4000) -> dict:
    """分析文档内容，返回 ``{title, summary, keywords, category}``。"""
    cleaned = (content or "").strip()
    if not cleaned:
        return {"title": title, **_EMPTY}

    snippet = cleaned[:max_len]
    prompt = (
        "你是一个文献分析助手。分析下面文档的要点，只输出一个 JSON 对象，不要输出解释，格式：\n"
        '{"summary": "150 字以内的核心内容概括", '
        '"keywords": ["关键词1", "关键词2", "关键词3"], '
        '"category": "文档所属领域/技术类别"}\n\n'
        f"文档标题：{title}\n文档内容:\n{snippet}"
    )

    try:
        response = _get_llm().invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(_extract_json(text))
    except Exception:
        logger.exception("文档分析失败：%s", title)
        return {"title": title, **_EMPTY}

    keywords = data.get("keywords")
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        keywords = []

    return {
        "title": title,
        "summary": str(data.get("summary", ""))[:500],
        "keywords": [str(item)[:40] for item in keywords[:8]],
        "category": str(data.get("category", ""))[:100],
    }
