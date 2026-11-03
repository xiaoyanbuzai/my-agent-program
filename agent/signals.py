"""问题级文本信号：寒暄、直答标记、论文生成诉求、关键词命中。"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Agent 图
# ---------------------------------------------------------------------------


def _is_paper_request(text: str) -> bool:
    """识别“生成论文 doc / 导出 Word / 把 x.pdf 生成 doc”等意图。"""
    lowered = (text or "").lower()
    return bool(
        re.search(r"(生成|导出|写|做|转成|整理).{0,12}(?:docx?(?!\w)|word|文档)", lowered)
        or re.search(r"(论文|paper).{0,12}(?:docx?(?!\w)|word|文档)", lowered)
    )


_PURE_GREETING_PATTERN = re.compile(
    r"[\s,.!?!?]*(你好|您好|嗨|hello|hi|谢谢|thanks|你是谁|介绍一下你自己)[\s,。!?!?]*"
)


_ROUTE_DIRECT_MARKERS = (
    "直接回答",
    "不要检索",
    "不用检索",
    "别检索",
    "无需检索",
    "不用查",
    "别查",
    "简单说",
    "简短回答",
)


def _token_hit(token: str, lowered: str) -> bool:
    """短英文词按单词边界匹配，避免 "os" 命中 "prognosis" 之类的误判。"""
    if token.isascii() and token.isalnum() and len(token) <= 3:
        return re.search(rf"\b{re.escape(token)}\b", lowered) is not None
    return token in lowered
