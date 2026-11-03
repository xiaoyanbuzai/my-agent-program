"""回答清洗：去免责声明与学术套话、去工具/函数痕迹、去重复。"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("web_app")


# ---------------------------------------------------------------------------
# 最终答案后处理：确定性剔除免责声明与学术废话
#
# 只靠提示词约束不住模型：同样的免责声明每次都会换个说法回来。这里在「最终
# 交付」这一步做一次确定性清洗，和提示词里的【最终输出硬约束】形成双保险。
# 清洗粒度是整行：带硬数据(数字/PMID/CI)的行一律保留，避免误删结论。
# ---------------------------------------------------------------------------

_DISCLAIMER_PATTERNS = (
    re.compile(
        r"仅供参考|不构成.{0,16}(?:建议|依据|诊断|治疗)|请(?:务必)?(?:咨询|遵)|"
        r"建议(?:咨询|就医)|及时就医|遵医嘱|风险自担|免责声明|"
        r"本文(?:不|仅)|以上内容(?:仅|不)|特此(?:说明|声明)|"
        r"不能替代.{0,12}(?:医生|专业)|如有(?:不适|疑问)"
    ),
    re.compile(r"^(?:注|说明|提示|备注)[:：].{0,60}$"),
)
_FILLER_PATTERNS = (
    re.compile(
        r"^(?:综上所述|总而言之|总的来说|总之|由此可见|值得注意的是|"
        r"需要指出的是|需要强调的是|不难看出|可以看出)"
    ),
    re.compile(
        r"随着.{0,24}(?:发展|进步|普及)|近年来|在当今|众所周知|毋庸置疑|"
        r"具有重要意义|具有重要的(?:理论|临床|参考|指导)意义|"
        r"为进一步.{0,24}(?:提供|奠定)|希望(?:以上|这些|本文).{0,20}(?:帮助|参考)|"
        r"感谢(?:您的)?(?:阅读|关注)"
    ),
    re.compile(
        r"^(?:根据(?:检索|查询|搜索结果|工具返回|资料)|我(?:查到|检索到|搜索了)|"
        r"资料(?:显示|表明)|以下是(?:基于|根据)|下面(?:将|我))"
    ),
)
_HARD_DATA_PATTERN = re.compile(
    r"\d|PMID|DOI|HR|OR|RR|CI|p\s*[=<]", re.I
)
_INLINE_DISCLAIMER_PATTERN = re.compile(
    r"[（(]\s*(?:仅供参考|不构成[^）)]{0,16}|仅作参考)\s*[）)]", re.I
)


def _line_is_disclaimer(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DISCLAIMER_PATTERNS)


def _line_is_filler(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FILLER_PATTERNS)


def _line_carries_hard_data(text: str) -> bool:
    return bool(_HARD_DATA_PATTERN.search(text))


# ---------------------------------------------------------------------------
# 工具/函数痕迹清洗
#
# 回答里出现 ``search_pubmed_tool({"query": "TP53"})``、``fetch_article_details``
# 这类实现细节，对用户是噪音，也把内部编排暴露给了用户。提示词约束不住——模型
# 会把“我调用了什么工具”当成可信度证据写进正文，所以这里在最终交付前做一次
# 确定性剥离：先删参数 JSON，再删“调用了某工具”的叙述，最后删工具标识符。
# ---------------------------------------------------------------------------

# 1) MCP 工具名：xxx_tool / xxx_mcp / xxx_v2，以及动词前缀族(search_/fetch_/...)
_TOOL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![\w-])(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_(?:tool|mcp|v\d{1,2})"
    r"|(?:search|fetch|get|find|unified|generate|learn|web|kb|semantic_scholar|"
    r"build|analyze|validate|prepare|save|read|delete|schedule|unschedule|list|"
    r"test|configure|convert|diagnose|verify|query)_[a-z][a-z0-9_]{2,}"
    r")(?![\w-])"
)
# 2) 工具参数 JSON（模型偶尔会把入参原样抄进答案）
_TOOL_ARGS_PATTERN = re.compile(
    r"\{[^{}\n]{0,200}?\"(?:pmid|pmids|query|term|doi|gene|keyword|top_k|ids|"
    r"sections|source|kind|value|limit|max_results)\""
    r"[^{}\n]{0,200}?\}"
)
# 2b) 模型把“工具调用语法”当正文吐出来：DeepSeek DSML / 通用 XML call 标记
_TOOL_MARKUP_PATTERN = re.compile(
    r"<\s*/?\s*[^<>]{0,16}DSML[^<>]{0,24}>"
    r"|<\s*/?\s*(?:tool_call|tool_calls|function_call|tool|function|parameter|"
    r"invoke|arguments|result)\b[^<>]{0,120}>"
    r"|<\|[^<>|]{0,30}\|>",
    re.IGNORECASE,
)
# 2c) 剥离标记后残留的裸 JSON 片段（如 {"kind": "pmcid", ...}）
_ARGS_FRAGMENT_PATTERN = re.compile(r"^\s*[\{\[].*[\}\]]\s*[,;]?\s*$")
# 3) “通过/调用 + 工具名(…)”：只删这个短语，保留同句里真正的结论与数据
_TOOL_CALL_PHRASE_PATTERN = re.compile(
    r"(?:(?:通过|借助)\s*)?(?:(?:调用|使用)(?:了)?\s*)?"
    r"(?:工具|函数|接口|API|MCP|插件)?\s*"
    r"(?:[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[A-Za-z_][A-Za-z0-9_]*\s*[（(])"
    r"(?:[^）)\n]{0,120}[）)])?"
)
# 4) “工具返回/函数输出/参数……”这类实现细节
_TOOL_RESULT_PATTERN = re.compile(
    r"(?:^|[，,。；;：:]\s*)(?:工具|函数|接口|API|MCP)"
    r"(?:返回|调用|参数|名称|结果|输出)[^。；;！!？?\n]{0,40}"
)
# 5) 删掉工具名之后残留的引导语
_LEFTOVER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:返回|输出|结果|返回值)(?:的)?\s*[:：，,]?\s*"
)
_LABEL_PREFIX_PATTERN = re.compile(
    r"^\s*(?:工具|函数|接口|API|MCP|插件|实现细节|调用过程)\s*[:：]\s*"
)
# 6) 整行只剩“工具：”“参数：”这类空标签
_BARE_LABEL_PATTERN = re.compile(
    r"^\s*(?:工具|函数|接口|API|MCP|插件|实现细节|调用过程|参数|入参)\s*[:：]?\s*$"
)
# 7) 句子里出现这些词，才认为它在讲“怎么查的”而不是“查到了什么”
_TOOL_TRACE_HINT = re.compile(
    r"(?:工具|函数|接口|MCP|插件|tool_calls?|function_call|tool_call_id)"
)
_SENTENCE_SPLIT_PATTERN = re.compile(r"([。；;！!？?\n])")


def _strip_tool_phrases(text: str) -> str:
    """删掉一句话里的工具调用短语，保留结论与数据。"""
    cleaned = _TOOL_CALL_PHRASE_PATTERN.sub(" ", text)
    cleaned = _TOOL_ARGS_PATTERN.sub(" ", cleaned)
    cleaned = _TOOL_RESULT_PATTERN.sub(" ", cleaned)
    cleaned = _TOOL_IDENTIFIER_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"`\s*`", " ", cleaned)
    cleaned = re.sub(r"[（(]\s*[)）]", " ", cleaned)
    cleaned = re.sub(
        r"(?:^|[\s（(])(?:tool_calls?|function_call|tool_call_id|arguments)(?:[\s）)]|$)",
        " ",
        cleaned,
    )
    cleaned = cleaned.lstrip()
    cleaned = _LABEL_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _LEFTOVER_PREFIX_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"^(?:我|我们)(?:已|已经|先|再次|也)?\s*[，,]?\s*", "", cleaned)
    cleaned = re.sub(r"^(?:之后|后)\s*[，,]?\s*", "", cleaned)
    cleaned = re.sub(r"^[，,、；;：:\s]+", "", cleaned)
    cleaned = re.sub(r"[，,、]\s*(?=[。；;！!？?])", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _sentence_has_tool_trace(sentence: str) -> bool:
    return bool(
        _TOOL_IDENTIFIER_PATTERN.search(sentence)
        or _TOOL_ARGS_PATTERN.search(sentence)
        or _TOOL_TRACE_HINT.search(sentence)
    )


def scrub_tool_leakage(text: str) -> tuple[str, int]:
    """从回答中剥离工具名/函数名/入参 JSON/调用过程；返回 (清洗后文本, 删除行数)。

    粒度是句子：讲“怎么查的”又没有数据的句子整句丢弃；夹带硬数据的句子只删
    工具短语，结论与数值原样保留。这样既不会泄露实现细节，也不会误删证据。
    """
    raw = str(text or "")
    if not raw.strip():
        return "", 0

    removed = 0
    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            kept.append("")
            continue
        if _BARE_LABEL_PATTERN.match(line):
            removed += 1
            continue
        # 工具调用语法/裸 JSON 片段：整行丢弃，绝不进正文
        if _TOOL_MARKUP_PATTERN.search(line) or _ARGS_FRAGMENT_PATTERN.match(line):
            removed += 1
            continue

        sentences: list[str] = []
        parts = _SENTENCE_SPLIT_PATTERN.split(line)
        for index in range(0, len(parts), 2):
            sentence = parts[index]
            delimiter = parts[index + 1] if index + 1 < len(parts) else ""
            if not sentence.strip():
                continue
            if not _sentence_has_tool_trace(sentence):
                sentences.append(sentence + delimiter)
                continue
            if _line_carries_hard_data(sentence):
                stripped = _strip_tool_phrases(sentence)
                if stripped and not _BARE_LABEL_PATTERN.match(stripped):
                    sentences.append(stripped + delimiter)
                else:
                    removed += 1
            else:
                removed += 1

        cleaned = "".join(sentences).strip()
        cleaned = re.sub(r"^[，,、；;：:]\s*", "", cleaned)
        if not cleaned or re.fullmatch(r"[。；;！!？?，,、\s]*", cleaned):
            if line.strip():
                removed += 1
            continue
        kept.append(cleaned)

    result = "\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result, removed



def _strip_answer_filler(text: str) -> tuple[str, int]:
    """剔除免责声明/学术套话/检索过程描述；返回 (清洗后文本, 删除行数)。"""
    raw = str(text or "")
    if not raw.strip():
        return "", 0

    lines = raw.splitlines()

    # 1. 显式的「免责声明 / 风险提示」小节整段丢弃
    for index, line in enumerate(lines):
        if re.match(
            r"^[#*\s>]*(?:免责声明|风险提示|特别声明|声明)\s*[:：]?\s*$",
            line.strip(),
        ):
            lines = lines[:index]
            break

    removed = 0
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        body = re.sub(r"^[#>*\-•\s]+", "", stripped)
        body = re.sub(r"^\d+[.、)]\s*", "", body)
        if (
            _line_is_disclaimer(body) or _line_is_filler(body)
        ) and not _line_carries_hard_data(body):
            removed += 1
            continue
        # 带数据的结论只去掉开头的套话，不删整行
        cleaned = re.sub(
            r"^(?:#+\s*)?(?:\*\*)?(?:综上所述|总而言之|总的来说|总之|由此可见)"
            r"[，,、：:]\s*",
            "",
            line,
        )
        cleaned = re.sub(
            r"^(?:根据(?:检索结果|查询结果|搜索结果|资料|工具返回)|"
            r"我(?:查到|检索到|搜索了)|资料(?:显示|表明)|以下是(?:基于|根据))"
            r"[，,、：:]\s*",
            "",
            cleaned,
        )
        cleaned = _INLINE_DISCLAIMER_PATTERN.sub("", cleaned).rstrip()
        kept.append(cleaned)

    result = "\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result, removed


def _remove_repeated_answer(text: str) -> str:
    midpoint = len(text) // 2
    first = text[:midpoint].rstrip()
    second = text[midpoint:].lstrip()
    if first and first == second:
        return first

    line_starts = [0]
    line_starts.extend(
        index + 1 for index, char in enumerate(text) if char == "\n"
    )
    for start in line_starts:
        if start >= midpoint:
            break
        line_end = text.find("\n", start)
        if line_end < 0:
            break
        first_line = text[start:line_end].strip()
        if not first_line:
            continue
        duplicate_start = text.find(first_line, midpoint)
        if duplicate_start <= 0:
            continue
        first_copy = text[start:duplicate_start].rstrip()
        second_copy = text[duplicate_start:].strip()
        if len(first_copy) >= 100 and first_copy == second_copy:
            return text[:start] + first_copy
    return text
