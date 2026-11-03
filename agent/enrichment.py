"""工具返回的数据丰富化：HTML 表格转 Markdown、补全摘要、错误文本识别。"""

from __future__ import annotations

import hashlib
import html

import logging
import re


logger = logging.getLogger("web_app")


# ---------------------------------------------------------------------------
# 执行层数据丰富化
#
# MCP 工具的原始返回经常“有链接没摘要、有统计表但都是 HTML”：直接丢给模型，
# 生成的结论就会缺硬数据。这里在调用与摘要之间加一层确定性补全：
#   1. HTML 表格 → Markdown 表格（保留列结构，最多 max_tables 个）；
#   2. 命中 PMID 但整段没有摘要 → 自动回查 fetch_article_details 补摘要；
#   3. HTML 实体转义还原、超长截断，保证 token 预算可控。
# 整个过程只依赖调用方注入的 fetch_detail，失败一律降级为“没补到”，不抛异常。
# ---------------------------------------------------------------------------

_HTML_TABLE_PATTERN = re.compile(r"<table\b.*?</table>", re.S | re.I)
_HTML_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_CELL_PATTERN = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_HTML_SPACE_PATTERN = re.compile(r"&nbsp;|&#160;")
_ABSTRACT_MARKERS = ("abstract", "摘要", "背景：", "objective:", "methods:")
# 只有检索类工具才回查摘要；基因/化合物工具里的编号不是文献编号
_ENRICHABLE_TOOL_PATTERN = re.compile(
    r"^(unified_search|search_pubmed|pubmed_search|semantic_scholar|search_papers"
    r"|find_citing|find_related|web_news)",
    re.I,
)
# MCP 工具习惯把参数校验失败当成“正常返回”，这些文本绝不能进证据
_TOOL_ERROR_MARKERS = (
    "invalid tool arguments",
    "invalid_input",
    "missing_fi",
    "unknown fields",
    "traceback (most recent call last)",
    "tool execution failed",
    "not found:",
)
# PMID 的可靠写法：JSON 字段、PMID 标签、PubMed 链接
_PMID_LABEL_PATTERNS = (
    re.compile(r'"pmid"\s*:\s*"?(\d{6,9})', re.I),
    re.compile(r"\bPMID[:\s#]*(\d{6,9})", re.I),
    re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})", re.I),
    re.compile(r"\bpubmed/(\d{6,9})", re.I),
)


def _html_fragment_to_text(fragment: str) -> str:
    text = _HTML_SPACE_PATTERN.sub(" ", fragment)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    return " ".join(html.unescape(text).split())


def _html_tables_to_markdown(
    raw: str, max_tables: int = 2, max_rows: int = 10
) -> tuple[str, int]:
    """把返回文本里的 HTML 表格转成 Markdown；返回 (新文本, 转换个数)。"""
    text = str(raw or "")
    if "<table" not in text.lower():
        return text, 0

    converted = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal converted
        if converted >= max_tables:
            return ""
        rows: list[list[str]] = []
        for row_html in _HTML_ROW_PATTERN.findall(match.group(0))[:max_rows]:
            cells = [
                _html_fragment_to_text(cell)
                for cell in _HTML_CELL_PATTERN.findall(row_html)
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        converted += 1
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        return "\n" + "\n".join(lines) + "\n"

    return _HTML_TABLE_PATTERN.sub(_replace, text), converted


def _extract_pmids(text: str, limit: int = 3) -> list[str]:
    """只从明确的 PMID 标记里取编号（JSON 字段 / PMID 标签 / PubMed 链接）。

    不能拿“任意 7-9 位数字”当 PMID：PMC13474983、年份+页码、样本量都会误命中，
    进而触发一次毫无意义的详情回查（MCP 还会以“成功文本”返回参数错误）。
    """
    raw = str(text or "")
    hits: list[tuple[int, str]] = []
    for pattern in _PMID_LABEL_PATTERNS:
        for match in pattern.finditer(raw):
            hits.append((match.start(), match.group(1)))
    hits.sort()

    seen: set[str] = set()
    found: list[str] = []
    for _position, value in hits:
        if value in seen:
            continue
        seen.add(value)
        found.append(value)
        if len(found) >= limit:
            break
    return found


def _looks_like_tool_error(text: str) -> bool:
    """MCP 工具常把参数错误当成正常返回值，这里识别出来避免当成证据。"""
    normalized = str(text or "").strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if lowered.startswith(("error", "错误", "invalid")):
        return True
    head = lowered[:500]
    return any(marker in head for marker in _TOOL_ERROR_MARKERS)


def _tool_supports_abstract_enrichment(tool_name: str) -> bool:
    """只有检索类工具才值得回查摘要；基因/化合物类返回的编号不是文献。"""
    return bool(_ENRICHABLE_TOOL_PATTERN.search(str(tool_name or "")))


def _needs_abstract_enrichment(raw: str) -> bool:
    """整段返回里完全没有摘要迹象时，才值得回查详情。"""
    lowered = str(raw or "").lower()
    if not lowered.strip():
        return False
    return not any(marker in lowered for marker in _ABSTRACT_MARKERS)


def _enrich_tool_output(
    tool_name: str,
    raw: str,
    *,
    fetch_detail=None,
    max_details: int = 2,
    max_chars: int = 1600,
) -> tuple[str, tuple[str, ...]]:
    """把工具原始返回补全成可直接写进结论的数据；返回 (新文本, 补全说明)。

    ``fetch_detail`` 接收一个 PMID 列表并返回整段详情（批量回查，一次调用）。
    """
    text = str(raw or "")
    if not text.strip():
        return text, ()

    notes: list[str] = []
    text = html.unescape(text)
    text, tables = _html_tables_to_markdown(text)
    if tables:
        notes.append(f"表格×{tables}")

    can_enrich = (
        fetch_detail is not None
        and max_details > 0
        and _tool_supports_abstract_enrichment(tool_name)
        and _needs_abstract_enrichment(text)
    )
    if can_enrich:
        pmids = _extract_pmids(text, limit=max_details)
        detail = ""
        if pmids:
            try:
                detail = fetch_detail(pmids)
            except Exception:
                logger.warning(
                    "自动补全摘要失败 tool=%s pmids=%s", tool_name, pmids, exc_info=True
                )
                detail = ""
        detail = " ".join(str(detail or "").split())
        if detail and not _looks_like_tool_error(detail):
            text = text + "\n\n【自动补全的摘要/详情】\n" + detail[: max_chars // 2]
            notes.append(f"补全摘要×{len(pmids)}")
        elif detail:
            logger.warning(
                "自动补全返回疑似错误，已丢弃 tool=%s head=%s",
                tool_name,
                detail[:120],
            )

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "…（已截断）"
    return text, tuple(notes)



def _detail_tool_arguments(fetcher, pmids: list[str]) -> dict:
    """按工具真实 schema 决定批量参数名。

    PubMed MCP 的 ``fetch_article_details`` 接收 ``pmids``（逗号分隔字符串），
    用 ``pmid`` 调用会得到一条“参数错误”的正常返回值——静默污染证据。
    """
    fields = getattr(fetcher, "args", None) or {}
    if isinstance(fields, dict):
        if "pmids" in fields:
            return {"pmids": ",".join(pmids)}
        if "pmid" in fields:
            return {"pmid": pmids[0]}
    return {"pmids": ",".join(pmids)}


def _merge_evidence_excerpts(
    evidence: list, messages: list, limit: int = 3000
) -> str:
    """把执行期证据与 messages 里的工具原文合并去重。

    replanner 以前只看 past_steps 的 300 字摘要，具体数值(HR/剂量/年份)会丢；
    这里补上原文摘录，同时按前 200 字做去重，避免同一段结果重复占预算。
    """
    chunks: list[str] = []
    seen: set[str] = set()

    for item in evidence or []:
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        excerpt = " ".join(str(item.get("excerpt") or "").split())
        if not excerpt:
            continue
        key = hashlib.sha256(excerpt[:200].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(f"[{item.get('tool') or '工具'}] {excerpt}")

    for message in messages or []:
        if getattr(message, "type", "") != "tool":
            continue
        text = " ".join(str(getattr(message, "content", "") or "").split())
        if not text:
            continue
        key = hashlib.sha256(text[:200].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(f"[工具结果] {text[:400]}")

    return "\n".join(chunks)[:limit]
