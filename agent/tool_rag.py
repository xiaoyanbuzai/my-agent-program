"""工具动态检索（Tool RAG）：来源族 → 语义相似度 → 专有意图分。"""

from __future__ import annotations

import logging
import re

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.builtin_tools import (
    generate_paper_docx_tool,
    kb_search_tool,
    learn_from_url_tool,
    search_papers_tool,
    search_pubmed_tool,
    web_news_tool,
)
from agent.common import (
    _is_supported_extra_tool,
    _tool_name,
)
from agent.intent import IntentDecision, _classify_intent
from agent.signals import _is_paper_request

from agent.signals import _token_hit
from agent.sources import _CITATION_CUES, _COMPOUND_CUES, _GENE_CUES, _ICD_CUES, _IMAGE_CUES, _KB_REFERENCE_CUES, _MODE_DIRECT, _PROGNOSIS_CUES, _PUBMED_SOURCE_CUES, _SOURCE_ARXIV, _SOURCE_CITATION, _SOURCE_COMPOUND, _SOURCE_DOCX, _SOURCE_GENE, _SOURCE_HERMES, _SOURCE_ICD, _SOURCE_IMAGE, _SOURCE_KB, _SOURCE_PUBMED, _SOURCE_URL, _SOURCE_WEB, _WEB_SOURCE_CUES

logger = logging.getLogger("web_app")


# 单轮最多绑定多少个工具：过多会撑大 prompt 并让模型难以选择
_MAX_SELECTED_TOOLS = 8
# 已经点名具体实体(PMID/DOI/变异)时，工具集进一步收紧
_MAX_ENTITY_TOOLS = 4
_ENTITY_INTENT_KINDS = frozenset({"pmid", "doi", "arxiv_id", "rsid", "hgvs", "nct"})
# Semantic Scholar MCP 的工具很多，只暴露最核心的少数几个
_SEMANTIC_SCHOLAR_CORE_TOOLS = (
    "semantic_scholar_search_papers",
    "semantic_scholar_get_paper",
    "semantic_scholar_recommendations",
)
# 工具被选中时的默认优先级（仅作为同分时的兜底顺序，真实排序由
# _rank_tool_names() 根据当前提问动态打分决定）
_TOOL_PRIORITY = (
    "kb_search_tool",
    "search_pubmed_tool",
    "unified_search",
    "search_papers_tool",
    "search_gene",
    "get_gene_details",
    "fetch_article_details",
    "generate_search_queries",
    "find_related_articles",
    "find_citing_articles",
    "get_fulltext",
    "semantic_scholar_search_papers",
    "semantic_scholar_get_paper",
    "semantic_scholar_recommendations",
    "web_news_tool",
    "learn_from_url_tool",
    "generate_paper_docx_tool",
)
_TOOL_PRIORITY_INDEX = {name: index for index, name in enumerate(_TOOL_PRIORITY)}


# ---------------------------------------------------------------------------
# 工具动态优先级
#
# 同一个问题该优先用哪个工具，不能靠一张写死的顺序表：问“PMID 42260443 的摘要”
# 时需要 fetch_article_details，问“TP53 基因有哪些突变”时需要 search_gene。
# 这里按「工具族正则 + 提问信号」给候选工具打分：
#   - 匹配用正则族而不是精确工具名，MCP 侧工具改名/新增仍然能命中；
#   - 只在该工具所属来源被意图层激活时才给满分，跨来源降权；
#   - 实体(PMID/URL/rsID)直接给高权重，保证专用工具不会被泛化检索挤掉。
# ---------------------------------------------------------------------------

# 工具族：工具名 → 所属来源集合(可属于多个，例如引用工具既属 PubMed 也属引用网络)
_TOOL_SOURCE_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"^kb_search"), (_SOURCE_KB,)),
    (
        re.compile(r"^(search_gene|get_gene_details|get_gene_literature|search_clinvar)$"),
        (_SOURCE_GENE,),
    ),
    (
        re.compile(r"^(search_compound|get_compound_details|get_compound_literature)$"),
        (_SOURCE_COMPOUND,),
    ),
    (re.compile(r"^convert_icd_mesh$"), (_SOURCE_ICD,)),
    (
        re.compile(r"^(search_biomedical_images|prepare_figure_search)$"),
        (_SOURCE_IMAGE, _SOURCE_PUBMED),
    ),
    (
        re.compile(
            r"^(find_citing_articles|find_related_articles|build_citation_tree"
            r"|get_citation_metrics|get_article_references)$"
        ),
        (_SOURCE_CITATION, _SOURCE_PUBMED),
    ),
    (re.compile(r"^(search_papers_tool|search_arxiv)$"), (_SOURCE_ARXIV,)),
    (re.compile(r"^semantic_scholar"), (_SOURCE_ARXIV,)),
    (re.compile(r"^web_news"), (_SOURCE_WEB,)),
    (re.compile(r"^learn_from_url"), (_SOURCE_URL,)),
    (re.compile(r"^generate_paper_docx"), (_SOURCE_DOCX,)),
    (
        re.compile(
            r"^(channels_list|conversations_list|events_poll|permissions_list_open)$"
        ),
        (_SOURCE_HERMES,),
    ),
    (
        re.compile(
            r"^(unified_search|search_pubmed_tool|analyze_search_query|validate_pico_plan"
            r"|generate_search_queries|fetch_article_details|get_fulltext"
            r"|verify_reference_list|prepare_export|save_literature_notes"
            r"|get_text_mined_terms|read_session|save_pipeline|list_pipelines"
            r"|load_pipeline|delete_pipeline|get_pipeline_history|schedule_pipeline"
            r"|unschedule_pipeline|build_research_chronicle|read_research_chronicle"
            r"|configure_institutional_access|get_institutional_link"
            r"|list_resolver_presets|test_institutional_access"
            r"|diagnose_institutional_access)$"
        ),
        (_SOURCE_PUBMED,),
    ),
)


_TOOL_NAME_SUFFIX_PATTERN = re.compile(r"(?:[_-]v\d+|[_-]tool|[_-]api|_mcp)+$", re.I)


def _tool_sources(tool_name: str) -> frozenset[str]:
    """工具名 → 来源族；未登记的工具返回空集(不参与按来源选题)。

    MCP 侧升级工具时经常出现 ``search_gene_v2``、``fetch_article_details_mcp``
    这类改名，这里先按原名匹配，失败再去掉版本/后缀重试，避免升级后工具
    突然全部“掉线”。注意先试原名：核心工具本身就叫 ``*_tool``。
    """
    name = str(tool_name or "").strip()
    if not name:
        return frozenset()
    for candidate in (name, _TOOL_NAME_SUFFIX_PATTERN.sub("", name)):
        if not candidate:
            continue
        for pattern, sources in _TOOL_SOURCE_PATTERNS:
            if pattern.search(candidate):
                return frozenset(sources)
    return frozenset()


# (可读标签, 工具名正则, 触发词, 权重)
_TOOL_CUE_RULES: tuple[
    tuple[str, re.Pattern[str], tuple[str, ...], float], ...
] = (
    (
        "知识库/本地文档",
        re.compile(r"^kb_search"),
        _KB_REFERENCE_CUES,
        6.0,
    ),
    (
        "PubMed 综述检索",
        re.compile(r"^unified_search$"),
        _PUBMED_SOURCE_CUES,
        5.0,
    ),
    (
        "PubMed 检索",
        re.compile(r"^search_pubmed_tool$"),
        _PUBMED_SOURCE_CUES,
        4.5,
    ),
    (
        "预后/生存",
        re.compile(r"^(unified_search|search_pubmed_tool)$"),
        _PROGNOSIS_CUES,
        3.0,
    ),
    (
        "检索式生成",
        re.compile(r"^(generate_search_queries|analyze_search_query)$"),
        _PUBMED_SOURCE_CUES,
        2.0,
    ),
    (
        "文献详情",
        re.compile(r"^(fetch_article_details|get_article_details)"),
        _PUBMED_SOURCE_CUES,
        3.0,
    ),
    (
        "全文",
        re.compile(r"^(get_fulltext|fetch_fulltext)"),
        ("全文", "fulltext", "全文下载", "原文"),
        6.0,
    ),
    (
        "基因检索",
        re.compile(r"^(search_gene|gene_search)"),
        _GENE_CUES,
        5.0,
    ),
    (
        "基因详情",
        re.compile(r"^get_gene_details"),
        _GENE_CUES,
        3.0,
    ),
    (
        "ClinVar 变异",
        re.compile(r"clinvar"),
        ("突变", "位点", "变异", "variant", "clinvar", "致病"),
        4.0,
    ),
    (
        "基因文献",
        re.compile(r"^get_gene_literature"),
        ("基因", "突变", "文献", "文献综述"),
        2.5,
    ),
    (
        "引用网络",
        re.compile(r"^(find_citing_articles|citing)"),
        _CITATION_CUES,
        5.0,
    ),
    (
        "相似文献",
        re.compile(r"^find_related_articles$"),
        _CITATION_CUES,
        4.0,
    ),
    (
        "引文指标",
        re.compile(r"^get_citation_metrics$"),
        ("影响因子", "引用次数", "citation metrics", "journal metric"),
        4.0,
    ),
    (
        "arXiv/论文检索",
        re.compile(r"^search_papers_tool$"),
        ("arxiv", "预印本", "paper", "论文检索"),
        4.5,
    ),
    (
        "语义学术检索",
        re.compile(r"^semantic_scholar"),
        ("arxiv", "预印本", "paper", "论文检索", "文献"),
        3.0,
    ),
    (
        "网页/新闻",
        re.compile(r"^web_news"),
        _WEB_SOURCE_CUES,
        5.0,
    ),
    (
        "网页入库",
        re.compile(r"^learn_from_url"),
        ("学习网页", "网页入库", "网页加入知识库", "抓取网页", "抓取", "入库"),
        1.5,
    ),
    (
        "文档生成",
        re.compile(r"^generate_paper_docx"),
        ("生成", "导出", "word", "docx", "doc", "报告文档"),
        7.0,
    ),
    (
        "化合物检索",
        re.compile(r"^(search_compound|compound_search)"),
        _COMPOUND_CUES,
        5.0,
    ),
    (
        "化合物详情",
        re.compile(r"^get_compound_details"),
        _COMPOUND_CUES,
        3.0,
    ),
    (
        "化合物文献",
        re.compile(r"^get_compound_literature"),
        ("化合物", "药物", "文献"),
        2.5,
    ),
    (
        "疾病编码",
        re.compile(r"^convert_icd_mesh$"),
        _ICD_CUES,
        5.0,
    ),
    (
        "医学图像",
        re.compile(r"^(search_biomedical_images|prepare_figure_search)$"),
        _IMAGE_CUES,
        4.0,
    ),
    (
        "Hermes 平台",
        re.compile(
            r"^(channels_list|conversations_list|events_poll|permissions_list_open)$"
        ),
        ("hermes", "平台频道", "平台会话", "权限审批"),
        3.0,
    ),
)

# 意图激活后必须保证可用、不能被截断掉的工具
_INTENT_REQUIRED_TOOLS: dict[str, tuple[str, ...]] = {
    _SOURCE_GENE: ("search_gene",),
    _SOURCE_DOCX: ("generate_paper_docx_tool",),
    _SOURCE_URL: ("learn_from_url_tool",),
    _SOURCE_ARXIV: ("search_papers_tool",),
    _SOURCE_WEB: ("web_news_tool",),
    _SOURCE_KB: ("kb_search_tool",),
}

# 意图激活后额外补进来的候选(MCP 工具名与本地工具混用)
_INTENT_EXTRA_TOOLS: dict[str, tuple[str, ...]] = {
    _SOURCE_GENE: ("search_gene", "get_gene_details", "search_clinvar", "search_pubmed_tool"),
    _SOURCE_CITATION: ("find_citing_articles", "find_related_articles"),
    _SOURCE_IMAGE: ("search_biomedical_images", "prepare_figure_search"),
    _SOURCE_COMPOUND: ("search_compound", "get_compound_details"),
    _SOURCE_ICD: ("convert_icd_mesh",),
}


def _tool_affinity(
    question: str, tool_name: str, intent: IntentDecision | None = None
) -> tuple[float, tuple[str, ...]]:
    """计算某个工具与当前提问的匹配度，返回 (分数, 可读依据)。

    ``intent`` 为空时内部现算；传入可避免同一轮重复分类。
    """
    lowered = (question or "").lower()
    if not lowered or not tool_name:
        return 0.0, ()

    decision = intent if intent is not None else _classify_intent(question)
    if decision.mode == _MODE_DIRECT:
        # 通用直答不该带任何检索工具，否则模型会为“什么是 XX”去翻知识库
        return 0.0, ()

    score = 0.0
    reasons: list[str] = []
    sources = _tool_sources(tool_name)
    aligned = bool(sources & set(decision.sources)) if sources else False

    for label, pattern, cues, weight in _TOOL_CUE_RULES:
        if not pattern.search(tool_name):
            continue
        if not any(_token_hit(cue, lowered) for cue in cues):
            continue
        effective = weight if aligned else weight * 0.3
        score += effective
        reasons.append(label)

    if tool_name.lower() in lowered:
        score += 10.0
        reasons.append("显式指定工具")

    kinds = decision.entity_kinds
    if kinds & {"pmid", "doi", "nct"}:
        if re.match(r"^(fetch_article_details|get_article_details)", tool_name):
            score += 8.0
            reasons.append("含文献编号")
        elif re.match(r"^(get_fulltext|fetch_fulltext)", tool_name):
            score += 5.0
            reasons.append("含文献编号")
    if "url" in kinds and tool_name.startswith("learn_from_url"):
        score += 8.0
        reasons.append("含 URL")
    if "arxiv_id" in kinds and tool_name.startswith(("semantic_scholar", "search_papers")):
        score += 6.0
        reasons.append("含 arXiv 编号")
    if ("rsid" in kinds or "hgvs" in kinds) and tool_name == "search_clinvar":
        score += 6.0
        reasons.append("含 rsID/HGVS")

    return score, tuple(reasons)


def _static_priority_key(name: str) -> tuple[int, str]:
    return (_TOOL_PRIORITY_INDEX.get(name, len(_TOOL_PRIORITY)), name)


def _rank_tool_names(question: str, names) -> list[str]:
    """按“与当前提问的匹配度 → 默认顺序 → 名称”排序候选工具。"""
    intent = _classify_intent(question)
    scored = []
    for name in names:
        affinity, _ = _tool_affinity(question, name, intent)
        scored.append((-affinity, _static_priority_key(name), name))
    scored.sort()
    return [item[2] for item in scored]


def _rank_tools_with_reasons(
    question: str, names
) -> list[tuple[str, float, tuple[str, ...]]]:
    """给日志用的排序结果：工具名、分数、命中依据。"""
    intent = _classify_intent(question)
    scored = []
    for name in names:
        affinity, reasons = _tool_affinity(question, name, intent)
        scored.append((-affinity, _static_priority_key(name), name, affinity, reasons))
    scored.sort()
    return [(item[2], item[3], item[4]) for item in scored]


def _hermes_tool_names(question: str, available: dict) -> set[str]:
    """Hermes 只暴露只读工具，按问句里的对象精确挑选。"""
    lowered = (question or "").lower()
    selected: set[str] = set()
    if any(term in lowered for term in ("频道", "渠道", "平台", "channel")):
        selected.add("channels_list")
    if any(term in lowered for term in ("会话", "conversation", "聊天记录")):
        selected.add("conversations_list")
    if any(term in lowered for term in ("事件", "实时", "轮询", "等待", "event")):
        selected.add("events_poll")
    if any(term in lowered for term in ("权限", "授权", "审批", "permission")):
        selected.add("permissions_list_open")
    return {name for name in selected if name in available}


def _select_tools(question: str, tools: list) -> list:
    """按「意图来源」选择小型工具集。

    与旧实现的区别：不再用一堆 if 拼关键词，而是先由 _classify_intent 判定
    本轮要查哪些源，再把这些源名下的工具族全部纳入候选，最后按动态优先级
    截断。通用直答(direct)直接返回空列表，不绑定任何检索工具。
    """
    available = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    if not available:
        return []

    intent = _classify_intent(question)
    if intent.mode == _MODE_DIRECT:
        return []

    if _SOURCE_HERMES in intent.sources:
        return _cap_selected_tools(
            _hermes_tool_names(question, available), available, question
        )

    active = set(intent.sources)
    selected: set[str] = set()
    for name in available:
        if _tool_sources(name) & active:
            selected.add(name)
    for source in intent.sources:
        selected.update(
            name for name in _INTENT_EXTRA_TOOLS.get(source, ()) if name in available
        )
        selected.update(
            name for name in _INTENT_REQUIRED_TOOLS.get(source, ()) if name in available
        )
    if _is_paper_request(question):
        selected.update({"kb_search_tool", "generate_paper_docx_tool"})
    selected &= set(available)
    if not selected:
        return []
    limit = _MAX_SELECTED_TOOLS
    if intent.entity_kinds & _ENTITY_INTENT_KINDS:
        # 已经点名了具体 PMID/DOI/位点，只需要“取详情 + 兜底检索”，
        # 不必把整个 PubMed 工具族都绑给模型（工具越少，选错概率越低）
        limit = _MAX_ENTITY_TOOLS
    return _cap_selected_tools(selected, available, question, limit=limit)


def _cap_selected_tools(
    selected: set[str], available: dict, question: str = "", limit: int | None = None
) -> list:
    """按“当前提问”的动态优先级截断到单轮工具上限。

    没有 question 时退回默认顺序，保证既有调用方行为不变。
    """
    names = [name for name in selected if name in available]
    if question and question.strip():
        ranked = _rank_tool_names(question, names)
    else:
        ranked = sorted(names, key=_static_priority_key)
    budget = _MAX_SELECTED_TOOLS if limit is None else max(1, limit)
    kept = ranked[:budget]
    return [available[name] for name in kept]


def _compact_tool_schema(tool) -> dict:
    """压缩工具及参数描述，不改工具名、参数类型、必填项或约束。"""
    schema = convert_to_openai_tool(tool)

    def compact_descriptions(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "description" and isinstance(child, str):
                    normalized = " ".join(child.split())
                    value[key] = normalized[:120].rstrip()
                    if len(normalized) > 120:
                        value[key] += "…"
                else:
                    compact_descriptions(child)
        elif isinstance(value, list):
            for child in value:
                compact_descriptions(child)

    compact_descriptions(schema)
    return schema


def _agent_tools(extra_tools=None) -> list:
    """注册本地知识库、外部检索/网页学习及论文生成工具。"""
    core_tools = [
        kb_search_tool,
        search_papers_tool,
        search_pubmed_tool,
        web_news_tool,
        learn_from_url_tool,
        generate_paper_docx_tool,
    ]
    hermes_tools = [
        tool
        for tool in (extra_tools or [])
        if _is_supported_extra_tool(tool)
    ]
    return core_tools + hermes_tools
