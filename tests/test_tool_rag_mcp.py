"""Tool RAG 与 MCP 专有意图匹配测试。

关注三件事：
1. 工具族按正则识别，MCP 侧改名（search_gene_v2 / fetch_article_details_mcp）
   不会导致工具“掉线”；
2. 打分受意图门控：通用直答一律 0 分，跨来源工具降权；
3. 选题阶段必须保住意图要求的工具（基因题保 search_gene，PMID 题保取详情）。
"""

from __future__ import annotations

from types import SimpleNamespace

from web_app import (
    _INTENT_REQUIRED_TOOLS,
    _MAX_ENTITY_TOOLS,
    _SOURCE_ARXIV,
    _SOURCE_GENE,
    _SOURCE_HERMES,
    _SOURCE_KB,
    _SOURCE_PUBMED,
    _classify_intent,
    _rank_tool_names,
    _select_tools,
    _tool_affinity,
    _tool_sources,
)

# PubMed MCP 的常见工具族 + 核心本地工具
_PUBMED_FAMILY = [
    "unified_search",
    "search_pubmed_tool",
    "analyze_search_query",
    "validate_pico_plan",
    "generate_search_queries",
    "find_related_articles",
    "find_citing_articles",
    "get_article_references",
    "fetch_article_details",
    "get_citation_metrics",
    "verify_reference_list",
    "prepare_export",
    "save_literature_notes",
    "get_fulltext",
    "get_text_mined_terms",
    "get_article_figures",
    "build_citation_tree",
]
_GENE_FAMILY = ["search_gene", "get_gene_details", "get_gene_literature", "search_clinvar"]


def _tools(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=name) for name in names]


# --------------------------------------------------------------- 工具族识别

def test_tool_families_are_detected_by_regex():
    assert _tool_sources("kb_search_tool") == frozenset({_SOURCE_KB})
    assert _SOURCE_PUBMED in _tool_sources("unified_search")
    assert _SOURCE_PUBMED in _tool_sources("fetch_article_details")
    assert _tool_sources("search_gene") == frozenset({_SOURCE_GENE})
    assert _tool_sources("semantic_scholar_search_papers") == frozenset({_SOURCE_ARXIV})
    assert _tool_sources("channels_list") == frozenset({_SOURCE_HERMES})


def test_mcp_tool_renames_still_match():
    """MCP 升级常带 _v2/_mcp/_tool 后缀，识别不能因此失效。"""
    assert _tool_sources("search_gene_v2") == frozenset({_SOURCE_GENE})
    assert _SOURCE_PUBMED in _tool_sources("fetch_article_details_mcp")
    assert _SOURCE_PUBMED in _tool_sources("unified_search_v2")
    assert _tool_sources("not_a_real_tool") == frozenset()


# --------------------------------------------------------------- 意图门控

def test_direct_intent_scores_every_tool_zero():
    for name in ("kb_search_tool", "unified_search", "web_news_tool"):
        score, reasons = _tool_affinity("什么是 Transformer", name)
        assert score == 0.0
        assert reasons == ()


def test_cross_source_tool_is_demoted():
    """同一句里的 arXiv 工具，在 PubMed 问题里必须低于 arXiv 问题里的自己。"""
    pubmed_question = "PubMed 上 TP53 预后的最新综述"
    arxiv_question = "arXiv 上 TP53 预后的最新论文"

    pubmed_score, _ = _tool_affinity(pubmed_question, "search_papers_tool")
    arxiv_score, _ = _tool_affinity(arxiv_question, "search_papers_tool")

    assert arxiv_score > pubmed_score


def test_entity_bonus_puts_detail_tool_first():
    ranked = _rank_tool_names(
        "帮我取 PMID 42260443 的摘要",
        ["kb_search_tool", "unified_search", "fetch_article_details", "get_fulltext"],
    )

    assert ranked[0] == "fetch_article_details"


def test_affinity_reasons_explain_the_choice():
    score, reasons = _tool_affinity("TP53 基因常见突变位点", "search_gene")

    assert score > 0
    assert any("基因" in reason for reason in reasons)


# --------------------------------------------------------------- 选题保底

def test_required_gene_tool_survives_large_pubmed_family():
    tools = _tools("kb_search_tool", *_PUBMED_FAMILY, *_GENE_FAMILY)

    names = [tool.name for tool in _select_tools("TP53 基因常见突变位点", tools)]

    assert "search_gene" in names
    required = set(_INTENT_REQUIRED_TOOLS[_SOURCE_GENE])
    assert required <= set(names)


def test_entity_question_keeps_detail_tool_even_in_huge_tool_set():
    tools = _tools("kb_search_tool", *_PUBMED_FAMILY, *_GENE_FAMILY)

    names = [tool.name for tool in _select_tools("取 PMID 42260443 的摘要", tools)]

    assert len(names) <= _MAX_ENTITY_TOOLS
    assert "fetch_article_details" in names


def test_knowledge_base_tool_is_kept_for_pure_kb_question():
    tools = _tools("kb_search_tool", *_PUBMED_FAMILY, *_GENE_FAMILY)

    names = [tool.name for tool in _select_tools("我的文档里怎么描述 checkpoint 的", tools)]

    assert names[0] == "kb_search_tool"


def test_hermes_selection_ignores_other_families():
    tools = _tools("kb_search_tool", "unified_search", "channels_list", "permissions_list_open")

    names = [tool.name for tool in _select_tools("看看 Hermes 待处理权限", tools)]

    assert names == ["permissions_list_open"]


def test_intent_sources_drive_candidate_pool():
    intent = _classify_intent("TP53 基因突变位点")
    tools = _tools("kb_search_tool", *_PUBMED_FAMILY, *_GENE_FAMILY, "web_news_tool")

    names = {tool.name for tool in _select_tools("TP53 基因突变位点", tools)}

    assert "web_news_tool" not in names
    assert _SOURCE_GENE in intent.sources
