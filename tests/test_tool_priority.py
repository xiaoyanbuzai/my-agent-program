"""工具动态优先级测试。

验证“同一个问题优先用哪个工具”由提问内容决定，而不是一张写死的顺序表：
- PMID / 基因 / 综述 / arXiv / Word / URL 等信号各自把相关工具顶到前面；
- 没有信号时退回默认顺序，既有行为不变；
- 截断到单轮上限时按动态顺序保留，而不是按默认顺序。
"""

from __future__ import annotations

from types import SimpleNamespace

from web_app import (
    _MAX_SELECTED_TOOLS,
    _cap_selected_tools,
    _rank_tool_names,
    _rank_tools_with_reasons,
    _select_tools,
    _tool_affinity,
)

_PUBMED_TOOLS = [
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
    "generate_paper_docx_tool",
    "learn_from_url_tool",
    "web_news_tool",
]


def _tools(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=name) for name in names]


def _top(question: str, limit: int = 3) -> list[str]:
    return _rank_tool_names(question, _PUBMED_TOOLS)[:limit]


# ---------------------------------------------------------------- 单点信号


def test_pmid_question_prioritizes_article_details():
    top = _top("帮我取 PMID 42260443 和 42431390 这两篇的摘要和期刊信息")

    assert top[0] in {"fetch_article_details", "get_fulltext"}
    assert "search_papers_tool" not in top


def test_gene_question_prioritizes_gene_tools():
    top = _top("TP53 基因常见突变位点有哪些")

    assert top[0] == "search_gene"
    assert "get_gene_details" in top


def test_pubmed_review_question_prioritizes_unified_search():
    top = _top("请检索 PubMed 上关于 TP53 预后的最新综述")

    assert top[0] == "unified_search"
    assert "search_pubmed_tool" in top


def test_arxiv_question_prioritizes_paper_search():
    assert _top("用 arXiv 搜索多模态检索论文")[0] == "search_papers_tool"


def test_word_request_prioritizes_docx_generation():
    assert _top("基于这些资料生成一篇 Word 文档")[0] == "generate_paper_docx_tool"


def test_url_question_prioritizes_learning_tool():
    assert _top("https://example.com/a 这个网页帮我入库")[0] == "learn_from_url_tool"


def test_explicit_tool_name_is_ranked_first():
    top = _top("直接调用 get_fulltext 拿全文")

    assert top[0] == "get_fulltext"


def test_affinity_reasons_are_readable():
    score, reasons = _tool_affinity("PubMed 上 TP53 突变与预后", "unified_search")

    assert score > 0
    assert reasons
    assert any("PubMed" in reason for reason in reasons)


# ---------------------------------------------------------------- 动态性


def test_same_candidates_rank_differently_for_different_questions():
    pmid_order = _top("PMID 42260443 的摘要", 2)
    gene_order = _top("TP53 基因突变位点", 2)

    assert pmid_order != gene_order


def test_no_signal_falls_back_to_static_priority():
    ranked = _rank_tool_names("你好呀", _PUBMED_TOOLS)

    assert ranked[0] == "kb_search_tool"
    assert ranked.index("search_pubmed_tool") < ranked.index("search_gene")


def test_short_english_token_requires_word_boundary():
    standalone, _ = _tool_affinity("OS 和 PFS 分别是多少", "unified_search")
    embedded, _ = _tool_affinity("diagnosis of cancer", "unified_search")

    assert standalone > embedded


def test_rank_tools_with_reasons_matches_rank_order():
    with_reasons = _rank_tools_with_reasons("TP53 基因突变", _PUBMED_TOOLS)
    names = _rank_tool_names("TP53 基因突变", _PUBMED_TOOLS)

    assert [item[0] for item in with_reasons] == names
    assert with_reasons[0][1] > 0


# ---------------------------------------------------------------- 截断与端到端


def test_cap_selected_tools_keeps_dynamic_top_tools():
    available = {name: SimpleNamespace(name=name) for name in _PUBMED_TOOLS}

    kept = _cap_selected_tools(set(_PUBMED_TOOLS), available, "PMID 42260443 的摘要")

    assert len(kept) == _MAX_SELECTED_TOOLS
    assert kept[0].name in {"fetch_article_details", "get_fulltext"}


def test_cap_selected_tools_without_question_keeps_default_order():
    available = {name: SimpleNamespace(name=name) for name in _PUBMED_TOOLS}

    kept = _cap_selected_tools(set(_PUBMED_TOOLS), available)

    assert kept[0].name == "kb_search_tool"


def test_select_tools_orders_pubmed_tools_by_question():
    tools = _tools(*_PUBMED_TOOLS)

    gene = _select_tools("TP53 基因突变的常见位点", tools)
    review = _select_tools("PubMed 上 TP53 预后的最新综述", tools)

    gene_names = [tool.name for tool in gene]
    review_names = [tool.name for tool in review]

    assert gene_names[0] == "search_gene"
    assert review_names[0] == "unified_search"
    assert gene_names != review_names


def test_select_tools_still_caps_to_maximum():
    tools = _tools(*_PUBMED_TOOLS)

    selected = _select_tools("请检索 PubMed 上关于 TP53 预后的最新综述", tools)

    assert len(selected) <= _MAX_SELECTED_TOOLS


def test_pmid_question_adds_article_detail_candidates():
    """只有 PMID 的提问没有"PubMed"关键词，也必须拿到取详情的工具。"""
    tools = _tools(*_PUBMED_TOOLS)

    names = [tool.name for tool in _select_tools("帮我取 PMID 42260443 的摘要", tools)]

    assert names[0] == "fetch_article_details"
    assert "get_fulltext" in names


def test_gene_question_keeps_kb_and_adds_gene_tools():
    """基因/突变类提问既保留知识库，也要带上基因工具。"""
    tools = _tools(*_PUBMED_TOOLS)

    names = [tool.name for tool in _select_tools("TP53 基因常见突变位点有哪些", tools)]

    assert names[0] == "search_gene"
    assert "kb_search_tool" in names
    assert "search_pubmed_tool" in names
