"""意图边界测试：把「通用直答」和「外部检索」精准切开。

覆盖三层判定：
1. 硬规则（寒暄 / 显式“不要检索” / 论文生成 / Hermes）；
2. 实体锚点（PMID、DOI、URL、rsID、HGVS、基因符号）；
3. 来源线索与兜底（PubMed / arXiv / 联网 / 知识库）。
同时校验工具选择真的跟着意图走：direct 不绑任何工具，实体题收窄工具集。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from web_app import (
    _MAX_ENTITY_TOOLS,
    _MODE_DIRECT,
    _MODE_RETRIEVAL,
    _SOURCE_ARXIV,
    _SOURCE_DOCX,
    _SOURCE_GENE,
    _SOURCE_HERMES,
    _SOURCE_KB,
    _SOURCE_PUBMED,
    _SOURCE_URL,
    _classify_intent,
    _select_tools,
)

_ALL_TOOLS = [
    "kb_search_tool",
    "search_pubmed_tool",
    "unified_search",
    "search_papers_tool",
    "fetch_article_details",
    "get_fulltext",
    "search_gene",
    "get_gene_details",
    "search_clinvar",
    "find_citing_articles",
    "find_related_articles",
    "get_citation_metrics",
    "semantic_scholar_search_papers",
    "convert_icd_mesh",
    "search_compound",
    "get_compound_details",
    "web_news_tool",
    "learn_from_url_tool",
    "generate_paper_docx_tool",
    "channels_list",
    "conversations_list",
]


def _tools(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=name) for name in names]


def _names(question: str, *names: str) -> list[str]:
    return [tool.name for tool in _select_tools(question, _tools(*(names or _ALL_TOOLS)))]


# ------------------------------------------------------------------ 直答

@pytest.mark.parametrize(
    "question",
    [
        "你好",
        "谢谢",
        "你是谁",
        "什么是 Transformer 的注意力机制？",
        "Transformer 的 self-attention 怎么计算？",
        "梯度消失和梯度爆炸有什么区别",
    ],
)
def test_general_knowledge_questions_stay_direct(question):
    decision = _classify_intent(question)

    assert decision.mode == _MODE_DIRECT
    assert decision.sources == ()
    assert not decision.needs_retrieval


def test_direct_intent_binds_no_tools():
    assert _names("什么是 Transformer") == []
    assert _names("你好") == []


def test_explicit_no_search_request_wins_over_entities():
    decision = _classify_intent("不要检索，直接说 PMID 42260443 是哪篇")

    assert decision.mode == _MODE_DIRECT
    assert "用户要求直接作答" in decision.reasons


def test_empty_question_is_direct():
    assert _classify_intent("").mode == _MODE_DIRECT
    assert _classify_intent("   ").mode == _MODE_DIRECT


# ------------------------------------------------------------------ 实体锚点

def test_pmid_is_a_pubmed_entity_anchor():
    decision = _classify_intent("帮我取 PMID 42260443 和 42431390 的摘要")

    assert decision.mode == _MODE_RETRIEVAL
    assert _SOURCE_PUBMED in decision.sources
    assert "pmid" in decision.entity_kinds
    assert any("PMID" in label for label in decision.entities)


def test_doi_is_a_pubmed_entity_anchor():
    decision = _classify_intent("帮我查一下 10.1038/s41586-023-06000-1 这篇文献")

    assert decision.mode == _MODE_RETRIEVAL
    assert _SOURCE_PUBMED in decision.sources
    assert "doi" in decision.entity_kinds


def test_url_selects_url_source():
    decision = _classify_intent("这个网页讲了什么 https://example.com/a")

    assert decision.mode == _MODE_RETRIEVAL
    assert decision.sources == (_SOURCE_URL,)
    assert "url" in decision.entity_kinds


def test_arxiv_id_selects_arxiv_source():
    decision = _classify_intent("看一下 arXiv:2301.01234 这篇预印本")

    assert decision.mode == _MODE_RETRIEVAL
    assert _SOURCE_ARXIV in decision.sources
    assert "arxiv_id" in decision.entity_kinds


def test_rsid_selects_gene_source():
    decision = _classify_intent("rs1234567 这个位点的临床意义是什么")

    assert decision.mode == _MODE_RETRIEVAL
    assert _SOURCE_GENE in decision.sources
    assert "rsid" in decision.entity_kinds


def test_gene_symbol_without_digit_is_detected():
    decision = _classify_intent("2024 年 EGFR 抑制剂有哪些新药")

    assert _SOURCE_GENE in decision.sources
    assert "gene_symbol" in decision.entity_kinds


# ------------------------------------------------------------------ 来源线索

def test_pubmed_review_does_not_drag_in_knowledge_base():
    names = _names("PubMed 上 TP53 预后的最新综述")

    assert names[0] == "unified_search"
    assert "kb_search_tool" not in names


def test_gene_question_keeps_knowledge_base_and_pubmed():
    names = _names("TP53 基因常见突变位点有哪些")

    assert names[0] == "search_gene"
    assert "kb_search_tool" in names
    assert "search_pubmed_tool" in names


def test_arxiv_and_web_intents_are_isolated():
    arxiv = _names("用 arXiv 搜索多模态检索论文")

    assert arxiv[0] == "search_papers_tool"
    assert set(arxiv) <= {"search_papers_tool", "semantic_scholar_search_papers"}
    assert _names("搜索最新新闻") == ["web_news_tool"]


def test_unrecognized_question_falls_back_to_knowledge_base():
    decision = _classify_intent("ccfa")

    assert decision.mode == _MODE_RETRIEVAL
    assert decision.sources == (_SOURCE_KB,)
    assert _names("ccfa") == ["kb_search_tool"]


def test_local_document_reference_selects_knowledge_base():
    decision = _classify_intent("本文的 self-attention 是怎么计算的？")

    assert decision.mode == _MODE_RETRIEVAL
    assert _SOURCE_KB in decision.sources
    assert _names("本文的 self-attention 是怎么计算的？") == ["kb_search_tool"]


def test_paper_generation_selects_kb_and_docx():
    decision = _classify_intent("基于知识库生成一篇论文 Word 文档")

    assert _SOURCE_DOCX in decision.sources
    assert _SOURCE_KB in decision.sources
    assert set(_names("基于知识库生成一篇论文 Word 文档")) == {
        "kb_search_tool",
        "generate_paper_docx_tool",
    }


def test_hermes_intent_only_exposes_read_only_tools():
    assert _classify_intent("查看 Hermes 的平台频道").sources == (_SOURCE_HERMES,)
    assert _names("查看 Hermes 的平台频道") == ["channels_list"]
    assert _names("列出 Hermes 会话") == ["conversations_list"]
    assert _names("读取 Hermes 消息记录") == []


# ------------------------------------------------------------------ 工具规模

def test_entity_question_uses_tight_tool_budget():
    names = _names("帮我取 PMID 42260443 的摘要")

    assert len(names) <= _MAX_ENTITY_TOOLS
    assert names[0] == "fetch_article_details"
    assert "get_fulltext" in names


def test_non_entity_question_may_bind_more_tools():
    names = _names("2024 年 EGFR 抑制剂有哪些新药")

    assert len(names) > _MAX_ENTITY_TOOLS


def test_unknown_tools_are_never_selected():
    assert _names("TP53 基因突变", "kb_search_tool", "mystery_tool") == ["kb_search_tool"]
