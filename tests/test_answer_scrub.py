"""回答中「工具/函数痕迹」清洗测试。

用户视角只该看到结论与数据，不该看到 search_pubmed_tool、入参 JSON、
tool_calls 这类实现细节。这里锁定 scrub_tool_leakage 的剥离行为，
以及它不误伤结论行的边界。
"""

from __future__ import annotations

import pytest

from web_app import scrub_tool_leakage


def test_plain_conclusion_is_untouched():
    text = "TP53 突变显著降低肺癌患者生存（HR=1.74，95%CI 1.12–2.71，p=0.008）。"
    cleaned, dropped = scrub_tool_leakage(text)
    assert cleaned == text
    assert dropped == 0


@pytest.mark.parametrize(
    "tool_name",
    [
        "kb_search_tool",
        "search_pubmed_tool",
        "fetch_article_details",
        "search_gene_v2",
        "get_fulltext_mcp",
        "semantic_scholar_search_papers",
        "learn_from_url_tool",
    ],
)
def test_tool_names_never_survive(tool_name):
    cleaned, _ = scrub_tool_leakage(f"结论见下：{tool_name} 返回了 3 条记录。")
    assert tool_name not in cleaned
    assert "3 条记录" in cleaned


def test_json_arguments_are_stripped():
    text = '我已调用 search_pubmed_tool({"query": "TP53 lung cancer"}) 检索到 3 篇。'
    cleaned, _ = scrub_tool_leakage(text)
    assert "search_pubmed_tool" not in cleaned
    assert '"query"' not in cleaned
    assert "{" not in cleaned and "}" not in cleaned


def test_tool_only_lines_are_dropped_entirely():
    text = "工具：search_pubmed_tool\n结论：HR=2.10，95%CI 1.58–2.80。"
    cleaned, dropped = scrub_tool_leakage(text)
    assert "search_pubmed_tool" not in cleaned
    assert cleaned == "结论：HR=2.10，95%CI 1.58–2.80。"
    assert dropped >= 1


def test_hard_data_in_the_same_sentence_is_preserved():
    text = "通过调用 fetch_article_details(pmids='42260443') 得到期刊信息：Lung Cancer 2024。"
    cleaned, _ = scrub_tool_leakage(text)
    assert "fetch_article_details" not in cleaned
    assert "Lung Cancer 2024" in cleaned


def test_narration_without_data_is_dropped():
    text = "我通过 search_pubmed_tool 检索了一轮。\n中位 OS 为 11.2 个月（95%CI 9.8–12.6）。"
    cleaned, dropped = scrub_tool_leakage(text)
    assert "search_pubmed_tool" not in cleaned
    assert "检索了一轮" not in cleaned
    assert "11.2" in cleaned
    assert dropped >= 1


def test_tool_call_terminology_is_removed():
    cleaned, _ = scrub_tool_leakage("tool_calls: fetch_article_details")
    assert "tool_calls" not in cleaned
    assert "fetch_article_details" not in cleaned
    assert cleaned == ""


def test_backticked_identifier_is_removed():
    cleaned, _ = scrub_tool_leakage("`get_fulltext` 返回的全文显示 p=0.03。")
    assert "get_fulltext" not in cleaned
    assert "p=0.03" in cleaned


def test_multi_bullet_answer_keeps_structure():
    text = (
        "- 生存期：HR=2.10（95%CI 1.58–2.80），12 项研究。\n"
        "- 工具：fetch_article_details\n"
        "- 样本量：4,285 例。\n"
    )
    cleaned, dropped = scrub_tool_leakage(text)
    assert "fetch_article_details" not in cleaned
    assert cleaned.splitlines() == ["- 生存期：HR=2.10（95%CI 1.58–2.80），12 项研究。", "- 样本量：4,285 例。"]
    assert dropped >= 1


def test_ordinary_english_and_chinese_are_not_mangled():
    text = "使用 CRISPR 筛选发现 EGFR 通路富集（p=0.001）；PubMed 中位随访 24 个月。"
    cleaned, dropped = scrub_tool_leakage(text)
    assert cleaned == text
    assert dropped == 0


def test_empty_input():
    assert scrub_tool_leakage("") == ("", 0)
    assert scrub_tool_leakage("   \n\n") == ("", 0)


def test_deepseek_dsml_tool_call_markup_is_removed():
    raw = (
        "<｜｜DSML｜｜ calls>\n"
        '<｜｜DSML｜｜ parameter name="sections" string="true">OS</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="source" string="false">{"kind": "pmcid"}</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        "</｜｜DSML｜｜ calls>"
    )
    cleaned, dropped = scrub_tool_leakage(raw)
    assert cleaned == ""
    assert dropped >= 4


def test_xml_tool_call_markup_is_removed_but_data_survives():
    raw = "结论：HR=2.10（95%CI 1.58–2.80）。\n<tool_call>search_pubmed_tool</tool_call>"
    cleaned, _ = scrub_tool_leakage(raw)
    assert cleaned == "结论：HR=2.10（95%CI 1.58–2.80）。"
    assert "tool_call" not in cleaned


def test_bare_json_fragment_is_dropped():
    cleaned, _ = scrub_tool_leakage('{"kind": "pmcid", "value": "PMC13474983"}')
    assert cleaned == ""


def test_empty_answer_fallback_exists():
    import web_app

    assert web_app.EMPTY_ANSWER_FALLBACK.strip()
    assert "工具" not in web_app.EMPTY_ANSWER_FALLBACK
