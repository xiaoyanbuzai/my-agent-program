"""执行层数据丰富化测试。

覆盖三件事：
1. HTML 表格 → Markdown 表格（保留列结构、限制数量与行数）；
2. 返回里有 PMID 却没有摘要时，自动回查详情补全；
3. 补全失败/超长/空结果一律降级，不抛异常、不拖垮整轮。
另外用 build_graph 做一次端到端校验：executor 交给模型的证据里必须带着
补全后的摘要和 Markdown 表格。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from web_app import (
    _detail_tool_arguments,
    _enrich_tool_output,
    _extract_pmids,
    _html_tables_to_markdown,
    _needs_abstract_enrichment,
)

# ------------------------------------------------------------------ 表格

_TABLE_HTML = """
<h2>Table 1</h2>
<table>
  <tr><th>Study</th><th>HR</th><th>n</th></tr>
  <tr><td>A</td><td>1.8</td><td>1234</td></tr>
  <tr><td>B</td><td>0.9</td><td>567</td></tr>
</table>
"""


def test_html_table_becomes_markdown():
    text, count = _html_tables_to_markdown(_TABLE_HTML)

    assert count == 1
    assert "| Study | HR | n |" in text
    assert "| --- | --- | --- |" in text
    assert "| A | 1.8 | 1234 |" in text
    assert "<table" not in text


def test_html_table_limits_rows():
    rows = "".join(f"<tr><td>S{index}</td><td>{index}</td></tr>" for index in range(30))
    text, count = _html_tables_to_markdown(f"<table>{rows}</table>", max_rows=5)

    assert count == 1
    assert text.count("\n|") <= 7  # 表头 + 分隔 + 5 行


def test_html_tables_beyond_limit_are_dropped():
    text, count = _html_tables_to_markdown(
        "<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr></table>",
        max_tables=1,
    )

    assert count == 1
    assert "<table" not in text


def test_plain_text_is_untouched_by_table_conversion():
    text, count = _html_tables_to_markdown("PMID 42260443 没有表格")

    assert count == 0
    assert text == "PMID 42260443 没有表格"


# ------------------------------------------------------------------ PMID 抽取

def test_pmids_are_deduplicated_and_limited():
    text = "PMID 42260443 / PMID 42431390 / PMID 42260443 / PMID 1234567"

    assert _extract_pmids(text, limit=2) == ["42260443", "42431390"]
    assert _extract_pmids(text, limit=10) == ["42260443", "42431390", "1234567"]


def test_pmid_extraction_ignores_short_numbers():
    assert _extract_pmids("2024 年 1234 例样本") == []


def test_pmids_require_explicit_labels():
    """任意 7-9 位数字不能当 PMID：PMC 号、页码会误触发回查。"""
    assert _extract_pmids("PMC13474983 与 2026;26:1234") == []
    assert _extract_pmids('{"pmid": "42260443"}') == ["42260443"]
    assert _extract_pmids("PMID: 42260443") == ["42260443"]
    assert _extract_pmids("https://pubmed.ncbi.nlm.nih.gov/42260443/") == ["42260443"]


def test_abstract_detection():
    assert _needs_abstract_enrichment("PMID 42260443 — title only")
    assert not _needs_abstract_enrichment("PMID 42260443 Abstract: HR 1.8")
    assert not _needs_abstract_enrichment("摘要：TP53 突变与预后相关")
    assert not _needs_abstract_enrichment("")


# ------------------------------------------------------------------ 补全

def test_missing_abstract_triggers_batched_detail_fetch():
    calls: list[list[str]] = []

    def _fetch(pmids: list[str]) -> str:
        calls.append(pmids)
        return "Abstract: HR 1.8 (95%CI 1.2-2.7)"

    text, notes = _enrich_tool_output(
        "search_pubmed_tool",
        "PMID 42260443 / PMID 42431390 — titles only",
        fetch_detail=_fetch,
        max_details=2,
    )

    assert calls == [["42260443", "42431390"]], "应合并成一次批量回查"
    assert "【自动补全的摘要/详情】" in text
    assert "HR 1.8" in text
    assert "补全摘要×2" in notes


def test_detail_argument_name_follows_tool_schema():
    class _BatchTool:
        args = {"pmids": {"type": "string"}}

    class _SingleTool:
        args = {"pmid": {"type": "string"}}

    assert _detail_tool_arguments(_BatchTool(), ["1", "2"]) == {"pmids": "1,2"}
    assert _detail_tool_arguments(_SingleTool(), ["1", "2"]) == {"pmid": "1"}


def test_tool_error_text_is_not_treated_as_evidence():
    """MCP 会把参数错误当成功返回，这类文本必须丢弃。"""

    def _fetch(pmids: list[str]) -> str:
        return (
            "Invalid tool arguments. Consult tools/list inputSchema and retry "
            'with exact field names. {"status": "invalid_input", "executed": false}'
        )

    text, notes = _enrich_tool_output(
        "search_pubmed_tool", "PMID 42260443 — title only", fetch_detail=_fetch
    )

    assert "【自动补全" not in text
    assert "invalid_input" not in text
    assert notes == ()


def test_non_search_tool_never_triggers_detail_fetch():
    def _fetch(pmids: list[str]) -> str:  # pragma: no cover - 不应被调用
        raise AssertionError("基因/化合物类工具不应回查文献详情")

    text, notes = _enrich_tool_output(
        "search_gene", "PMID 42260443 — gene record", fetch_detail=_fetch
    )

    assert notes == ()
    assert text == "PMID 42260443 — gene record"


def test_existing_abstract_skips_detail_fetch():
    def _fetch(pmids: list[str]) -> str:  # pragma: no cover - 不应被调用
        raise AssertionError("已有摘要时不应回查")

    text, notes = _enrich_tool_output(
        "search_pubmed_tool",
        "PMID 42260443 Abstract: HR 1.8",
        fetch_detail=_fetch,
    )

    assert notes == ()
    assert text.startswith("PMID 42260443 Abstract")


def test_detail_fetch_failure_degrades_quietly():
    def _fetch(pmids: list[str]) -> str:
        raise RuntimeError("network down")

    text, notes = _enrich_tool_output(
        "search_pubmed_tool", "PMID 42260443 — title only", fetch_detail=_fetch
    )

    assert "【自动补全" not in text
    assert notes == ()


def test_enrichment_truncates_long_output():
    long_text = "PMID 42260443 " + "数据" * 2000

    text, _ = _enrich_tool_output("search_pubmed_tool", long_text, max_chars=200)

    assert len(text) <= 220
    assert text.endswith("…（已截断）")


def test_html_entities_are_unescaped():
    text, _ = _enrich_tool_output("web_news_tool", "AT&amp;T &lt;HR 1.8&gt;")

    assert "AT&T" in text
    assert "&amp;" not in text


def test_empty_tool_output_is_returned_as_is():
    assert _enrich_tool_output("kb_search_tool", "") == ("", ())


# ------------------------------------------------------------------ 端到端

SEARCH_RESULT = (
    "PMID 42260443 — TP53 mutation and survival\n"
    "<table><tr><th>Study</th><th>HR</th></tr>"
    "<tr><td>A</td><td>1.8</td></tr></table>"
)


@tool
def fetch_article_details(pmid: str) -> str:
    """Fetch PubMed article details for one PMID."""
    return f"Abstract for {pmid}: HR 1.8 (95%CI 1.2-2.7), n=1234, 2019."


class _FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0]


class _ScriptedLLM:
    instances: list = []

    def __init__(self, **kwargs):
        self.temperature = kwargs.get("temperature", 0)
        self.bound_tools = None
        self.calls: list[str] = []
        type(self).instances.append(self)

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        prompt = _prompt_text(messages)
        self.calls.append(prompt)
        if "学术研究指挥官" in prompt:
            return AIMessage(content='["在 PubMed 检索 TP53 突变与预后"]')
        if "执行助理" in prompt:
            names = [item["function"]["name"] for item in self.bound_tools or []]
            name = "search_pubmed_tool" if "search_pubmed_tool" in names else names[0]
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": {"query": "TP53"}, "id": "c1", "type": "tool_call"}
                ],
            )
        if "只提取能直接写进结论" in prompt:
            return AIMessage(content="摘要：HR 1.8")
        if "学术总结专家" in prompt:
            return AIMessage(content="最终答案")
        if "学术审稿人" in prompt:
            return AIMessage(content="PASS")
        return AIMessage(content="回答")


def _prompt_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(getattr(message, "content", "")))
    return "\n".join(parts)


def test_executor_hands_enriched_evidence_to_the_model(monkeypatch):
    monkeypatch.setattr("web_app.get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", _ScriptedLLM)
    monkeypatch.setattr("web_app._SafeChatOpenAI", _ScriptedLLM)
    monkeypatch.setattr("agent.routing._ROUTE_LLM", None, raising=False)

    def _invoke(fn, arguments, timeout):
        name = getattr(fn, "name", "") or getattr(fn, "__name__", "")
        if name == "fetch_article_details":
            return (f"Abstract for {arguments['pmid']}: HR 1.8 (95%CI 1.2-2.7)", True, "")
        return (SEARCH_RESULT, True, "")

    monkeypatch.setattr("web_app._invoke_tool_with_timeout", _invoke)

    from web_app import build_graph

    graph = build_graph(extra_tools=[fetch_article_details])
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请先检索 PubMed 上 TP53 突变与预后的文献，再比较差异",
                }
            ]
        },
        {"configurable": {"thread_id": "enrich-test", "user_id": 7}},
    )

    tool_texts = [
        str(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert any("| Study | HR |" in text for text in tool_texts), "HTML 表格应转成 Markdown"
    assert any("【自动补全的摘要/详情】" in text for text in tool_texts), "缺失摘要应被补全"
    assert any("Abstract for 42260443" in text for text in tool_texts)
