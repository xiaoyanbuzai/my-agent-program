from web_app import _needs_planning, _replanner_prompt, _select_tools


class FakeTool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description


def test_replanner_prompt_requests_one_integrated_comparison():
    question = "比较两篇论文的训练方法"
    history = "论文 A 的结论。[来源: A p.1]\n论文 B 的结论。[来源: B p.2]"

    prompt = _replanner_prompt(question, history)

    assert question in prompt
    assert history in prompt
    assert "只生成一份完整、连贯的答复" in prompt
    assert "合并重复观点，只陈述一次" in prompt
    assert "共同点、关键差异" in prompt
    assert "不要把逐篇摘要拼接成答案" in prompt


def test_pubmed_review_requests_trigger_planning_and_pubmed_tool_selection():
    question = "帮我查一下 PubMed 上关于 CRISPR 基因编辑的最新综述"
    assert _needs_planning(question) is True

    tools = [
        FakeTool("unified_search", "PubMed unified academic search"),
        FakeTool("generate_search_queries", "Generate PubMed search strategies"),
        FakeTool("fetch_article_details", "Fetch details for PubMed IDs"),
        FakeTool("search_pubmed_tool", "Search PubMed biomedical literature"),
        FakeTool("search_papers_tool", "Arxiv paper search"),
        FakeTool("kb_search_tool", "Local knowledge base search"),
    ]

    selected = _select_tools(question, tools)
    assert {tool.name for tool in selected}.issuperset({"search_pubmed_tool"})
