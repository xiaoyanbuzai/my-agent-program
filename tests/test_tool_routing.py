from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from web_app import (
    _compact_tool_schema,
    _compress_retrieval_content,
    _agent_tools,
    _needs_planning,
    _select_tools,
    _system_prompt,
)


@tool
def sample_tool(query: str) -> str:
    """Search an external source for relevant information.

    This additional explanation should be compacted before it is sent to the model.
    """
    return query


def _named_tools(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=name) for name in names]


def test_tool_selection_keeps_only_tools_for_current_intent():
    tools = _named_tools(
        "kb_search_tool",
        "search_papers_tool",
        "learn_from_url_tool",
        "web_news_tool",
        "generate_paper_docx_tool",
        "channels_list",
        "conversations_list",
        "conversation_get",
        "messages_read",
    )

    assert [item.name for item in _select_tools("你好", tools)] == []
    assert [item.name for item in _select_tools("找论文：多模态检索", tools)] == [
        "kb_search_tool",
    ]
    assert [item.name for item in _select_tools("用 arXiv 搜索多模态检索论文", tools)] == [
        "search_papers_tool",
    ]
    assert [item.name for item in _select_tools("搜索最新新闻", tools)] == [
        "web_news_tool",
    ]
    assert [item.name for item in _select_tools("学习网页 https://example.com/a", tools)] == [
        "learn_from_url_tool",
    ]
    assert [item.name for item in _select_tools("Transformer 的 self-attention 怎么计算？", tools)] == []
    assert [item.name for item in _select_tools("本文的 self-attention 是怎么计算的？", tools)] == [
        "kb_search_tool",
    ]
    assert [item.name for item in _select_tools("查看 Hermes 的平台频道", tools)] == [
        "channels_list",
    ]
    assert [item.name for item in _select_tools("列出 Hermes 会话", tools)] == [
        "conversations_list",
    ]
    assert [item.name for item in _select_tools("读取 Hermes 消息记录", tools)] == []
    assert [item.name for item in _select_tools("找论文：多模态检索", tools)] == [
        "kb_search_tool",
    ]


def test_graph_uses_intent_tool_selection_and_skips_unneeded_router_call(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

    class FakeLLM:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.bound_tools = None
            self.invoke_count = 0
            llms.append(self)

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        def invoke(self, messages):
            self.invoke_count += 1
            return AIMessage(content="PASS" if self.temperature == 0 else "回答")

    llms = []
    monkeypatch.setattr("web_app.get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", FakeLLM)

    from web_app import build_graph

    graph = build_graph()
    graph.invoke(
        {"messages": [{"role": "user", "content": "Transformer 的 self-attention 怎么计算？"}]},
        {"configurable": {"thread_id": "tool-routing-test", "user_id": 7}},
    )

    chatbot_llm, judge_llm = llms
    assert chatbot_llm.bound_tools is None
    assert judge_llm.invoke_count == 1


def test_graph_binds_only_tools_selected_for_the_question(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

    class FakeLLM:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.bound_tools = None
            llms.append(self)

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        def invoke(self, messages):
            return AIMessage(content="PASS" if self.temperature == 0 else "回答")

    llms = []
    monkeypatch.setattr("web_app.get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", FakeLLM)

    from web_app import build_graph

    graph = build_graph()
    graph.invoke(
        {"messages": [{"role": "user", "content": "找论文：多模态检索"}]},
        {"configurable": {"thread_id": "tool-routing-test", "user_id": 7}},
    )

    chatbot_llm, _ = llms
    assert [
        tool["function"]["name"] for tool in chatbot_llm.bound_tools
    ] == ["kb_search_tool"]


def test_agent_registers_core_tools_and_supported_hermes_tools():
    hermes_tools = _named_tools(
        "channels_list",
        "conversations_list",
        "events_poll",
        "permissions_list_open",
        "messages_read",
        "messages_send",
        "permissions_respond",
    )

    tools = _agent_tools(hermes_tools)
    names = [getattr(item, "__name__", getattr(item, "name", "")) for item in tools]

    assert names == [
        "kb_search_tool",
        "search_papers_tool",
        "web_news_tool",
        "learn_from_url_tool",
        "generate_paper_docx_tool",
        "channels_list",
        "conversations_list",
        "events_poll",
        "permissions_list_open",
    ]


def test_agent_registers_core_tools_without_hermes():
    names = [tool.__name__ for tool in _agent_tools()]
    assert names == [
        "kb_search_tool",
        "search_papers_tool",
        "web_news_tool",
        "learn_from_url_tool",
        "generate_paper_docx_tool",
    ]


def test_system_prompt_describes_registered_external_tools():
    prompt = _system_prompt()

    assert "search_papers_tool" in prompt
    assert "web_news_tool" in prompt
    assert "learn_from_url_tool" in prompt
    assert "当前不提供联网" not in prompt
    assert "默认简洁直接" in prompt
    assert "不要为了知识库没有收录而拒绝回答" in prompt
    assert "比较多篇资料时按用户指定维度归纳共同点与差异" in prompt
    assert "不要声称能读取或发送消息" in prompt


def test_planning_is_limited_to_explicit_multistage_research():
    assert not _needs_planning("请详细解释 Transformer self-attention 的计算步骤")
    assert not _needs_planning("比较两篇论文的训练方法")
    assert not _needs_planning("基于知识库生成一篇论文 Word 文档")
    assert _needs_planning("请先检索两篇论文的训练方法，再比较它们的差异")
    assert _needs_planning("分别搜索两篇论文的实验结果并综合比较")


def test_tool_schema_compaction_preserves_call_contract():
    schema = _compact_tool_schema(sample_tool)["function"]

    assert schema["name"] == "sample_tool"
    assert "\n" not in schema["description"]
    assert len(schema["description"]) <= 121
    assert schema["parameters"]["properties"]["query"]["type"] == "string"
    assert schema["parameters"]["required"] == ["query"]


def test_core_tool_schemas_are_convertible():
    for item in _agent_tools():
        schema = _compact_tool_schema(item)["function"]
        assert schema["name"] == item.__name__


def test_retrieval_compression_prefers_query_relevant_sentences():
    content = "背景介绍内容。模型可以降低上下文长度。无关的历史说明。模型成本因此下降。"

    result = _compress_retrieval_content("模型成本", content)

    assert "模型可以降低上下文长度" in result
    assert "模型成本因此下降" in result
    assert "无关的历史说明" not in result


def test_retrieval_compression_enforces_character_budget():
    content = f"模型相关内容{'背景' * 400}。补充内容。再次补充。"

    result = _compress_retrieval_content("模型", content)

    assert len(result) <= 600
    assert result.endswith("…")
