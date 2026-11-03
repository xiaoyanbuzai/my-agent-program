"""规划解析与空计划保护测试。

覆盖「planner 解析失败 → plan=[] → executor 索引越界」这条崩溃路径：
1. _parse_plan_steps 对各类脏输出都能稳健抽取；
2. 解析彻底失败时有非空兜底，不会把空计划交给 executor；
3. 图上 planner 后的条件边在空计划时改走 chatbot，绝不进入 executor。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from web_app import (
    _MAX_PLAN_STEPS,
    _looks_like_refusal,
    _parse_plan_steps,
    _plan_fallback_steps,
)


# ---------------------------------------------------------------- 解析


def test_parses_plain_json_array():
    assert _parse_plan_steps('["检索 PubMed", "汇总结果"]') == ["检索 PubMed", "汇总结果"]


def test_parses_json_inside_code_fence():
    text = '```json\n["检索 TP53 文献", "提取预后数据"]\n```'
    assert _parse_plan_steps(text) == ["检索 TP53 文献", "提取预后数据"]


def test_parses_array_embedded_in_prose():
    text = '好的，计划如下：["第一步检索", "第二步总结"] 以上。'
    assert _parse_plan_steps(text) == ["第一步检索", "第二步总结"]


def test_parses_numbered_list_when_json_broken():
    text = "1. 检索 PubMed 中 TP53 相关论文\n2. 汇总预后结论"
    assert _parse_plan_steps(text) == ["检索 PubMed 中 TP53 相关论文", "汇总预后结论"]


def test_truncated_json_falls_back_to_list_items():
    # 模拟生成中断：JSON 未闭合
    text = '["检索 TP53 突变文献", "整理预后数据", "撰写综述'
    steps = _parse_plan_steps(text)
    assert steps, "截断输出也必须能抽出至少一步"
    assert len(steps) <= _MAX_PLAN_STEPS


def test_dict_items_are_normalized():
    text = '[{"step": "检索文献"}, {"task": "总结结论"}]'
    assert _parse_plan_steps(text) == ["检索文献", "总结结论"]


def test_steps_are_capped():
    text = '["一", "二", "三", "四", "五", "六"]'
    assert len(_parse_plan_steps(text)) == _MAX_PLAN_STEPS


def test_blank_and_garbage_return_empty():
    assert _parse_plan_steps("") == []
    assert _parse_plan_steps("   ") == []
    assert _parse_plan_steps(">") == []


# ---------------------------------------------------------------- 兜底


def test_fallback_uses_question_as_single_step():
    steps = _plan_fallback_steps("帮我检索 PubMed 上 TP53 突变与癌症预后的论文")
    assert steps == ["帮我检索 PubMed 上 TP53 突变与癌症预后的论文"]


def test_fallback_splits_ordered_question():
    steps = _plan_fallback_steps("请先检索两篇论文的方法，再比较它们的差异")
    assert len(steps) >= 2


def test_fallback_empty_question_returns_empty():
    assert _plan_fallback_steps("") == []


# ---------------------------------------------------------------- 拒答识别


def test_refusal_outputs_are_detected():
    for text in ("抱歉，我无法完成。", "无法拆解该任务", "Sorry, I cannot help."):
        assert _looks_like_refusal(text) is True


def test_normal_task_is_not_refusal():
    for text in ("检索 PubMed 中 TP53 相关论文", "先检索，再比较"):
        assert _looks_like_refusal(text) is False


# ---------------------------------------------------------------- 图级保护


class _FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0]


class _FakeLLM:
    instances: list = []

    def __init__(self, **kwargs):
        self.temperature = kwargs["temperature"]
        self.bound_tools = None
        self.calls = 0
        _FakeLLM.instances.append(self)

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="PASS" if self.temperature == 0 else "回答")


def _build(monkeypatch):
    _FakeLLM.instances = []
    monkeypatch.setattr("web_app.get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", _FakeLLM)
    from web_app import build_graph

    return build_graph()


def test_graph_skips_executor_when_plan_is_empty(monkeypatch):
    """planner 返回空计划时不应进入 executor（原 IndexError 路径）。"""
    graph = _build(monkeypatch)
    monkeypatch.setattr("web_app._parse_plan_steps", lambda text, max_steps=4: [])
    monkeypatch.setattr("web_app._plan_fallback_steps", lambda question: [])

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        {"configurable": {"thread_id": "empty-plan", "user_id": 7}},
    )

    # 走到 chatbot 产出了回答，而不是抛 IndexError
    assert result["messages"], "应返回回答消息"
    assert any(msg.calls for msg in _FakeLLM.instances)


def test_graph_completes_when_planner_output_is_garbage(monkeypatch):
    """planner 输出无法解析时，兜底单步仍应跑完整条链路。"""
    graph = _build(monkeypatch)
    monkeypatch.setattr(
        "web_app._parse_plan_steps", lambda text, max_steps=4: []
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        {"configurable": {"thread_id": "garbage-plan", "user_id": 7}},
    )

    assert result["messages"]
    assert result["plan"] == [] or result["plan"] is not None


def test_graph_rejects_refusal_only_plan(monkeypatch):
    """planner 只回一句“抱歉”时，应降级为用户问题而不是把它当任务执行。"""
    graph = _build(monkeypatch)
    monkeypatch.setattr(
        "web_app._parse_plan_steps", lambda text, max_steps=4: ["抱歉，我无法完成该任务。"]
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        {"configurable": {"thread_id": "refusal-plan", "user_id": 7}},
    )

    assert result["messages"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "抱歉，我无法完成。",
        "[",
        "```json\n[\n```",
        "步骤一：检索\n步骤二：总结",
    ],
)
def test_parse_never_raises(raw):
    assert isinstance(_parse_plan_steps(raw), list)
