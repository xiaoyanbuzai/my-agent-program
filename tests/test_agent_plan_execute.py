"""planner / executor / replanner 链路的行为测试。

覆盖计划执行阶段的真实缺陷：
1. planner 不再被 _needs_planning 二次否决；
2. executor 拿不到 tool_calls 时重试一次，仍无证据就记缺口而不是编造结论；
3. 未授权工具名、工具异常/超时一律记为证据缺口，不进入摘要；
4. 工具结果摘要失败时回退原文摘录；
5. router 每轮清零 past_steps / plan / evidence / critique / reflection_count；
6. replanner 能看到证据原文与缺口。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from web_app import (
    _filter_plan_steps,
    _invoke_tool_with_timeout,
    _merge_evidence_excerpts,
    _repair_tool_pairs,
    _replanner_prompt,
    _summarize_tool_result,
)

PLAN_JSON = '["检索 TP53 文献", "汇总预后结论"]'
CONFIG = {"configurable": {"thread_id": "t", "user_id": 7}}


class _FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0]


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


class ScriptedLLM:
    """按提示词内容分派响应的假模型，覆盖 planner/executor/replanner/反思。"""

    instances: list = []
    PLAN = PLAN_JSON
    EXECUTOR = None

    def __init__(self, **kwargs):
        self.temperature = kwargs.get("temperature", 0)
        self.bound_tools = None
        self.calls: list[str] = []
        ScriptedLLM.instances.append(self)

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        prompt = _prompt_text(messages)
        self.calls.append(prompt)
        cls = type(self)
        if "任务编排判别器" in prompt:
            return AIMessage(
                content='{"route":"chatbot","confidence":0.9,"reason":"测试"}'
            )
        if "学术研究指挥官" in prompt:
            return AIMessage(content=cls.PLAN)
        if "执行助理" in prompt:
            executor = cls.EXECUTOR
            if callable(executor):
                return executor(self)
            return executor or AIMessage(content="没有合适的工具")
        if "提取出对最终对比" in prompt:
            return AIMessage(content="摘要：TP53 突变与较差预后相关。")
        if "学术总结专家" in prompt:
            return AIMessage(content="最终报告")
        if "学术审稿人" in prompt:
            return AIMessage(content="PASS")
        return AIMessage(content="回答")


def _build(monkeypatch, executor=None, plan: str = PLAN_JSON):
    ScriptedLLM.instances = []
    fake = type(
        "FakeLLM",
        (ScriptedLLM,),
        {"PLAN": plan, "EXECUTOR": executor},
    )
    monkeypatch.setattr("web_app.get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", fake)
    monkeypatch.setattr("web_app._SafeChatOpenAI", fake)
    monkeypatch.setattr("agent.routing._ROUTE_LLM", None, raising=False)
    from web_app import build_graph

    return build_graph()


def _executor_calls() -> list[str]:
    return [
        call
        for instance in ScriptedLLM.instances
        for call in instance.calls
        if "执行助理" in call
    ]
# ---------------------------------------------------------------- planner


def test_planner_runs_even_when_keyword_rule_says_no(monkeypatch):
    """router 判 planner 后，planner 不得再用 _needs_planning 自我否决。"""
    monkeypatch.setattr("agent.routing._needs_planning", lambda text: False)
    graph = _build(monkeypatch, AIMessage(content="没有工具"))

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文的训练方法，再比较它们的差异"}]},
        CONFIG,
    )

    assert len(result["past_steps"]) == 2, "两步计划都应被执行"
    assert _executor_calls(), "planner 产出的计划必须真的进入 executor"


def test_planner_falls_back_when_output_is_meta_only(monkeypatch):
    graph = _build(monkeypatch, AIMessage(content="没有工具"), plan='["分析用户需求"]')

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        CONFIG,
    )

    assert result["past_steps"], "空转步骤被过滤后必须回退到用户问题本身"


# ---------------------------------------------------------------- executor


def test_executor_retries_once_then_records_evidence_gap(monkeypatch):
    graph = _build(monkeypatch, AIMessage(content="我没有调用工具"))

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        CONFIG,
    )

    assert len(_executor_calls()) == 4, "两个步骤各重试一次"
    assert len(result["evidence_gaps"]) == 2
    assert all("未获得检索证据" in gap for gap in result["evidence_gaps"])
    assert all("未获得检索证据" in value for _, value in result["past_steps"])


def test_executor_rejects_tool_outside_selection(monkeypatch):
    def _boom(fn, arguments, timeout):
        raise AssertionError("未授权工具绝不能被调用")

    monkeypatch.setattr("web_app._invoke_tool_with_timeout", _boom)
    bad_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "not_a_real_tool", "args": {}, "id": "call_x", "type": "tool_call"}
        ],
    )
    graph = _build(monkeypatch, bad_call)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        CONFIG,
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and "本轮不可用" in str(message.content)
    ]
    assert tool_messages, "应给模型回一条“工具不可用”的 ToolMessage"
    assert any("本轮不可用" in gap for gap in result["evidence_gaps"])


def test_executor_tool_failure_becomes_gap_not_evidence(monkeypatch):
    monkeypatch.setattr(
        "web_app._invoke_tool_with_timeout",
        lambda fn, arguments, timeout: ("", False, "TimeoutError: 超时"),
    )
    monkeypatch.setattr(
        "web_app._summarize_tool_result",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("失败结果不应摘要")),
    )

    def _executor(llm):
        name = llm.bound_tools[0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {}, "id": "c1", "type": "tool_call"}],
        )

    graph = _build(monkeypatch, _executor)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        CONFIG,
    )

    assert any("TimeoutError" in gap for gap in result["evidence_gaps"])
    assert result["evidence"], "失败也要留痕"
    assert all(item["ok"] is False for item in result["evidence"])


def test_executor_records_evidence_and_summary(monkeypatch):
    monkeypatch.setattr(
        "web_app._invoke_tool_with_timeout",
        lambda fn, arguments, timeout: ("原始检索结果", True, ""),
    )
    monkeypatch.setattr("web_app._summarize_tool_result", lambda *a, **k: "摘要 A")

    def _executor(llm):
        name = llm.bound_tools[0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {}, "id": "c1", "type": "tool_call"}],
        )

    graph = _build(monkeypatch, _executor)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请先检索两篇论文，再比较它们的差异"}]},
        CONFIG,
    )

    assert result["past_steps"][0] == ("检索 TP53 文献", "摘要 A")
    assert result["evidence"][0]["ok"] is True
    assert result["evidence"][0]["excerpt"] == "原始检索结果"
    assert not result["evidence_gaps"]


# ---------------------------------------------------------------- 轮次隔离


def test_router_resets_stale_turn_state(monkeypatch):
    """上一轮残留的 plan/past_steps/critique 不能影响新一轮。"""
    graph = _build(monkeypatch, AIMessage(content="没有工具"))

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "你好"}],
            "plan": ["上一轮的步骤"],
            "past_steps": [("旧任务", "旧结果")],
            "evidence": [{"task": "旧", "tool": "kb_search_tool", "excerpt": "x", "ok": True}],
            "evidence_gaps": ["旧缺口"],
            "critique": "缺少引用",
            "reflection_count": 2,
        },
        CONFIG,
    )

    assert result["plan"] == []
    assert result["past_steps"] == []
    assert result["evidence"] == []
    assert result["evidence_gaps"] == []
    assert result["critique"] == "PASS"
    assert result["reflection_count"] == 1, "反思计数应从 0 重新开始"


# ---------------------------------------------------------------- 纯函数


def test_filter_plan_steps_drops_meta_duplicates_and_refusals():
    steps = [
        "分析用户需求",
        "检索 TP53 文献",
        "检索 TP53 文献",
        "抱歉，我无法完成该任务。",
        "  ",
    ]

    assert _filter_plan_steps(steps) == ["检索 TP53 文献"]


def test_summarize_tool_result_falls_back_to_excerpt():
    class _BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("boom")

    assert _summarize_tool_result(_BrokenLLM(), "任务", " 原始  结果 ") == "原始 结果"


def test_summarize_tool_result_uses_llm_output():
    class _EchoLLM:
        def invoke(self, prompt):
            return AIMessage(content=" 核心  事实 ")

    assert _summarize_tool_result(_EchoLLM(), "任务", "原始") == "核心 事实"


def test_merge_evidence_excerpts_dedupes_and_keeps_tool_messages():
    evidence = [
        {"task": "t", "tool": "kb_search_tool", "excerpt": "同一段证据", "ok": True},
        {"task": "t", "tool": "kb_search_tool", "excerpt": "同一段证据", "ok": True},
        {"task": "t", "tool": "pubmed", "excerpt": "失败的", "ok": False},
    ]
    messages = [ToolMessage(content="messages 里的工具结果", tool_call_id="c1")]

    merged = _merge_evidence_excerpts(evidence, messages)

    assert merged.count("同一段证据") == 1
    assert "失败的" not in merged
    assert "messages 里的工具结果" in merged


def test_replanner_prompt_carries_evidence_and_gaps():
    prompt = _replanner_prompt(
        "问题",
        "【步骤】结果",
        evidence_excerpts="[kb_search_tool] HR=0.62",
        gaps="- 步骤2：未获得检索证据",
    )

    assert "HR=0.62" in prompt
    assert "证据缺口" in prompt
    assert "未获得检索证据" in prompt
    assert "不得编造" in prompt


def test_replanner_prompt_without_extra_sections_stays_backward_compatible():
    prompt = _replanner_prompt("问题", "历史")

    assert "证据缺口" not in prompt
    assert "只生成一份完整、连贯的答复" in prompt


def test_invoke_tool_with_timeout_handles_sync_tool():
    assert _invoke_tool_with_timeout(lambda **kwargs: "ok", {}, 1) == ("ok", True, "")


def test_invoke_tool_with_timeout_reports_exception():
    def _boom(**kwargs):
        raise ValueError("bad args")

    result, ok, error = _invoke_tool_with_timeout(_boom, {}, 1)

    assert result == ""
    assert ok is False
    assert "ValueError" in error


def test_invoke_tool_with_timeout_supports_async_tool():
    class _AsyncTool:
        coroutine = object()

        async def ainvoke(self, arguments):
            return "async ok"

    assert _invoke_tool_with_timeout(_AsyncTool(), {}, 1) == ("async ok", True, "")


def test_invoke_tool_with_timeout_aborts_slow_async_tool():
    class _SlowTool:
        coroutine = object()

        async def ainvoke(self, arguments):
            await asyncio.sleep(5)
            return "never"

    result, ok, error = _invoke_tool_with_timeout(_SlowTool(), {}, 0.01)

    assert result == ""
    assert ok is False
    assert "TimeoutError" in error


# ---------------------------------------------------------------- 工具消息配对


def _ai_with_calls(*call_ids):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": f"tool_{index}", "args": {}, "id": call_id, "type": "tool_call"}
            for index, call_id in enumerate(call_ids)
        ],
    )


def test_repair_tool_pairs_keeps_valid_history_untouched():
    messages = [
        AIMessage(content="q"),
        _ai_with_calls("call_1", "call_2"),
        ToolMessage(content="r1", tool_call_id="call_1"),
        ToolMessage(content="r2", tool_call_id="call_2"),
    ]

    repaired = _repair_tool_pairs(messages)

    assert [type(item).__name__ for item in repaired] == [type(item).__name__ for item in messages]
    assert [getattr(item, "tool_call_id", None) for item in repaired] == [
        None,
        None,
        "call_1",
        "call_2",
    ]


def test_repair_tool_pairs_fills_missing_tool_result():
    messages = [
        _ai_with_calls("call_1", "call_2"),
        ToolMessage(content="r1", tool_call_id="call_1"),
    ]

    repaired = _repair_tool_pairs(messages)

    assert len(repaired) == 3
    assert repaired[2].tool_call_id == "call_2"
    assert "没有返回结果" in str(repaired[2].content)


def test_repair_tool_pairs_drops_orphan_tool_message():
    messages = [
        AIMessage(content="q"),
        ToolMessage(content="孤儿", tool_call_id="call_x"),
    ]

    assert len(_repair_tool_pairs(messages)) == 1


def test_repair_tool_pairs_normalizes_empty_call_id():
    messages = [
        _ai_with_calls(""),
        ToolMessage(content="结果", tool_call_id=""),
    ]

    repaired = _repair_tool_pairs(messages)

    assert len(repaired) == 2
    call_id = repaired[0].tool_calls[0]["id"]
    assert call_id
    assert repaired[1].tool_call_id == call_id
    assert repaired[1].content == "结果"


# ---------------------------------------------------------------- 工具轮次上限

def test_tool_rounds_are_capped_and_force_a_final_answer(monkeypatch):
    """模型反复检索时不能无限循环：到上限后不再绑定工具，直接基于证据作答。"""

    class LoopingLLM:
        instances: list = []

        def __init__(self, **kwargs):
            self.temperature = kwargs.get("temperature", 0)
            self.bound_tools = None
            self.calls: list[str] = []
            LoopingLLM.instances.append(self)

        def bind_tools(self, tools):
            # 真实 ChatOpenAI 返回新的 runnable，不会污染未绑定的实例
            clone = type(self)()
            clone.bound_tools = tools
            return clone

        def invoke(self, messages):
            prompt = _prompt_text(messages)
            self.calls.append(prompt)
            if "学术审稿人" in prompt:
                return AIMessage(content="PASS")
            if self.bound_tools:
                name = self.bound_tools[0]["function"]["name"]
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": {}, "id": "c1", "type": "tool_call"}
                    ],
                )
            return AIMessage(content="最终回答")

    monkeypatch.setattr("web_app.get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr("web_app._ReasoningContentChatOpenAI", LoopingLLM)
    monkeypatch.setattr("web_app._SafeChatOpenAI", LoopingLLM)
    monkeypatch.setattr("agent.routing._ROUTE_LLM", None, raising=False)
    monkeypatch.setattr(
        "web_app._invoke_tool_with_timeout",
        lambda fn, arguments, timeout: ("工具原始结果", True, ""),
    )

    from web_app import build_graph, get_settings

    graph = build_graph()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "我的文档里 checkpoint 是什么"}]},
        CONFIG,
    )

    assert result["tool_rounds"] == get_settings().max_tool_rounds
    assert result["messages"][-1].content == "最终回答"
    chatbot_calls = [
        instance
        for instance in LoopingLLM.instances
        for call in instance.calls
        if "学术审稿人" not in call
    ]
    assert chatbot_calls, "收束后仍要由 chatbot 生成最终回答"
