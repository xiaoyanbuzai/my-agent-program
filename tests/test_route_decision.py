"""开局路由（planner vs chatbot）判定测试。

覆盖 _decide_route 的三层判定：硬规则 → 特征打分 → 模糊带 LLM 仲裁，
以及仲裁结果的容错解析、置信度门限与缓存。全部为纯函数/打桩测试，
不访问网络。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from agent import routing as agent_routing
from config import get_settings
from web_app import (
    _classify_route_with_llm,
    _decide_route,
    _needs_planning,
    _parse_route_payload,
    _route_features,
    _score_route,
)


@pytest.fixture(autouse=True)
def _reset_route_state(monkeypatch):
    """每个用例都从空的缓存/模型实例开始，避免相互污染。"""
    monkeypatch.setattr(agent_routing, "_ROUTE_CACHE", OrderedDict())
    monkeypatch.setattr(agent_routing, "_ROUTE_LLM", None, raising=False)


def _settings(monkeypatch, **overrides):
    base = get_settings()
    patched = replace(base, **overrides)
    monkeypatch.setattr(agent_routing, "get_settings", lambda: patched)
    return patched


class _FakeRouteLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=self.payload)


# ---------------------------------------------------------------- 硬规则


def test_empty_question_routes_to_chatbot():
    decision = _decide_route("   ")
    assert decision.route == "chatbot"
    assert decision.source == "hard_rule"


def test_greeting_routes_to_chatbot():
    assert _decide_route("你好").route == "chatbot"


def test_paper_generation_routes_to_chatbot():
    decision = _decide_route("基于知识库生成一篇论文 Word 文档")
    assert decision.route == "chatbot"
    assert decision.source == "hard_rule"
    assert "论文生成模式" in decision.reasons


def test_explicit_direct_answer_request_wins_over_scoring():
    decision = _decide_route("不要检索，直接回答 TP53 的常见突变位点")
    assert decision.route == "chatbot"
    assert "用户要求直接作答" in decision.reasons


def test_explicit_plan_request_goes_to_planner():
    decision = _decide_route("请分步帮我研究 TP53 突变与预后的关系")
    assert decision.route == "planner"
    assert decision.source == "hard_rule"


# ---------------------------------------------------------------- 打分层


@pytest.mark.parametrize(
    "question",
    [
        "请先检索两篇论文的训练方法，再比较它们的差异",
        "分别搜索两篇论文的实验结果并综合比较",
        "请先检索 PubMed 上 TP53 突变与癌症预后的最新论文，"
        "再分别汇总近期综述和原始研究的结论并比较差异",
    ],
)
def test_multi_stage_questions_route_to_planner(question):
    decision = _decide_route(question)
    assert decision.route == "planner"
    assert decision.source == "score"
    assert decision.score >= 3
    assert decision.reasons, "打分命中时必须给出可读依据"


@pytest.mark.parametrize(
    "question",
    [
        "什么是 TP53 基因",
        "你好",
        "比较两篇论文的训练方法",
        "这篇论文的实验设置是什么？",
    ],
)
def test_simple_questions_route_to_chatbot(question):
    decision = _decide_route(question)
    assert decision.route == "chatbot"


def test_multi_entity_comparison_lands_in_ambiguous_band(monkeypatch):
    """多对象对比只有 2 分，落在模糊带，交给 LLM 仲裁而不是直接判 chatbot。"""
    monkeypatch.setattr(
        agent_routing,
        "_classify_route_with_llm",
        lambda question: ("planner", 0.9, "多对象需分别取证"),
    )

    decision = _decide_route("比较 TP53 和 KRAS 突变在肺癌中的预后差异")

    assert decision.score == 2
    assert decision.route == "planner"
    assert decision.source == "llm"


def test_conceptual_question_never_scores_high():
    features = _route_features("什么是 TP53 基因？它有几个异构体？")
    score, reasons = _score_route(features)
    assert score <= 1
    assert score == 1 or "概念解释,降级" in reasons


# ---------------------------------------------------------------- 模糊带


def test_rules_mode_never_calls_classifier(monkeypatch):
    _settings(monkeypatch, router_mode="rules")

    def _boom(question):
        raise AssertionError("rules 模式不应调用 LLM 仲裁")

    monkeypatch.setattr(agent_routing, "_classify_route_with_llm", _boom)

    decision = _decide_route("比较 TP53 和 KRAS 突变在肺癌中的预后差异")
    assert decision.route == "chatbot"
    assert decision.source == "score"
    assert any("模糊带" in reason for reason in decision.reasons)


def test_llm_disabled_never_calls_classifier(monkeypatch):
    _settings(monkeypatch, router_llm_enabled=False)
    monkeypatch.setattr(
        agent_routing,
        "_classify_route_with_llm",
        lambda question: ("planner", 0.99, "x"),
    )

    decision = _decide_route("比较 TP53 和 KRAS 突变在肺癌中的预后差异")
    assert decision.route == "chatbot"
    assert decision.source == "score"
    assert any("模糊带" in reason for reason in decision.reasons)


def test_low_confidence_falls_back_to_chatbot(monkeypatch):
    monkeypatch.setattr(
        agent_routing,
        "_classify_route_with_llm",
        lambda question: ("planner", 0.3, "不太确定"),
    )

    decision = _decide_route("比较 TP53 和 KRAS 突变在肺癌中的预后差异")
    assert decision.route == "chatbot"
    assert decision.source == "llm_fallback"


def test_classifier_failure_falls_back_to_chatbot(monkeypatch):
    monkeypatch.setattr(
        agent_routing, "_classify_route_with_llm", lambda question: (None, 0.0, "")
    )

    decision = _decide_route("比较 TP53 和 KRAS 突变在肺癌中的预后差异")
    assert decision.route == "chatbot"
    assert decision.source == "llm_fallback"


def test_llm_mode_asks_classifier_even_for_low_score(monkeypatch):
    _settings(monkeypatch, router_mode="llm")
    monkeypatch.setattr(
        agent_routing,
        "_classify_route_with_llm",
        lambda question: ("planner", 0.8, "需要两步"),
    )

    decision = _decide_route("介绍一下这篇论文")
    assert decision.route == "planner"
    assert decision.source == "llm"


def test_forced_modes_override_scoring(monkeypatch):
    _settings(monkeypatch, router_mode="always_planner")
    assert _decide_route("你好").route == "planner"

    _settings(monkeypatch, router_mode="always_chatbot")
    decision = _decide_route("请先检索两篇论文，再比较它们的差异")
    assert decision.route == "chatbot"
    assert decision.source == "forced"


# ---------------------------------------------------------------- 解析与缓存


@pytest.mark.parametrize(
    "raw",
    [
        '{"route": "planner", "confidence": 0.8, "reason": "多源"}',
        '```json\n{"route": "planner", "confidence": 0.8}\n```',
        '好的，结论如下：{"route": "PLANNER", "confidence": 0.8} 以上。',
        '{"needs_planning": true}',
        '{"route": "planner"}',
    ],
)
def test_parse_route_payload_accepts_planner_variants(raw):
    route, confidence, _ = _parse_route_payload(raw)
    assert route == "planner"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    "raw",
    [
        '{"route": "chatbot", "confidence": 0.7}',
        '{"needs_planning": false}',
        "route: chatbot",
    ],
)
def test_parse_route_payload_accepts_chatbot_variants(raw):
    route, _, _ = _parse_route_payload(raw)
    assert route == "chatbot"


@pytest.mark.parametrize("raw", ["", "我觉得应该分步", "PASS", "```json\n{}\n```"])
def test_parse_route_payload_returns_none_for_garbage(raw):
    assert _parse_route_payload(raw)[0] is None


def test_parse_route_payload_clamps_confidence():
    assert _parse_route_payload('{"route":"planner","confidence":9}')[1] == 1.0
    assert _parse_route_payload('{"route":"planner","confidence":-3}')[1] == 0.0


def test_classifier_result_is_cached(monkeypatch):
    _settings(monkeypatch, router_cache_ttl=3600)
    fake = _FakeRouteLLM('{"route": "planner", "confidence": 0.9, "reason": "多源"}')
    monkeypatch.setattr(agent_routing, "_route_llm", lambda: fake)

    first = _classify_route_with_llm("比较 A 和 B 的差异")
    second = _classify_route_with_llm("比较 A 和 B 的差异")

    assert first[0] == second[0] == "planner"
    assert fake.calls == 1, "同一问题应命中缓存，不重复调用 LLM"


def test_classifier_cache_can_be_disabled(monkeypatch):
    _settings(monkeypatch, router_cache_ttl=0)
    fake = _FakeRouteLLM('{"route": "planner", "confidence": 0.9}')
    monkeypatch.setattr(agent_routing, "_route_llm", lambda: fake)

    _classify_route_with_llm("比较 A 和 B 的差异")
    _classify_route_with_llm("比较 A 和 B 的差异")

    assert fake.calls == 2


def test_classifier_network_failure_is_swallowed(monkeypatch):
    class _BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("network down")

    _settings(monkeypatch, router_cache_ttl=3600)
    monkeypatch.setattr(agent_routing, "_route_llm", lambda: _BrokenLLM())

    assert _classify_route_with_llm("比较 A 和 B 的差异") == (None, 0.0, "")


# ---------------------------------------------------------------- 与旧规则的一致性


@pytest.mark.parametrize(
    "question,expected",
    [
        ("请详细解释 Transformer self-attention 的计算步骤", False),
        ("比较两篇论文的训练方法", False),
        ("基于知识库生成一篇论文 Word 文档", False),
        ("请先检索两篇论文的训练方法，再比较它们的差异", True),
        ("分别搜索两篇论文的实验结果并综合比较", True),
    ],
)
def test_needs_planning_regression_cases_unchanged(question, expected):
    """_needs_planning 仍是打分信号之一，行为保持向后兼容。"""
    assert _needs_planning(question) is expected
