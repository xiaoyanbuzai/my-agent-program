"""开局判别：planner(分步调研) 还是 chatbot(直接作答)。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from agent.llm import (
    _ReasoningContentChatOpenAI,
    _SafeChatOpenAI,
    _is_deepseek_reasoning_endpoint,
)
from agent.signals import (
    _PURE_GREETING_PATTERN,
    _ROUTE_DIRECT_MARKERS,
    _is_paper_request,
)
from config import get_settings

from agent.intent import _is_general_knowledge_request

logger = logging.getLogger("web_app")


def _needs_planning(text: str) -> bool:
    """识别需要拆解的复杂研究任务，覆盖 PubMed 与多源综述型检索。"""
    lowered = (text or "").lower()
    if not lowered or _is_paper_request(lowered):
        return False

    if lowered.count(".pdf") >= 2 or lowered.count("《") >= 2:
        return True

    pubmed_related = any(
        term in lowered
        for term in (
            "pubmed",
            "ncbi",
            "文献综述",
            "综述",
            "review",
            "latest review",
            "最新综述",
            "研究进展",
            "state of the art",
        )
    )
    has_search = any(term in lowered for term in ("检索", "查找", "查询", "搜索", "联网", "查一下"))
    has_synthesis = any(
        term in lowered
        for term in (
            "比较",
            "对比",
            "综合",
            "汇总",
            "总结",
            "抓取",
            "入库",
            "最新",
            "趋势",
            "进展",
            "证据",
        )
    )
    separate_sources = (
        any(term in lowered for term in ("分别", "逐一", "多篇", "多篇文献", "多源"))
        and has_search
        and has_synthesis
    )
    ordered_steps = (
        has_search
        and has_synthesis
        and bool(re.search(r"(?:先|首先).{0,100}(?:再|然后|接着|最后)", lowered))
    )
    latest_review = pubmed_related and has_search and has_synthesis
    return separate_sources or ordered_steps or latest_review


# ---------------------------------------------------------------------------
# 开局路由:多步研究(planner) vs 直接作答(chatbot)
#
# 判定分三层,任何一层给出确定结论就停止,越往后成本越高:
#   1. 硬规则(空问题/问候/论文生成/用户显式要求)
#   2. 可解释打分(关键词特征投票),命中两端带宽直接出结论
#   3. 模糊带才调用 LLM 仲裁(严格 JSON + 置信度 + 结果缓存)
# 每次决策都带 source/score/reasons,便于日志与前端提示解释“为什么这样走”。
# ---------------------------------------------------------------------------

_ROUTE_MODES = frozenset(
    {"auto", "rules", "llm", "always_planner", "always_chatbot"}
)
_ROUTE_LLM_MIN_CONFIDENCE = 0.6
_ROUTE_CACHE_MAX = 256

_ROUTE_PLAN_MARKERS = (
    "分步",
    "一步一步",
    "一步步",
    "制定计划",
    "拆解",
    "多个步骤",
    "计划一下",
    "先给我计划",
    "列个计划",
)


_ROUTE_SEARCH_CUES = (
    "检索",
    "查找",
    "查询",
    "搜索",
    "联网",
    "查一下",
    "帮我查",
    "搜一下",
)
_ROUTE_SYNTHESIS_CUES = (
    "比较",
    "对比",
    "综合",
    "汇总",
    "总结",
    "抓取",
    "入库",
    "最新",
    "趋势",
    "进展",
    "证据",
    "梳理",
)
_ROUTE_PUBMED_CUES = (
    "pubmed",
    "ncbi",
    "文献综述",
    "综述",
    "review",
    "最新综述",
    "研究进展",
    "state of the art",
)
_ROUTE_SOURCE_GROUPS = (
    ("pubmed", "ncbi"),
    ("arxiv",),
    ("知识库", "上传的", "我上传", "文档库"),
    ("联网", "全网", "网页", "新闻"),
    ("论文", "文献", "paper", "literature"),
)
_ROUTE_ENTITY_SPLIT = re.compile(r"(?:和|与|、|vs\.?|对比|以及)")
_ROUTE_ENTITY_HINTS = ("比较", "对比", "差异", "优劣", "哪个", "区别")


@dataclass(frozen=True)
class RouteFeatures:
    """从用户问题里抽出的可解释特征,供打分与日志使用。"""

    question: str
    length: int
    is_paper: bool
    is_greeting: bool
    is_conceptual: bool
    explicit_plan: bool
    explicit_direct: bool
    search_cues: int
    synthesis_cues: int
    source_cues: int
    doc_refs: int
    multi_entity: bool
    ordered_markers: bool
    sub_questions: int
    needs_planning: bool


@dataclass(frozen=True)
class RouteDecision:
    """一次开局路由决策的完整依据,便于日志/前端解释。"""

    route: str
    source: str
    score: int
    reasons: tuple[str, ...]

    def reason_text(self) -> str:
        return " + ".join(self.reasons) if self.reasons else "无"


def _route_features(text: str) -> RouteFeatures:
    """抽取路由特征;纯函数,不读配置、不调模型。"""
    raw = " ".join(str(text or "").split())
    lowered = raw.lower()
    search_cues = sum(1 for cue in _ROUTE_SEARCH_CUES if cue in lowered)
    synthesis_cues = sum(1 for cue in _ROUTE_SYNTHESIS_CUES if cue in lowered)
    source_cues = sum(
        1
        for group in _ROUTE_SOURCE_GROUPS
        if any(term in lowered for term in group)
    )
    doc_refs = lowered.count(".pdf") + lowered.count("《")
    sub_questions = raw.count("？") + raw.count("?")
    return RouteFeatures(
        question=raw,
        length=len(raw),
        is_paper=_is_paper_request(raw),
        is_greeting=bool(_PURE_GREETING_PATTERN.fullmatch(raw)),
        is_conceptual=_is_general_knowledge_request(raw),
        explicit_plan=any(term in lowered for term in _ROUTE_PLAN_MARKERS),
        explicit_direct=any(term in lowered for term in _ROUTE_DIRECT_MARKERS),
        search_cues=search_cues,
        synthesis_cues=synthesis_cues,
        source_cues=source_cues,
        doc_refs=doc_refs,
        multi_entity=(
            bool(_ROUTE_ENTITY_SPLIT.search(raw))
            and any(term in lowered for term in _ROUTE_ENTITY_HINTS)
        ),
        ordered_markers=bool(
            re.search(r"(?:先|首先).{0,100}(?:再|然后|接着|最后)", lowered)
        ),
        sub_questions=sub_questions,
        needs_planning=_needs_planning(raw),
    )


def _score_route(features: RouteFeatures) -> tuple[int, tuple[str, ...]]:
    """按特征投票;返回 (分数, 可读依据)。分数越高越倾向 planner。"""
    score = 0
    reasons: list[str] = []

    if features.doc_refs >= 2:
        score += 3
        reasons.append(f"多文档({features.doc_refs})")
    if features.needs_planning:
        score += 2
        reasons.append("多阶段任务")
    if features.search_cues and features.synthesis_cues:
        score += 2
        reasons.append("检索+综合")
    if features.source_cues >= 2:
        score += 2
        reasons.append(f"多来源({features.source_cues})")
    if features.multi_entity:
        score += 2
        reasons.append("多实体比较")
    if features.ordered_markers:
        score += 2
        reasons.append("显式顺序")
    if features.sub_questions >= 2:
        score += 1
        reasons.append(f"多子问题({features.sub_questions})")
    if features.length >= 120 and features.synthesis_cues:
        score += 1
        reasons.append("长问题且需综合")

    if (
        features.is_conceptual
        and features.source_cues == 0
        and not features.multi_entity
    ):
        # “什么是 TP53”这类概念解释不需要拆步骤,压到 chatbot 带宽。
        # 注意排除多实体比较：_is_general_knowledge_request 对“差异/区别”
        # 的匹配很宽，会把“比较 A 和 B 的预后差异”也判成概念题。
        if score > 1:
            reasons.append("概念解释,降级")
        return min(score, 1), tuple(reasons)
    return score, tuple(reasons)


def _parse_route_payload(raw: str) -> tuple[str | None, float, str]:
    """容错解析路由判定输出;无法解析时返回 (None, 0.0, "")。"""
    text = str(raw or "")
    if not text.strip():
        return None, 0.0, ""

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    def _normalize_route(value) -> str | None:
        if value is None:
            return None
        # 注意不能用 value or ""：needs_planning=false 会被当成空值丢掉。
        token = str(value).strip().strip('"').lower()
        if token in {"planner", "plan", "true", "yes", "1", "计划"}:
            return "planner"
        if token in {"chatbot", "chat", "false", "no", "0", "直答"}:
            return "chatbot"
        return None

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        route = _normalize_route(
            payload.get("route")
            if payload.get("route") is not None
            else payload.get("needs_planning")
        )
        if route is None:
            continue
        try:
            confidence = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        reason = " ".join(str(payload.get("reason", "")).split())[:120]
        return route, max(0.0, min(1.0, confidence)), reason

    # 兜底:输出里至少出现了 route=planner 这类键值对
    match = re.search(r'"?route"?\s*[:=]\s*"?(planner|chatbot)"?', text, re.I)
    if match:
        return match.group(1).lower(), 0.5, "正则兜底解析"
    return None, 0.0, ""


_ROUTE_LLM: Any = None
_ROUTE_CACHE: "OrderedDict[str, tuple[float, RouteDecision]]" = OrderedDict()


def _route_llm():
    """懒加载路由仲裁模型:低温、短超时,失败由调用方兜底。"""
    global _ROUTE_LLM
    if _ROUTE_LLM is None:
        settings = get_settings()
        llm_class = (
            _ReasoningContentChatOpenAI
            if _is_deepseek_reasoning_endpoint()
            else _SafeChatOpenAI
        )
        _ROUTE_LLM = llm_class(
            model=settings.llm_model,
            temperature=0,
            timeout=min(settings.llm_timeout, 20.0),
            max_retries=1,
        )
    return _ROUTE_LLM


_ROUTE_LLM_PROMPT = """你是一个任务编排判别器。判断用户这次提问应该「分步调研」还是「直接作答」。

判定为 planner(分步调研)的典型情形:
- 需要从多个来源/多篇文献收集材料,再综合比较;
- 明确包含多个阶段(先检索,再汇总/对比);
- 需要分别处理多个对象后给出联合结论。
判定为 chatbot(直接作答)的典型情形:
- 单一事实、定义、概念解释;
- 寒暄、追问、对上一轮回答的澄清;
- 只需一次检索或一次工具调用即可回答。

只输出一个 JSON 对象,不要解释、不要 Markdown,格式:
{"route": "planner" 或 "chatbot", "confidence": 0 到 1 的小数, "reason": "不超过 30 字的理由"}

示例:
问:什么是 TP53 基因?
答:{"route": "chatbot", "confidence": 0.95, "reason": "单一概念解释"}
问:请先检索 PubMed 上 TP53 突变与预后的最新论文,再分别汇总综述和原始研究的结论并比较差异
答:{"route": "planner", "confidence": 0.93, "reason": "多来源检索后综合比较"}
问:比较 TP53 和 KRAS 突变在肺癌中的预后差异
答:{"route": "planner", "confidence": 0.72, "reason": "多对象对比需要分别取证"}
问:这篇论文的实验设置是什么?
答:{"route": "chatbot", "confidence": 0.9, "reason": "针对单篇文献的查找"}

用户提问:
"""


def _route_cache_key(question: str) -> str:
    normalized = " ".join(str(question or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _classify_route_with_llm(
    question: str,
) -> tuple[str | None, float, str]:
    """模糊带仲裁;命中缓存则零成本返回。异常一律返回 (None, 0.0, "")。"""
    settings = get_settings()
    key = _route_cache_key(question)
    ttl = max(0, settings.router_cache_ttl)
    now = time.time()

    cached = _ROUTE_CACHE.get(key)
    if cached and ttl and now - cached[0] < ttl:
        _ROUTE_CACHE.move_to_end(key)
        decision = cached[1]
        logger.info("路由仲裁缓存命中 key=%s", key[:8])
        return decision.route, 1.0, decision.reason_text()
    if cached:
        _ROUTE_CACHE.pop(key, None)

    try:
        # 注意：提示词里含有 JSON 花括号示例，不能用 str.format，
        # 否则 {"route": ...} 会被当成占位符（KeyError）。
        response = _route_llm().invoke(_ROUTE_LLM_PROMPT + str(question))
        route, confidence, reason = _parse_route_payload(
            str(getattr(response, "content", "") or "")
        )
    except Exception:
        logger.warning("路由仲裁 LLM 调用失败,回退保守策略", exc_info=True)
        return None, 0.0, ""

    if route is None:
        return None, 0.0, reason
    if ttl:
        _ROUTE_CACHE[key] = (
            now,
            RouteDecision(route, "llm", 0, (reason or route,)),
        )
        _ROUTE_CACHE.move_to_end(key)
        while len(_ROUTE_CACHE) > _ROUTE_CACHE_MAX:
            _ROUTE_CACHE.popitem(last=False)
    return route, confidence, reason


def _decide_route(question: str) -> RouteDecision:
    """开局路由唯一入口:硬规则 → 打分 → (模糊带) LLM 仲裁。"""
    settings = get_settings()
    mode = settings.router_mode if settings.router_mode in _ROUTE_MODES else "auto"

    if mode == "always_planner":
        return RouteDecision("planner", "forced", 99, ("ROUTER_MODE=always_planner",))
    if mode == "always_chatbot":
        return RouteDecision("chatbot", "forced", -1, ("ROUTER_MODE=always_chatbot",))

    features = _route_features(question)
    if not features.question:
        return RouteDecision("chatbot", "hard_rule", 0, ("空问题",))
    if features.is_paper:
        return RouteDecision("chatbot", "hard_rule", 0, ("论文生成模式",))
    if features.is_greeting:
        return RouteDecision("chatbot", "hard_rule", 0, ("寒暄",))
    if features.explicit_direct:
        return RouteDecision("chatbot", "hard_rule", 0, ("用户要求直接作答",))
    if features.explicit_plan:
        return RouteDecision("planner", "hard_rule", 99, ("用户显式要求分步",))

    score, reasons = _score_route(features)

    if mode == "llm":
        # 强制语义仲裁:规则仅作为兜底
        route, confidence, reason = _classify_route_with_llm(features.question)
        if route and confidence >= _ROUTE_LLM_MIN_CONFIDENCE:
            return RouteDecision(
                route, "llm", score, reasons + (f"LLM:{reason or route}",)
            )
        fallback = "planner" if score >= settings.router_score_plan else "chatbot"
        return RouteDecision(
            fallback,
            "llm_fallback",
            score,
            reasons + ("LLM 不可用,按分数兜底",),
        )

    if score >= settings.router_score_plan:
        return RouteDecision("planner", "score", score, reasons)
    if score <= settings.router_score_chatbot:
        return RouteDecision("chatbot", "score", score, reasons or ("无多步信号",))

    # 模糊带
    if mode == "rules" or not settings.router_llm_enabled:
        return RouteDecision(
            "chatbot",
            "score",
            score,
            reasons + (f"分数 {score} 落在模糊带,保守选 chatbot",),
        )

    route, confidence, reason = _classify_route_with_llm(features.question)
    if route and confidence >= _ROUTE_LLM_MIN_CONFIDENCE:
        return RouteDecision(
            route, "llm", score, reasons + (f"LLM:{reason or route}",)
        )
    return RouteDecision(
        "chatbot",
        "llm_fallback",
        score,
        reasons + ("LLM 判定不可用或置信度不足,保守选 chatbot",),
    )
