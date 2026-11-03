"""计划解析与工具调用封装（超时、结果摘要）。"""

from __future__ import annotations

import asyncio
import json
import logging
import re


logger = logging.getLogger("web_app")


_MAX_PLAN_STEPS = 4


def _parse_plan_steps(text: str, max_steps: int = _MAX_PLAN_STEPS) -> list[str]:
    """从大模型输出中稳健地抽取执行步骤。

    模型可能返回 ```json 代码块、带前后说明的数组、编号列表，甚至被截断的
    JSON。这里逐级降级，任何一级成功即返回；全部失败返回空列表，由调用方
    决定兜底策略（绝不能让空 plan 进入 executor）。
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    def _normalize(items) -> list[str]:
        steps: list[str] = []
        for item in items:
            if isinstance(item, dict):
                # 兼容 {"step": "...", "task": "..."} 这类结构
                item = item.get("step") or item.get("task") or item.get("action") or ""
            step = " ".join(str(item).split()).strip(" -—•·\t")
            if step:
                steps.append(step)
        return steps[:max_steps]

    # 1) 去掉 Markdown 代码围栏
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fenced:
        raw = fenced.group(1).strip()

    # 2) 直接解析整段，或首个 '[' 到末个 ']' 之间的片段
    candidates = [raw]
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        candidates.insert(0, raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            steps = _normalize(parsed)
            if steps:
                return steps
        elif isinstance(parsed, str):
            steps = _normalize([parsed])
            if steps:
                return steps

    # 3) 逐行解析：优先编号/引号项目，其次非空短行
    numbered = re.findall(
        r"^\s*(?:\d+[.、)]|[-*•])\s*(.+)$", raw, re.M
    )
    steps = _normalize(numbered)
    if steps:
        return steps

    quoted = re.findall(r"[\"“](.+?)[\"”]", raw)
    steps = _normalize(quoted)
    if steps:
        return steps

    # 4) 最后尝试：把整段当作单步（去掉代码块残留与多余空白）
    fallback = " ".join(
        line.strip(" `*#-•\t") for line in raw.splitlines() if line.strip()
    )
    # 只有像“任务描述”的文本才接受：至少 6 个字符且含文字/数字，
    # 避免把 ">"、"*"、"[ " 这类残片当成一个执行步骤。
    if len(fallback) >= 6 and re.search(r"[\w\u4e00-\u9fff]", fallback):
        steps = _normalize([fallback])
        if steps:
            return steps
    return []


_REFUSAL_PATTERN = re.compile(
    r"^(抱歉|对不起|很抱歉|无法|不能|sorry|i\s*(?:can\s*not|can't)|unable)"
)


def _looks_like_refusal(text: str) -> bool:
    """判断模型输出是否只是拒答/致歉，而不是真正的执行计划。"""
    normalized = " ".join(str(text or "").split()).strip()
    return bool(normalized) and bool(_REFUSAL_PATTERN.match(normalized.lower()))


def _plan_fallback_steps(question: str) -> list[str]:
    """规划解析失败时的兜底：把用户问题本身作为唯一执行步骤。

    返回非空列表，确保 executor 永远有任务可执行；若有“先…再…”这类明确
    顺序，则拆成两条，尽量保留多阶段意图。
    """
    question = " ".join(str(question or "").split())
    if not question:
        return []
    parts = re.split(r"(?:先|首先).{0,80}?(?:再|然后|接着|最后)", question)
    if len(parts) >= 2:
        cleaned = [p.strip(" ，,。;；") for p in parts if p.strip(" ，,。;；")]
        if len(cleaned) >= 2:
            return cleaned[:_MAX_PLAN_STEPS]
    return [question]


def _filter_plan_steps(steps: list[str]) -> list[str]:
    """过滤掉空转步骤、重复步骤和拒答句，保留真正可执行的检索/整理动作。"""
    kept: list[str] = []
    seen: set[str] = set()
    for raw_step in steps or []:
        text = " ".join(str(raw_step or "").split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        if _looks_like_refusal(text):
            continue
        # “分析用户需求/明确目标”这类短步骤不会产生任何证据，直接丢弃；
        # 但含检索动作的短步骤（如“检索 TP53 文献”）必须保留。
        if (
            len(text) <= 20
            and any(term in text for term in ("需求", "意图", "计划", "思路", "目标"))
            and not any(
                term in text
                for term in ("检索", "查询", "搜索", "查", "搜", "下载", "生成", "汇总")
            )
        ):
            continue
        seen.add(key)
        kept.append(text)
    return kept


async def _ainvoke_with_timeout(fn, arguments: dict, timeout: float):
    """异步工具带超时调用;超时抛 asyncio.TimeoutError,由调用方转成缺口。"""
    return await asyncio.wait_for(fn.ainvoke(arguments), timeout=timeout)


def _invoke_tool_with_timeout(
    fn, arguments: dict, timeout: float
) -> tuple[str, bool, str]:
    """统一调用同步/异步工具。

    返回 ``(结果文本, 是否成功, 错误信息)``。异步工具（MCP 的 StructuredTool
    只实现了 ainvoke）走 ``asyncio.run`` + ``wait_for`` 超时保护；同步工具无法
    安全中断，保持原有直接调用，超时交给上游 LLM/HTTP 超时兜底。
    """
    try:
        if getattr(fn, "coroutine", None) is not None:
            result = asyncio.run(_ainvoke_with_timeout(fn, arguments, timeout))
        elif hasattr(fn, "invoke"):
            result = fn.invoke(arguments)
        else:
            result = fn(**arguments)
        return str(result), True, ""
    except Exception as exc:  # noqa: BLE001 - 任何工具异常都只降级为“缺口”
        return "", False, f"{type(exc).__name__}: {exc}"


def _summarize_tool_result(llm, task: str, raw: str, max_chars: int = 3000) -> str:
    """把工具输出压缩成核心事实;"""
    prompt = f"""你刚才为了完成【{task}】进行了检索。
    工具返回的原始结果是：
    {str(raw)[:10000]}

    请提取出对回答有用的【所有核心事实、数据和观点】。
    如果有表格，请保留 Markdown 表格格式。不要写废话。
    """
    try:
        content = llm.invoke(prompt).content
        text = " ".join(str(content or "").split())
        if text:
            return text
    except Exception:
        logger.warning("工具结果摘要失败,改用原文摘录", exc_info=True)
    excerpt = " ".join(str(raw or "").split())[:max_chars]
    return excerpt or "（工具未返回内容）"
