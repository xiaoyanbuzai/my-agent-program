"""LLM 封装：DeepSeek reasoning_content 兼容、提示词缓存、消息修补、长期记忆。"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import logging
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from agent.prompts import _turn_context

logger = logging.getLogger("web_app")


def _message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "type", ""))


def _message_text(message) -> str:
    content = message.get("content", "") if isinstance(message, dict) else getattr(
        message, "content", ""
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _prompt_messages(
    messages,
    memory_text: str,
    *,
    paper_mode: bool,
    history_turns: int = 2,
    history_chars_per_turn: int = 1000,
) -> list:
    """保留最近两轮文本上下文，丢弃旧工具往返，并把动态上下文放到稳定前缀之后。"""
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if _message_role(message) in {"human", "user"}
    ]
    if not user_indexes:
        return list(messages)

    current_user_index = user_indexes[-1]
    history_start = max(0, len(user_indexes) - history_turns - 1)
    compact_history = []
    for user_position in range(len(user_indexes) - 2, history_start - 1, -1):
        start = user_indexes[user_position]
        end = user_indexes[user_position + 1]
        turn = messages[start:end]
        final_answer = next(
            (
                item
                for item in reversed(turn)
                if _message_role(item) in {"ai", "assistant"}
                and not getattr(item, "tool_calls", None)
            ),
            None,
        )
        if final_answer is None:
            continue

        question_text = _message_text(messages[start]).strip()
        answer_text = _message_text(final_answer).strip()
        if not question_text or not answer_text:
            continue
        question_budget = history_chars_per_turn // 3
        answer_budget = history_chars_per_turn - question_budget
        compact_history.append(
            (
                HumanMessage(content=question_text[:question_budget]),
                AIMessage(content=answer_text[:answer_budget]),
            )
        )

    compact_history.reverse()
    current_turn = list(messages[current_user_index:])
    context = _turn_context(memory_text, paper_mode=paper_mode)
    if context:
        current_message = current_turn[0]
        content = (
            current_message.get("content", "")
            if isinstance(current_message, dict)
            else getattr(current_message, "content", "")
        )
        if isinstance(content, str):
            updated_content = f"{context}\n\n【用户问题】\n{content}"
        elif isinstance(content, list):
            updated_content = [
                {"type": "text", "text": f"{context}\n\n【用户问题】"},
                *content,
            ]
        else:
            updated_content = f"{context}\n\n【用户问题】\n{content}"
        current_turn[0] = (
            {**current_message, "content": updated_content}
            if isinstance(current_message, dict)
            else current_message.model_copy(update={"content": updated_content})
        )

    return _repair_tool_pairs(
        [message for pair in compact_history for message in pair] + current_turn
    )


def _repair_tool_pairs(messages: list) -> list:
    """保证每个带 tool_calls 的 assistant 消息都紧跟齐全的 tool 消息。

    OpenAI 兼容端点要求 tool_calls 必须被紧随其后的 tool 消息逐条应答，否则
    直接 400（insufficient tool messages following tool_calls message）。历史里
    出现下面三种情况都会踩雷，这里统一兜底：
    1. 工具崩溃 / 请求中断导致某条 tool 消息缺失   → 补占位消息；
    2. 模型的 tool_call 没有 id（空串或 None）      → 补一个稳定的 id；
    3. 孤立的 tool 消息（前面没有对应 tool_calls） → 丢弃。
    """
    repaired: list = []
    index = 0
    total = len(messages)

    while index < total:
        message = messages[index]
        index += 1

        if not isinstance(message, AIMessage) or not getattr(message, "tool_calls", None):
            if isinstance(message, ToolMessage):
                logger.warning(
                    "丢弃孤立的工具消息: tool_call_id=%s", message.tool_call_id
                )
                continue
            repaired.append(message)
            continue

        normalized_calls = []
        original_ids: list[str] = []
        missing_id = False
        for position, call in enumerate(message.tool_calls):
            raw_id = call.get("id")
            original_ids.append(str(raw_id) if raw_id else "")
            call = dict(call)
            if not call.get("id"):
                call["id"] = f"call_repaired_{len(repaired)}_{position}"
                missing_id = True
            normalized_calls.append(call)
        if missing_id:
            logger.warning("补全缺失的 tool_call id")
            message = message.model_copy(update={"tool_calls": normalized_calls})
        repaired.append(message)

        collected: dict[str, ToolMessage] = {}
        while index < total and isinstance(messages[index], ToolMessage):
            tool_message = messages[index]
            index += 1
            key = str(tool_message.tool_call_id or "")
            collected.setdefault(key, tool_message)

        for call, original_id in zip(normalized_calls, original_ids):
            call_id = call["id"]
            tool_message = collected.pop(call_id, None)
            if tool_message is None and original_id != call_id:
                # 模型没给 id 时，历史里的工具消息 keys 是空串，按顺序认领
                tool_message = collected.pop(original_id, None)
            if tool_message is None:
                logger.warning("补齐缺失的工具结果: tool_call_id=%s", call_id)
                tool_message = ToolMessage(
                    content="（该工具没有返回结果，可能执行失败或已中断）",
                    tool_call_id=call_id,
                )
            elif str(tool_message.tool_call_id or "") != call_id:
                tool_message = ToolMessage(
                    content=tool_message.content, tool_call_id=call_id
                )
            repaired.append(tool_message)

        for orphan_id in collected:
            logger.warning("丢弃没有对应 tool_call 的工具消息: %s", orphan_id)

    return repaired


def _strip_tool_history(messages: list) -> list:
    """去掉所有带 tool_calls 的助手消息与工具消息，只留纯文本对话。

    上游偶发「insufficient tool messages」400：把证据文本直接并进最后一条
    用户消息，避免整轮问答 503。
    """
    kept: list = []
    evidence: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            text = _message_text(message).strip()
            if text:
                evidence.append(text)
            continue
        if getattr(message, "tool_calls", None):
            continue
        kept.append(message)
    if not kept:
        kept = [HumanMessage(content="")]
    if evidence:
        last = kept[-1]
        addition = "\n\n【已检索到的证据】\n" + "\n\n".join(evidence)[:4000]
        content = _message_text(last)
        if isinstance(last, dict):
            kept[-1] = {**last, "content": content + addition}
        else:
            kept[-1] = last.model_copy(update={"content": content + addition})
    return kept


def _is_tool_pair_error(exc: BaseException) -> bool:
    """识别上游因 tool_calls 与 tool 消息不配对而返回的 400。"""
    text = str(exc)
    return "tool_calls" in text and "tool message" in text


def _prompt_cache_model_kwargs(ttl: str) -> dict:
    """只对 Moonshot/Kimi endpoint 发送服务商支持的隐式缓存参数。"""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or ""
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname not in {"api.moonshot.cn", "api.moonshot.ai"}:
        return {}
    return {"prompt_cache_options": {"mode": "implicit", "ttl": ttl}}


def _log_prompt_usage(response, user_id: int) -> None:
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    provider_usage = response_metadata.get("token_usage") or response_metadata.get(
        "usage", {}
    )
    token_details = provider_usage.get("prompt_tokens_details") or {}
    input_details = usage_metadata.get("input_token_details") or {}
    input_tokens = usage_metadata.get("input_tokens") or provider_usage.get(
        "prompt_tokens"
    )
    cache_hit_tokens = (
        token_details.get("cached_tokens")
        or input_details.get("cached_tokens")
        or provider_usage.get("prompt_cache_hit_tokens")
    )
    cache_write_tokens = token_details.get("cache_write_tokens") or provider_usage.get(
        "prompt_cache_write_tokens"
    )
    cache_miss_tokens = provider_usage.get("prompt_cache_miss_tokens")
    if cache_miss_tokens is None and input_tokens is not None:
        cache_miss_tokens = max(
            0, input_tokens - (cache_hit_tokens or 0) - (cache_write_tokens or 0)
        )
    if input_tokens is None and cache_hit_tokens is None and cache_write_tokens is None:
        logger.debug("LLM usage 不可用(user_id=%s)", user_id)
        return
    logger.info(
        "LLM prompt usage user=%s input_tokens=%s prompt_cache_hit_tokens=%s "
        "prompt_cache_miss_tokens=%s prompt_cache_write_tokens=%s",
        user_id,
        input_tokens,
        cache_hit_tokens,
        cache_miss_tokens,
        cache_write_tokens,
    )


def _is_deepseek_endpoint() -> bool:
    """当前 OpenAI 兼容端点是否 DeepSeek（其 thinking 模式要求回传 reasoning_content）。"""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or ""
    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "api.deepseek.com"


def _is_deepseek_reasoning_endpoint() -> bool:
    """DeepSeek 的 reasoner / thinking 模型会返回 reasoning_content。"""
    return _is_deepseek_endpoint()


class _ReasoningContentChatOpenAI(ChatOpenAI):
    """在 DeepSeek thinking 模式下完整保留 reasoning_content。

    DeepSeek 要求把上一轮 assistant 消息里的 ``reasoning_content`` 原样回传，
    否则返回 400。langchain-openai 在流式路径不会提取该字段，因此这里同时
    处理流式与非流式两种响应形态。
    """

    def _effective_base_url(self) -> str:
        """取实际生效的 API 地址。

        新版 langchain-openai 不再从环境变量填充 ``openai_api_base``，真实地址
        保存在客户端的 ``base_url`` 上，因此这里两处都看。
        """
        base = self.openai_api_base
        if not base:
            root_client = getattr(self, "root_client", None)
            base = str(getattr(root_client, "base_url", "") or "")
        return base or (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "")

    def _uses_deepseek_reasoning(self) -> bool:
        hostname = (urlparse(self._effective_base_url()).hostname or "").lower()
        return hostname == "api.deepseek.com"

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        if not self._uses_deepseek_reasoning():
            return result

        choices = (
            response.get("choices", [])
            if isinstance(response, dict)
            else getattr(response, "choices", [])
        )
        for generation, choice in zip(result.generations, choices or []):
            response_message = (
                choice.get("message")
                if isinstance(choice, dict)
                else getattr(choice, "message", None)
            )
            if isinstance(response_message, dict):
                reasoning_content = response_message.get("reasoning_content")
            else:
                reasoning_content = getattr(
                    response_message, "reasoning_content", None
                )
            if isinstance(generation.message, AIMessage) and reasoning_content:
                generation.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: Any, base_generation_info: dict | None
    ) -> Any:
        """流式 delta 里的 reasoning_content 也一并带进 chunk。"""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or not self._uses_deepseek_reasoning():
            return generation_chunk

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if not choices:
            return generation_chunk
        delta = choices[0].get("delta") or {}
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs["reasoning_content"] = (
                reasoning_content
            )
        return generation_chunk

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if not self._uses_deepseek_reasoning():
            # 非 DeepSeek 端点不认识该字段，一律剔除，避免触发 400。
            for msg in payload.get("messages", []):
                msg.pop("reasoning_content", None)
            return payload

        input_messages = convert_to_messages(input_)
        for source, target in zip(input_messages, payload.get("messages", [])):
            if not isinstance(source, AIMessage):
                continue
            # DeepSeek thinking 模式要求：只要历史里出现过 assistant 消息，
            # 该消息就必须带 reasoning_content（无论是否含 tool_calls），否则 400。
            # 因此:有原始思考摘要就回传原文，没有就补空串（实测空串可接受）。
            reasoning_content = source.additional_kwargs.get("reasoning_content")
            if reasoning_content:
                target["reasoning_content"] = reasoning_content
            else:
                target.setdefault("reasoning_content", "")
        return payload


class _SafeChatOpenAI(ChatOpenAI):
    """普通（非 DeepSeek）端点使用：剔除历史脏数据里的 reasoning_content 字段。"""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for msg in payload.get("messages", []):
            msg.pop("reasoning_content", None)
        return payload


def _memory_tokens(text: str) -> set[str]:
    """粗粒度词项：英文单词 + 中文字符 + 中文二元组，用于本地记忆相关性排序。"""
    lowered = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.update(chinese)
    tokens.update(a + b for a, b in zip(chinese, chinese[1:]))
    return tokens


def _memory_relevance(text: str, question: str) -> float:
    query_tokens = _memory_tokens(question)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _memory_tokens(text)) / len(query_tokens)


def _load_memory_text(memory_store, user_id: int, question: str, limit: int) -> str:
    """SqliteStore 未配置向量索引时按词项重叠排序，避免“永远取最近 N 条”。"""
    if memory_store is None or not question or limit <= 0:
        return ""
    items = memory_store.search(
        (str(user_id), "memories"), limit=max(limit * 5, 20)
    )
    scored = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        text = str(value.get("text", "")).strip()
        if not text:
            continue
        scored.append((_memory_relevance(text, question), text))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [text[:300] for score, text in scored[:limit] if score > 0]
    return "\n".join(selected)
