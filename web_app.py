"""FastAPI 应用：多租户学术文献知识库 Agent。

分层约定：
- 路由 / 鉴权 / Agent 编排在本文件；
- 解析与入库在 kb_loader / indexer;
- 混合检索在 kb_retriever;
- 多模态在 multimodal;
- 联网与安全抓取在 search_online;
- 配置在 config。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from contextlib import ExitStack, asynccontextmanager
from typing import Annotated, cast
import numpy as np
from vector_store import get_embeddings

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_config, get_store
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from typing_extensions import TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient

from auth import (
    AuthRequest,
    create_token,
    decode_token,
    get_user_id,
    login_user,
    register_user,
)
from config import ensure_directories, get_settings, user_dir
from conversations import (
    ConversationNotFound,
    append_message,
    conversation_to_dict,
    create_conversation as create_conversation_record,
    delete_conversation as delete_conversation_record,
    get_conversation,
    list_conversations as list_conversation_records,
    list_messages as list_conversation_messages,
    thread_key,
)
from db import SessionLocal, get_db, init_db
from documents import delete_document as remove_document
from documents import list_documents
from indexer import build_index
from kb_loader import (
    TEXT_EXTENSIONS,
    DocumentParseError,
    load_document,
)
from conversations import delete_messages_after
from kb_retriever import count_documents, invalidate_cache
from knowledge import analyze_document
from logging_utils import setup_logging
from models import User
from multimodal import MultimodalError, describe_image_bytes
from search_online import (
    FetchError,
)
from uploads import (
    UploadValidationError,
    decode_image_payload,
    sanitize_filename,
    save_stream_limited,
    unique_path,
    validate_extension,
)
from vector_store import EmbeddingModelMismatch, get_embedding_model_name

from agent.answer_style import (
    _remove_repeated_answer,
    _strip_answer_filler,
    scrub_tool_leakage,
)
from agent.builtin_tools import _learn_url
from agent.common import _configurable, _current_user_id, _tool_name
from agent.enrichment import (
    _detail_tool_arguments,
    _enrich_tool_output,
    _merge_evidence_excerpts,
)
from agent.intent import _classify_intent
from agent.llm import (
    _ReasoningContentChatOpenAI,
    _SafeChatOpenAI,
    _is_deepseek_reasoning_endpoint,
    _load_memory_text,
    _log_prompt_usage,
    _is_tool_pair_error,
    _prompt_cache_model_kwargs,
    _prompt_messages,
    _strip_tool_history,
)
from agent.planning import (
    _MAX_PLAN_STEPS,
    _filter_plan_steps,
    _invoke_tool_with_timeout,
    _looks_like_refusal,
    _parse_plan_steps,
    _plan_fallback_steps,
    _summarize_tool_result,
)
from agent.prompts import _replanner_prompt, _system_prompt
from agent.routing import _decide_route
from agent.signals import _is_paper_request
from agent.tool_rag import (
    _INTENT_REQUIRED_TOOLS,
    _MAX_SELECTED_TOOLS,
    _agent_tools,
    _compact_tool_schema,
    _rank_tools_with_reasons,
    _select_tools,
    _static_priority_key,
    _tool_affinity,
    _tool_sources,
)

# --- 兼容导出 ---------------------------------------------------------------

from agent.builtin_tools import _compress_retrieval_content  # noqa: F401
from agent.enrichment import (  # noqa: F401
    _extract_pmids,
    _html_tables_to_markdown,
    _needs_abstract_enrichment,
)
from agent.llm import _repair_tool_pairs  # noqa: F401
# from agent.prompts import _ANSWER_CONTRACT  # noqa: F401
from agent.routing import (  # noqa: F401
    _classify_route_with_llm,
    _needs_planning,
    _parse_route_payload,
    _route_features,
    _score_route,
)
from agent.sources import (  # noqa: F401
    _MODE_DIRECT,
    _MODE_RETRIEVAL,
    _SOURCE_ARXIV,
    _SOURCE_DOCX,
    _SOURCE_GENE,
    _SOURCE_HERMES,
    _SOURCE_KB,
    _SOURCE_PUBMED,
    _SOURCE_URL,
)
from agent.tool_rag import (  # noqa: F401
    _MAX_ENTITY_TOOLS,
    _cap_selected_tools,
    _rank_tool_names,
)

logger = logging.getLogger(__name__)

# 模型只输出工具调用语法、清洗后为空时的兜底回答
EMPTY_ANSWER_FALLBACK = "本轮没有取得可用证据，请缩小问题范围或换个问法重试。"

MB = 1024 * 1024
_THREAD_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    ensure_directories()
    init_db()

    # 1. 初始化 MCP Client（逐个服务器独立加载，单个失败不影响其余）
    mcp_servers = {
        "hermes": {
            "command": "hermes",
            "args": ["mcp", "serve"],
            "transport": "stdio",
        },
        # 📚 语义化学术检索（AI + 生信通用，覆盖 2 亿+ 论文）
        "semantic_scholar": {
            "command": "uvx",
            "args": ["s2-mcp-server"],
            "transport": "stdio",
        },
        # 👇 PubMed 生物医学检索（41 个工具：检索/基因/化合物/全文/引用网络等）
        "pubmed": {
            "command": "uvx",
            "args": ["pubmed-search-mcp"],
            "transport": "stdio",
            "env": {
                "NCBI_EMAIL": os.getenv("NCBI_EMAIL", "default@example.com"),
            },
        },
    }
    mcp_client = MultiServerMCPClient(mcp_servers)

    # 2. 逐个服务器加载工具，互不拖累
    extra_tools = []
    for server_name in mcp_servers:
        try:
            server_tools = await mcp_client.get_tools(server_name=server_name)
            extra_tools.extend(server_tools)
            logger.info("成功加载 MCP 服务器 %s：%d 个工具", server_name, len(server_tools))
        except Exception as exc:
            logger.warning("无法连接 MCP 服务器 %s：%s", server_name, exc)

    with ExitStack() as stack:
        checkpointer = stack.enter_context(
            SqliteSaver.from_conn_string(str(settings.data_dir / "conversations.db"))
        )
        store = stack.enter_context(
            SqliteStore.from_conn_string(str(settings.data_dir / "memory.db"))
        )
        app.state.graph = build_graph(checkpointer=checkpointer, store=store, extra_tools=extra_tools)
        app.state.checkpointer = checkpointer
        logger.info(
            "应用启动:env=%s embedding=%s llm=%s",
            settings.app_env,
            settings.embedding_model,
            settings.llm_model,
        )
        yield

    logger.info("应用已关闭")


app = FastAPI(title="文献知识库 Agent", lifespan=lifespan)
templates = Jinja2Templates(directory=str(get_settings().templates_dir))


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


class State(TypedDict):
    messages: Annotated[list, add_messages]
    reflection_count: int
    critique: str
    route: str
    route_reason: str
    plan: list[str]
    # 注意:past_steps / evidence / evidence_gaps 不用 operator.add，
    # 而是由节点返回“累积后的完整列表”。这样 router 每轮可以真正清零，
    # 避免上一轮的执行记录污染本轮判断。
    past_steps: list[tuple[str, str]]
    evidence: list[dict]
    evidence_gaps: list[str]
    # 本轮已发生的“模型→工具”往返次数，用于收束重复检索
    tool_rounds: int
    enable_web_search: bool


class AskRequest(BaseModel):
    question: str = ""
    thread_id: str = "web-user"
    conversation_id: int | None = None
    image_base64: str | None = None
    edit_message_id: int | None = None
    enable_web_search: bool = False 


class LearnRequest(BaseModel):
    url: str
    title: str = ""


def _last_user_message(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return str(message.content or "")
    return ""


def router_node(state: State):
    """开局路由：唯一决策点，决定本轮走 planner 还是 chatbot。

    判定逻辑全部收敛到 _decide_route(硬规则 → 打分 → 模糊带 LLM 仲裁)，
    这里只负责调用、记录依据，并清零上一轮的执行态，避免跨轮污染。
    """
    user_question = _last_user_message(state["messages"])
    enable_web = state.get("enable_web_search", False)
    if not enable_web:
        # 如果没开联网，强制走 chatbot，并且后续只能查本地库或纯闲聊
        logger.info("用户未开启联网，强制走本地/通用直答模式")
        return {
            "route": "chatbot",
            "route_reason": "未开启联网，限定本地库",
            "plan": [],
            "past_steps": [],
            "evidence": [],
            "evidence_gaps": [],
            "critique": "",
            "reflection_count": 0,
            "tool_rounds": 0,
        }

    decision = _decide_route(user_question)
    logger.info(
        "路由决策 user_id=%s route=%s source=%s score=%d reasons=%s question=%s",
        _current_user_id(),
        decision.route,
        decision.source,
        decision.score,
        decision.reason_text(),
        user_question[:80],
    )
    return {
        "route": decision.route,
        "route_reason": decision.reason_text(),
        "plan": [],
        "past_steps": [],
        "evidence": [],
        "evidence_gaps": [],
        "critique": "",
        "reflection_count": 0,
        "tool_rounds": 0,
    }


def route_from_router(state: State):
    if state.get("route") == "planner":
        return "planner"
    return "chatbot"


def should_continue(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "reflection"


def route_after_reflection(state: State):
    """审稿人节点执行完后，决定下一步去哪。"""
    critique = state.get("critique", "")
    count = state.get("reflection_count", 0)

    # 1. 如果通过了，或者超过了最大重试次数（比如2次），强制结束
    if critique == "PASS" or count >= 1:
        return END
    # 2. 如果没通过，打回给 chatbot 重新生成
    return "chatbot"


def route_after_executor(state:State):
    if state.get("plan"):
        return "executor"
    if not state.get("past_steps"):
        # 一步都没执行成功（例如 executor 的 LLM 调用失败），
        # 交给 chatbot 正常检索作答，避免 replanner 无证据硬写报告。
        return "chatbot"
    return "replanner"


def route_after_planner(state: State):
    """空计划直接交回 chatbot，避免 executor 索引空列表越界。"""
    return "executor" if state.get("plan") else "chatbot"


def _build_tool_retriever(tool_map, embed_model, tool_vectors):
    """返回「意图来源 → 余弦相似度 → 专有意图分」三层打分的选择器。"""

    def _retrieve_relevant_tools(query: str, top_k: int = 5) -> list:
        """按「意图来源 → 语义相似度 → 专有意图分」动态挑选 top_k 个工具。

        只用 embedding 相似度时，“PMID 42260443 的摘要”会被泛化的检索工具挤掉
        fetch_article_details。这里叠三层信号：
        1. 意图层判定的来源族作为白名单，跨来源工具只做兜底；
        2. 余弦相似度决定同来源内的先后；
        3. _tool_affinity 的专有意图分(实体/线索)再抬高专用工具。
        意图激活后必须用的工具(search_gene / fetch_article_details 等)强制保留。
        """
        top_k = max(1, min(top_k, _MAX_SELECTED_TOOLS))
        if not query:
            return list(tool_map.values())[:top_k]

        intent = _classify_intent(query)
        active = set(intent.sources)
        allowed = {
            name for name in tool_map if active and (_tool_sources(name) & active)
        }
        required = {
            name
            for source in intent.sources
            for name in _INTENT_REQUIRED_TOOLS.get(source, ())
            if name in tool_map
        }
        kinds = intent.entity_kinds
        if kinds & {"pmid", "doi", "nct"}:
            required |= {"fetch_article_details", "get_fulltext"} & set(tool_map)

        query_vec = embed_model.embed_query(query)
        query_norm = np.linalg.norm(query_vec)

        scores = []
        for name, vec in tool_vectors.items():
            vec_norm = np.linalg.norm(vec)
            cosine = (
                float(np.dot(query_vec, vec) / (query_norm * vec_norm))
                if query_norm and vec_norm
                else 0.0
            )
            affinity, _reasons = _tool_affinity(query, name, intent)
            blended = cosine + 0.35 * min(1.0, affinity / 8.0)
            # 第一档：来源族内；第二档：族外兜底
            in_scope = 0 if (not allowed or name in allowed) else 1
            scores.append((in_scope, -blended, _static_priority_key(name), name))

        scores.sort()
        selected_names = [item[3] for item in scores[:top_k]]

        # 必需工具保底：被语义检索挤掉时强制换上
        for name in required:
            if name in selected_names:
                continue
            if len(selected_names) >= top_k:
                selected_names = selected_names[: max(0, top_k - 1)]
            selected_names.append(name)

        # 保底逻辑：如果用户明确说了“生成论文”，强行把生成工具加进去
        if _is_paper_request(query) and "generate_paper_docx_tool" not in selected_names:
            if len(selected_names) >= top_k:
                selected_names = selected_names[: top_k - 1]
            selected_names.append("generate_paper_docx_tool")

        logger.info(
            "Tool RAG user=%s intent=%s top=%s",
            _current_user_id(),
            intent.source_text(),
            selected_names,
        )
        return [tool_map[name] for name in selected_names if name in tool_map]

    return _retrieve_relevant_tools


def _build_chatbot_node(*, llm, tools, settings, store):
    """对话节点：选工具、控轮次、读长期记忆、写回记忆。"""

    def chatbot(state: State):
        config = get_config()
        configurable = _configurable(config)
        user_id = int(configurable.get("user_id", 0) or 0)
        memory_text = ""
        last_question = _last_user_message(state["messages"])
        memory_store = None

        try:
            memory_store = store or get_store()
            memory_text = _load_memory_text(
                memory_store, user_id, last_question, settings.memory_limit
            )
        except Exception:
            logger.exception("读取长期记忆失败(user_id=%s)", user_id)

        # 获取反思意见（只消费一次，长度截断避免把上一条长意见整段塞进系统提示）
        critique = str(state.get("critique", "") or "").strip()
        critique_prompt = ""
        if critique and critique != "PASS":
            critique_prompt = (
                "\n\n【极其重要:审稿人意见】\n"
                "你上一次的回答不合格。请必须根据以下意见重写回答:\n"
                f"{critique[:500]}"
            )

        enable_web = state.get("enable_web_search",False)
        intent = _classify_intent(last_question)

        if not enable_web :
            if intent.needs_retrieval and "本地" in last_question:
                selected_tools = [t for t in tools if _tool_name(t) == "kb_search_tool"]
            else:
                selected_tools = []
        else:
            selected_tools = _select_tools(last_question, tools)

        selected_tools = _select_tools(last_question, tools)
        intent = _classify_intent(last_question)
        rounds_used = int(state.get("tool_rounds") or 0)
        tools_exhausted = rounds_used >= settings.max_tool_rounds
        if tools_exhausted and selected_tools:
            logger.info(
                "工具轮次已达上限 user=%s rounds=%s，本轮不再绑定工具",
                user_id,
                rounds_used,
            )
            selected_tools = []
            critique_prompt += (
                "\n\n【工具轮次已达上限】不要再检索，也不要输出任何工具调用语法、"
                "XML 标签或 JSON 参数；直接基于已有工具结果作答，缺数据就写“无数据”。"
            )
        logger.info(
            "Chatbot 意图 user=%s mode=%s sources=%s reasons=%s",
            user_id,
            intent.mode,
            intent.source_text(),
            intent.reason_text(),
        )
        # 👇 安全获取工具名，兼容函数和 Tool 对象
        tool_names = [getattr(t, "__name__", getattr(t, "name", "unknown")) for t in selected_tools]
        logger.info("Chatbot 动态加载工具 user=%s tools=%s", user_id, tool_names)
        if tool_names:
            # 把“为什么排在前面的工具优先”记下来，方便排查选错工具的问题
            logger.info(
                "工具动态优先级 user=%s top=%s",
                user_id,
                [
                    (name, round(score, 1), " + ".join(reasons) or "默认顺序")
                    for name, score, reasons in _rank_tools_with_reasons(
                        last_question, tool_names
                    )[:4]
                ],
            )
        llm_for_turn = (
            llm.bind_tools([_compact_tool_schema(tool) for tool in selected_tools])
            if selected_tools
            else llm
        )
        logger.debug(
            "本轮可用工具 user=%s tools=%s",
            user_id,
            [getattr(tool, "__name__", getattr(tool, "name", "")) for tool in tools],
        )
        is_direct_chat = not enable_web and not intent.needs_retrieval

        system = {"role": "system", 
                  "content": _system_prompt(direct=is_direct_chat) + critique_prompt}
        prompt_messages = _prompt_messages(
            state["messages"],
            memory_text,
            paper_mode=_is_paper_request(last_question),
        )
        try:
            response = llm_for_turn.invoke([system] + prompt_messages)
        except Exception as e:
            if _is_tool_pair_error(e):
                # 上游偶发 tool_calls 与 tool 消息不配对：退化成纯文本对话再试一次，
                # 把已拿到的证据直接并进最后一条消息，避免整轮 503。
                logger.warning(
                    "工具消息配对异常，降级为纯文本重试(user_id=%s)", user_id
                )
                try:
                    response = llm_for_turn.invoke(
                        [system] + _strip_tool_history(prompt_messages)
                    )
                except Exception:
                    logger.exception("LLM 调用失败(user_id=%s)", user_id)
                    raise HTTPException(
                        status_code=503,
                        detail="模型服务暂时不可用，请稍后重试",
                    ) from None
            else:
                logger.exception("LLM 调用失败(user_id=%s)", user_id)
                raise HTTPException(
                    status_code=503,
                    detail="模型服务暂时不可用，请稍后重试",
                ) from e
        _log_prompt_usage(response, user_id)

        try:
            if memory_store is not None and last_question and response.content:
                answer = str(response.content)
                key = hashlib.sha256(
                    f"{user_id}:{last_question}".encode("utf-8")
                ).hexdigest()[:32]
                memory_store.put(
                    (str(user_id), "memories"),
                    key=key,
                    value={
                        "text": f"用户问：{last_question[:200]}\n助手答：{answer[:300]}",
                        "updated_at": int(time.time()),
                    },
                )
        except Exception:
            logger.exception("写入长期记忆失败(user_id=%s)", user_id)

        # critique 是一次性意见：本轮已经据此重写，立即清空，
        # 否则下一轮追问会把同一条审稿意见再注入一次。
        return {"messages": [response], "critique": ""}

    return chatbot


def _build_planner_node(llm_judge):
    """规划节点：把需求拆成 2-4 个可执行步骤。"""

    def planner_node(state:State):
        """把用户需求拆成 2-4 个可执行步骤。

        路由决策在 router 节点已经做完了,这里**不再**用 _needs_planning 二次
        否决,否则 router 说 planner、planner 又返回空计划,会白跑一轮再回落
        chatbot(历史上就是这样丢掉规划的)。
        """
        user_question = _last_user_message(state["messages"])
        if not user_question.strip():
            logger.warning("planner 收到空问题,跳过规划(user_id=%s)", _current_user_id())
            return {"plan": [], "past_steps": []}

        prompt = f"""你是一个高级学术研究指挥官。请把用户需求拆解为 2-4 个按顺序执行的检索/整理步骤。
        要求：
        1. 每一步都是一个能直接调用工具完成的动作（例如“在 PubMed 检索 TP53 突变与癌症预后的最新综述”）；
        2. 不要输出“分析需求”“理解问题”“制定计划”这类空转步骤；
        3. 步骤之间不要重复，后一步可以使用前一步的结果；
        4. 只输出纯 JSON 数组，不要任何解释，例如 ["步骤1", "步骤2"]。
        可用能力:本地知识库检索、PubMed / arXiv 学术检索、联网搜索、网页抓取入库、生成 Word 文档。
        用户需求：{user_question}"""

        response = llm_judge.invoke(prompt)
        plan = _filter_plan_steps(_parse_plan_steps(str(response.content)))
        # 只有一步且内容是“抱歉/无法完成”这类拒答时，同样视为解析失败，
        # 否则 executor 会把一句道歉当成检索任务执行，白跑一轮。
        if len(plan) == 1 and _looks_like_refusal(plan[0]):
            plan = []
        if not plan:
            plan = _filter_plan_steps(_plan_fallback_steps(user_question)) or [
                user_question
            ]
            logger.warning(
                "计划解析为空，已降级为单步执行(user_id=%s, steps=%d)",
                _current_user_id(),
                len(plan),
            )
        if len(plan) > _MAX_PLAN_STEPS:
            plan = plan[:_MAX_PLAN_STEPS]

        logger.info(
            "规划完成 user_id=%s steps=%d plan=%s",
            _current_user_id(),
            len(plan),
            plan,
        )
        return {"plan" : plan,"past_steps": []}

    return planner_node


def _build_executor_node(*, llm, llm_judge, _retrieve_relevant_tools, tool_map, settings):
    """执行节点：只调用本轮 Tool RAG 选中的工具，并做数据丰富化。"""

    def executor_node(state:State):
        """执行计划中的当前任务，并在执行后提炼核心信息。

        三条硬性约束：
        1. 只允许调用本轮 Tool RAG 选出的工具，模型幻觉出来的工具名不执行；
        2. 工具失败/超时算“证据缺口”，绝不摘要成“核心事实”；
        3. 没有 tool_calls 时先重试一次，仍拿不到证据就明说缺口，不编造结论。
        """
        plan = list(state.get("plan") or [])
        past_steps = list(state.get("past_steps") or [])
        evidence = list(state.get("evidence") or [])
        gaps = list(state.get("evidence_gaps") or [])
        if not plan:
            # 空计划保护：绝不索引空列表，直接收束并交给 replanner。
            logger.warning("executor 收到空计划，跳过执行(user_id=%s)", _current_user_id())
            return {
                "plan": [],
                "past_steps": past_steps,
                "evidence": evidence,
                "evidence_gaps": gaps,
            }
        current_task = plan[0]

        # 新增：从配置中获取 user_id
        config = get_config()
        configurable = _configurable(config)
        user_id = int(configurable.get("user_id", 0) or 0)

        # 替换：使用 Tool RAG 动态匹配工具
        selected_tools = _retrieve_relevant_tools(current_task, top_k=4)
        # 安全获取工具名，兼容函数和 Tool 对象
        tool_names = [_tool_name(t) for t in selected_tools]
        allowed = {name for name in tool_names if name}
        logger.info(
            "Executor 加载工具 user=%s task=%s tools=%s",
            user_id,
            current_task[:60],
            tool_names,
        )

        llm_exec = llm.bind_tools([_compact_tool_schema(tool) for tool in selected_tools])
        history = "\n".join(
            f"已完成: {str(task)}\n结果: {str(result)[:200]}"
            for task, result in past_steps
        )
        base_prompt = f"""你是一个执行助理。请调用合适的工具完成当前任务。不要自己编造答案。
        当前任务：{current_task}
        历史执行记录：{history or "无"}"""

        response = None
        for attempt in range(2):
            prompt = base_prompt
            if attempt == 1:
                prompt += (
                    "\n\n注意：上一轮你没有调用任何工具。本轮必须调用工具获取真实资料；"
                    "如果拿不准用哪个，就调用联网搜索。"
                )
            try:
                response = llm_exec.invoke([{"role": "user", "content": prompt}])
            except Exception:
                logger.exception("Executor 节点调用 LLM 失败(user_id=%s)", user_id)
                response = None
                break
            if getattr(response, "tool_calls", None):
                break

        if response is None:
            gaps.append(f"{current_task}：执行模型不可用，未产生证据")
            return {
                "plan": [],
                "past_steps": past_steps,
                "evidence": evidence,
                "evidence_gaps": gaps,
            }

        if not getattr(response, "tool_calls", None):
            logger.warning(
                "executor 两次都未调用工具，本步记为缺口(user_id=%s, task=%s)",
                user_id,
                current_task[:60],
            )
            gaps.append(f"{current_task}：未获得检索证据")
            marker = "（本步未获得检索证据）"
            return {
                "plan": plan[1:],
                "past_steps": past_steps + [(str(current_task), marker)],
                "evidence": evidence,
                "evidence_gaps": gaps,
                "messages": [AIMessage(content=marker)],
            }

        messages_to_add = [response]  # 必须先把带 tool_calls 的 AIMessage 放进去
        summaries: list[str] = []
        enrich_budget = {"remaining": max(0, settings.enrich_abstract_limit)}
        for index, call in enumerate(response.tool_calls):
            fn_name = str(call.get("name", ""))
            call_id = call.get("id") or f"call_{index}"
            arguments = call.get("args") or {}

            if fn_name not in allowed:
                logger.warning(
                    "executor 拒绝未授权工具 user=%s tool=%s", user_id, fn_name
                )
                gaps.append(f"{current_task} / {fn_name}：工具本轮不可用")
                summaries.append(f"（{fn_name or '未知工具'} 本轮不可用，无有效证据）")
                messages_to_add.append(
                    ToolMessage(
                        content=f"错误：工具 {fn_name or '未知'} 本轮不可用",
                        tool_call_id=call_id,
                    )
                )
                continue

            logger.info(
                "工具调用 user=%s name=%s args=%s",
                user_id,
                fn_name,
                str(arguments)[:500],
            )
            res, ok, err = _invoke_tool_with_timeout(
                tool_map[fn_name], arguments, settings.tool_timeout
            )
            if not ok:
                gaps.append(f"{current_task} / {fn_name}：{err or '调用失败'}")
                logger.warning(
                    "工具调用失败 task=%s tool=%s err=%s",
                    current_task[:60],
                    fn_name,
                    err,
                )
                summaries.append(f"（{fn_name} 调用失败，无有效证据）")
                evidence.append(
                    {"task": str(current_task), "tool": fn_name, "excerpt": "", "ok": False}
                )
                messages_to_add.append(
                    ToolMessage(
                        content=f"工具调用失败：{err or '未知错误'}",
                        tool_call_id=call_id,
                    )
                )
                continue

            def _fetch_detail(pmids: list[str], _name: str = fn_name) -> str:
                """命中 PMID 但没有摘要时批量回查详情；预算用完就放弃。"""
                if enrich_budget["remaining"] <= 0 or not pmids:
                    return ""
                if _name == "fetch_article_details":
                    return ""
                fetcher = tool_map.get("fetch_article_details")
                if fetcher is None:
                    return ""
                enrich_budget["remaining"] -= 1
                detail, ok_detail, _err = _invoke_tool_with_timeout(
                    fetcher,
                    _detail_tool_arguments(fetcher, pmids),
                    min(settings.tool_timeout, 30.0),
                )
                return detail if ok_detail else ""

            enriched, enrich_notes = _enrich_tool_output(
                fn_name,
                res,
                fetch_detail=_fetch_detail,
                max_details=settings.enrich_abstract_limit,
                max_chars=settings.enrich_max_chars,
            )
            if enrich_notes:
                logger.info(
                    "数据丰富化 user=%s tool=%s notes=%s",
                    user_id,
                    fn_name,
                    list(enrich_notes),
                )
            summaries.append(_summarize_tool_result(llm_judge, current_task, enriched))
            evidence.append(
                {
                    "task": str(current_task),
                    "tool": fn_name,
                    "excerpt": enriched[:1200],
                    "ok": True,
                    "enriched": list(enrich_notes),
                }
            )
            messages_to_add.append(ToolMessage(content=enriched, tool_call_id=call_id))
        combined_summary = (
            "\n\n".join(item for item in summaries if item) or "（本步未获得检索证据）"
        )
        return {
            "past_steps": past_steps + [(str(current_task), combined_summary)],
            "plan": plan[1:],  # 移除当前任务
            "evidence": evidence,
            "evidence_gaps": gaps,
            "messages": messages_to_add,
        }

    return executor_node


def _build_replanner_node(*, llm_judge, store):
    """汇总节点：把执行证据合并成最终回答。"""

    def replanner_node(state:State):
        """汇总执行证据，交给反思节点，然后生成最终报告。"""
        user_question = _last_user_message(state["messages"])
        past_steps = list(state.get("past_steps") or [])
        history = "\n".join(
            f"【{str(task)}】\n{str(result)}" for task, result in past_steps
        )
        gaps = "\n".join(
            f"- {item}" for item in (state.get("evidence_gaps") or []) if item
        )
        prompt = _replanner_prompt(
            user_question,
            history or "（无执行记录）",
            evidence_excerpts=_merge_evidence_excerpts(
                state.get("evidence") or [], state.get("messages") or []
            ),
            gaps=gaps,
        )
        response = llm_judge.invoke(prompt)
        # 补上长期记忆写入
        config = get_config()
        configurable = _configurable(config)
        user_id = int(configurable.get("user_id", 0) or 0)
        memory_store = store or get_store()

        if memory_store and user_question and response.content:
            key = hashlib.sha256(f"{user_id}:{user_question}".encode("utf-8")).hexdigest()[:32]
            memory_store.put(
                (str(user_id), "memories"),
                key=key,
                value={"text": f"用户问：{user_question[:200]}\n深度调研报告:{str(response.content)[:300]}", "updated_at": int(time.time())}
            )
        return {"messages": [AIMessage(content=str(response.content))]}

    return replanner_node


def _build_reflection_node(llm_judge):
    """审稿节点：检查回答是否捏造证据、是否跑题。"""

    def reflection_code(state:State):
        """审稿人节点：检查最后一条 AI 回答是否基于证据、是否切题。"""
        last_message = state["messages"][-1]
        # 如果最后一条是工具调用，说明还没生成最终回答，直接跳过反思
        if getattr(last_message, "tool_calls", None):
            return {"critique": "PASS"}

        user_question = _last_user_message(state["messages"])
        # 提取生成过程中的工具调用证据
        evidence = []
        for msg in state["messages"]:
            if msg.type == "tool":
                evidence.append(msg.content)

        past_steps_text = "\n".join([f"【{str(task)}】\n{str(result)}" for task, result in state.get("past_steps", [])])
        evidence_text = "\n\n".join(evidence)[:2000] + "\n\n" + past_steps_text[:2000] if evidence or past_steps_text else "无检索证据"

        prompt = f"""你是一个严苛的学术审稿人。请审查 AI 的回答。
        【用户问题】{user_question}
        【检索到的证据】{evidence_text}
        【AI的初始回答】{last_message.content}
        请严格审查:
        1. AI 是否在捏造证据(幻觉)?
        2. AI 是否偏离了用户的问题?
        3. 如果涉及事实，是否有证据支撑？
        如果回答完美，请只输出："PASS"。
        如果回答不合格,请输出简短的修改意见(如:缺少对XXX概念的解释,或者编造了不存在的作者)。不要输出其他废话。"""

        response = llm_judge.invoke(prompt)
        
        critique = str(response.content).strip()
        return {
            "critique": critique,
            "reflection_count": state.get("reflection_count", 0) + 1
        }

    return reflection_code


def _build_tools_node(*, tool_map, settings):
    """工具节点：执行 chatbot 发起的 tool_calls，并与 executor 共用丰富化。"""

    def tools_node(state: State):
        last = state["messages"][-1]
        outputs = []
        for call in getattr(last, "tool_calls", []) or []:
            function = tool_map.get(call.get("name", ""))
            if function is None:
                result = f"错误：不存在名为 {call.get('name')} 的工具。"
            else:
                try:
                    arguments = call.get("args") or {}
                    logger.info(
                        "工具调用 user=%s name=%s args=%s",
                        _current_user_id(),
                        call.get("name"),
                        str(arguments)[:500],
                    )
                    if getattr(function, "coroutine", None) is not None:
                        result = asyncio.run(function.ainvoke(arguments))
                    elif hasattr(function, "invoke"):
                        result = function.invoke(arguments)
                    else:
                        result = function(**arguments)
                except Exception:
                    logger.exception("工具执行异常：%s", call.get("name"))
                    result = "工具执行失败，请稍后重试。"

            # 与 executor 共用同一套数据丰富化：补摘要、表格转 Markdown。
            # 对话链路每次工具调用最多补 1 篇，避免一轮里连续回查拖慢响应。
            detail_fetcher = None
            if (
                call.get("name") != "fetch_article_details"
                and "fetch_article_details" in tool_map
                and settings.enrich_abstract_limit > 0
            ):

                def detail_fetcher(pmids: list[str]) -> str:
                    fetcher = tool_map["fetch_article_details"]
                    detail, ok_detail, _err = _invoke_tool_with_timeout(
                        fetcher,
                        _detail_tool_arguments(fetcher, pmids),
                        min(settings.tool_timeout, 30.0),
                    )
                    return detail if ok_detail else ""

            enriched, notes = _enrich_tool_output(
                str(call.get("name") or ""),
                str(result),
                fetch_detail=detail_fetcher,
                max_details=1,
                max_chars=settings.enrich_max_chars,
            )
            if notes:
                logger.info(
                    "数据丰富化 user=%s tool=%s notes=%s",
                    _current_user_id(),
                    call.get("name"),
                    list(notes),
                )
            outputs.append(
                ToolMessage(content=enriched, tool_call_id=call.get("id", ""))
            )
        return {
            "messages": outputs,
            "tool_rounds": int(state.get("tool_rounds") or 0) + 1,
        }

    return tools_node


def build_graph(checkpointer=None, store=None, extra_tools=None):
    """装配 LangGraph：router 是唯一决策点，planner/chatbot 两条主干。"""
    settings = get_settings()
    tools = _agent_tools(extra_tools)

    # 兼容普通函数与 MCP Tool 对象：函数取 __name__，Tool 取 .name
    tool_map = {}
    for tool in tools:
        name = _tool_name(tool)
        if name:
            tool_map[name] = tool

    logger.info("正在为 %d 个工具建立语义索引...", len(tools))
    embed_model = get_embeddings()
    tool_vectors = {}
    for name, tool in tool_map.items():
        description = getattr(tool, "description", name) or name
        tool_vectors[name] = embed_model.embed_query(f"{name}: {description}")
    retrieve_relevant_tools = _build_tool_retriever(tool_map, embed_model, tool_vectors)

    # DeepSeek endpoint 一律使用 reasoning_content 感知的实现：无论 reasoner 还是
    # 普通模型，只要服务端进入 thinking 模式就会返回 reasoning_content，而这个
    # 字段必须完整回传，否则 API 直接 400；其它端点退回剔除该字段的安全实现。
    llm_class = (
        _ReasoningContentChatOpenAI
        if _is_deepseek_reasoning_endpoint()
        else _SafeChatOpenAI
    )
    llm = llm_class(
        model=settings.llm_model,
        temperature=0.2,
        timeout=settings.llm_timeout,
        max_retries=2,
        **_prompt_cache_model_kwargs(settings.llm_prompt_cache_ttl),
    )
    llm_judge = llm_class(
        model=settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout,
        max_retries=2,
        **_prompt_cache_model_kwargs(settings.llm_prompt_cache_ttl),
    )

    builder = StateGraph(State)
    builder.add_node("router", router_node)
    builder.add_node(
        "chatbot", _build_chatbot_node(llm=llm, tools=tools, settings=settings, store=store)
    )
    builder.add_node("planner", _build_planner_node(llm_judge))
    builder.add_node(
        "executor",
        _build_executor_node(
            llm=llm,
            llm_judge=llm_judge,
            _retrieve_relevant_tools=retrieve_relevant_tools,
            tool_map=tool_map,
            settings=settings,
        ),
    )
    builder.add_node("replanner", _build_replanner_node(llm_judge=llm_judge, store=store))
    builder.add_node("reflection", _build_reflection_node(llm_judge))
    builder.add_node("tools", _build_tools_node(tool_map=tool_map, settings=settings))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", route_from_router, {"planner": "planner", "chatbot": "chatbot"}
    )
    builder.add_conditional_edges(
        "planner", route_after_planner, {"executor": "executor", "chatbot": "chatbot"}
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "replanner": "replanner", "chatbot": "chatbot"},
    )
    builder.add_edge("replanner", "reflection")
    builder.add_conditional_edges(
        "chatbot", should_continue, {"tools": "tools", "reflection": "reflection"}
    )
    builder.add_edge("tools", "chatbot")
    builder.add_conditional_edges(
        "reflection", route_after_reflection, {"chatbot": "chatbot", END: END}
    )
    return builder.compile(checkpointer=checkpointer, store=store)

# ---------------------------------------------------------------------------
# 鉴权路由
# ---------------------------------------------------------------------------


@app.post("/register")
async def register(req: AuthRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, req.username, req.password)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    logger.info("新用户注册:id=%s", user.id)
    return {"token": create_token(int(cast(int, user.id))), "username": user.username}


@app.post("/login")
async def login(req: AuthRequest, db: Session = Depends(get_db)):
    try:
        user = login_user(db, req.username, req.password)
    except ValueError:
        return JSONResponse(status_code=401, content={"error": "用户名或密码错误"})
    return {"token": create_token(int(cast(int, user.id))), "username": user.username}


@app.get("/me")
async def me(user_id: int = Depends(get_user_id), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"id": user.id, "username": user.username}


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    total = 0
    token = request.headers.get("authorization", "")
    if token.startswith("Bearer "):
        try:
            total = await run_in_threadpool(count_documents, decode_token(token[7:]))
        except Exception:
            total = 0
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"total_chunks": total},
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/healthz")
async def healthz():
    settings = get_settings()
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model,
        "offline": settings.hf_offline,
    }


# ---------------------------------------------------------------------------
# 问答与入库
# ---------------------------------------------------------------------------


def _thread_id(user_id: int, raw: str) -> str:
    cleaned = _THREAD_ID_RE.sub("-", raw or "")[:48].strip("-")
    return f"u{user_id}-{cleaned or 'default'}"


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content or "")


def _validate_question(req: AskRequest, settings) -> str:
    question = (req.question or "").strip()
    if not question and not req.image_base64:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(question) > settings.max_question_chars:
        raise HTTPException(status_code=400, detail="问题过长，请精简后重试")
    return question


async def _build_final_question(req: AskRequest, question: str, settings) -> str:
    if not req.image_base64:
        return question

    try:
        data, _ = decode_image_payload(req.image_base64, settings.max_image_mb * MB)
        description = await run_in_threadpool(describe_image_bytes, data)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except MultimodalError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

    return (
        f"用户上传了一张图片，视觉模型提取的内容如下：\n{description}\n\n"
        f"【用户的问题】{question or '请解释这张图片'}。"
        "请结合知识库检索结果回答。"
    )


def _ask_config(user_id: int, raw_thread_id: str, settings) -> dict:
    thread_id = _thread_id(user_id, str(raw_thread_id))
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "recursion_limit": settings.recursion_limit,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_conversation(db: Session, user_id: int, conversation_id: int | None):
    if conversation_id is None:
        return None
    try:
        return get_conversation(db, user_id, conversation_id)
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在") from None


def _display_question(req: AskRequest, question: str) -> str:
    if req.image_base64:
        return f"[图片] {question}".strip() if question else "[图片提问]"
    return question


def _persist_assistant_message(user_id: int, conversation_id: int | None, content: str) -> None:
    if not conversation_id or not content.strip():
        return
    try:
        with SessionLocal() as session:
            conversation = get_conversation(session, user_id, conversation_id)
            append_message(session, conversation, "assistant", content)
    except Exception:
        logger.exception("保存助手消息失败(conversation_id=%s)", conversation_id)


def _stream_answer(
    graph,
    final_question: str,
    config: dict,
    *,
    user_id: int | None = None,
    conversation_id: int | None = None,
    user_msg_id=None,
    enable_web_search: bool = False,
):
    """同步生成器：由 StreamingResponse 放到线程池迭代，兼容 SqliteSaver。"""
    pieces: list[str] = []

    prev_node = None  # 记录上一个流出的节点
    # 定义节点映射，给用户看的状态提示
    status_map = {
        "planner": "🧠 正在拆解任务计划...",
        "executor": "⚙️ 正在执行下一步任务...",
        "tools": "🔍 正在检索知识库与全网资料...",
        "replanner": "📝 正在汇总生成深度报告...",
        "reflection": "🧐 审稿人正在核对证据..."
    }

    try:
        # 新增：首先把用户消息的 ID 发给前端
        if user_msg_id:
            yield _sse({"type": "user_msg_id", "id": user_msg_id})

        # 开局先告诉用户本轮判定（命中缓存，零额外成本），
        # 这样“为什么这次走计划/为什么这次直接答”对用户是可见的。
        if get_settings().router_explain:
            try:
                decision = _decide_route(final_question)
                intent = _classify_intent(final_question)
                label = (
                    "多步研究"
                    if decision.route == "planner"
                    else "直接作答"
                )
                scope = (
                    f"外部检索:{intent.source_text()}"
                    if intent.needs_retrieval
                    else "通用知识·不检索"
                )
                yield _sse(
                    {
                        "type": "status",
                        "content": f"🧭 判定为{label}｜{scope}（{decision.reason_text()}）",
                    }
                )
            except Exception:
                logger.warning("生成路由状态提示失败", exc_info=True)

        for chunk, metadata in graph.stream(
            {
                "messages": [{"role": "user", "content": final_question}],
                "enable_web_search": enable_web_search,
            },
            config,
            stream_mode="messages",
        ):
            node_name = (metadata or {}).get("langgraph_node")

            if node_name in status_map:
                if prev_node != node_name:
                    yield _sse({"type": "status", "content": status_map[node_name]})
           
            if node_name in ("chatbot", "replanner"):
                text = _content_to_text(getattr(chunk, "content", ""))
                if text:
                    # 如果这是该节点产出的第一个字，发一个 reset 信号
                    if prev_node != node_name:
                        yield _sse({"type": "reset"})
                        pieces = []  # 清空后端缓存

                    pieces.append(text)
                    yield _sse({"type": "token", "content": text})
            # 更新记录的上一个节点
            if node_name:
                prev_node = node_name
    except GraphRecursionError:
        logger.warning("问答步数超过上限(thread=%s)", config.get("configurable", {}).get("thread_id"))
        yield _sse(
            {
                "type": "error",
                "message": "检索或生成步骤过多，已停止。请缩小问题范围，或直接说“基于某篇文献生成论文 doc”。",
            }
        )
        return
    except HTTPException as exc:
        logger.warning(
            "流式问答返回 HTTP 错误(status=%s, thread=%s)",
            exc.status_code,
            config.get("configurable", {}).get("thread_id"),
        )
        message = (
            "模型服务暂时不可用，请稍后重试；服务端日志已记录详细原因。"
            if exc.status_code == 503
            else "回答生成失败，请稍后重试"
        )
        yield _sse({"type": "error", "message": message})
        return
    except Exception:
        logger.exception("流式问答失败")
        yield _sse({"type": "error", "message": "回答生成失败，请稍后重试"})
        return

    if not pieces:
        yield _sse({"type": "error", "message": "模型没有返回内容，请稍后重试"})
        return

    full_answer = _remove_repeated_answer("".join(pieces))
    # 工具名/函数名/入参 JSON 属于实现细节，任何配置下都不该出现在正文里
    full_answer, leaked = scrub_tool_leakage(full_answer)
    if leaked:
        logger.info("最终答案清洗：剔除 %d 行工具/函数痕迹", leaked)
    if not full_answer.strip():
        # 模型最后只吐了一段工具调用语法（工具已被轮次上限关闭）→ 给用户明确交代
        logger.warning("清洗后回答为空：模型只输出了工具调用语法")
        full_answer = EMPTY_ANSWER_FALLBACK
    if get_settings().answer_concise:
        full_answer, dropped = _strip_answer_filler(full_answer)
        if dropped:
            logger.info("最终答案清洗：剔除 %d 行免责声明/学术套话", dropped)
    if user_id is not None:
        _persist_assistant_message(user_id, conversation_id, full_answer)
    yield _sse({"type": "done", "content": full_answer})


@app.post("/conversations")
async def create_conversation(
    user_id: int = Depends(get_user_id), db: Session = Depends(get_db)
):
    conversation = create_conversation_record(db, user_id)
    return {"conversation": conversation_to_dict(conversation, 0)}


@app.get("/conversations")
async def list_conversations(
    user_id: int = Depends(get_user_id), db: Session = Depends(get_db)
):
    return {"conversations": list_conversation_records(db, user_id)}


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    user_id: int = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    conversation = _resolve_conversation(db, user_id, conversation_id)
    return {"messages": list_conversation_messages(db, conversation)}


@app.delete("/conversations/{conversation_id}")
async def remove_conversation(
    conversation_id: int,
    request: Request,
    user_id: int = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    conversation = _resolve_conversation(db, user_id, conversation_id)
    checkpoint_thread = thread_key(user_id, conversation)

    checkpointer = getattr(request.app.state, "checkpointer", None)
    delete_conversation_record(db, conversation)
    if checkpointer is not None:
        try:
            checkpointer.delete_thread(checkpoint_thread)
        except Exception:
            logger.exception("删除会话 checkpoint 失败：%s", checkpoint_thread)
    return {"ok": True}


@app.post("/ask")
async def ask(
    req: AskRequest,
    request: Request,
    user_id: int = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    question = _validate_question(req, settings)
    final_question = await _build_final_question(req, question, settings)
    conversation = _resolve_conversation(db, user_id, req.conversation_id)
    display_question = _display_question(req, question)

    if conversation is not None:
        append_message(db, conversation, "user", display_question)

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    raw_thread = conversation.thread_id if conversation else req.thread_id
    config = _ask_config(user_id, str(raw_thread), settings)
    try:
        result = await run_in_threadpool(
            graph.invoke,
            {
                "messages": [{"role": "user", "content": final_question}],
                "enable_web_search": req.enable_web_search,
            },
            config,
        )
    except HTTPException:
        raise
    except GraphRecursionError:
        raise HTTPException(
            status_code=503,
            detail="检索或生成步骤过多，已停止。请缩小问题范围后重试。",
        ) from None
    except Exception:
        logger.exception("问答失败(user_id=%s)", user_id)
        raise HTTPException(status_code=503, detail="问答服务暂时不可用，请稍后重试") from None

    answer = _content_to_text(result["messages"][-1].content)
    answer, leaked = scrub_tool_leakage(answer)
    if leaked:
        logger.info("最终答案清洗：剔除 %d 行工具/函数痕迹", leaked)
    if not answer.strip():
        logger.warning("清洗后回答为空：模型只输出了工具调用语法")
        answer = EMPTY_ANSWER_FALLBACK
    if settings.answer_concise:
        answer, dropped = _strip_answer_filler(answer)
        if dropped:
            logger.info("最终答案清洗：剔除 %d 行免责声明/学术套话", dropped)
    if conversation is not None:
        append_message(db, conversation, "assistant", answer)
    return {
        "answer": answer,
        "conversation_id": conversation.id if conversation is not None else None,
    }


@app.post("/ask/stream")
async def ask_stream(
    req: AskRequest,
    request: Request,
    user_id: int = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    question = _validate_question(req, settings)
    final_question = await _build_final_question(req, question, settings)
    conversation = _resolve_conversation(db, user_id, req.conversation_id)

    graph = getattr(request.app.state, "graph", None)
    checkpointer = getattr(request.app.state, "checkpointer", None)

    if graph is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    raw_thread = conversation.thread_id if conversation else req.thread_id
    config = _ask_config(user_id, str(raw_thread), settings)
    if req.edit_message_id is not None:
        if conversation is None:
            raise HTTPException(status_code=400, detail="编辑消息需要指定有效会话")

        messages = list_conversation_messages(db, conversation)
        target_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message["id"] == req.edit_message_id
            ),
            None,
        )
        if target_index is None:
            raise HTTPException(status_code=404, detail="要编辑的消息不存在")
        if messages[target_index]["role"] != "user":
            raise HTTPException(status_code=400, detail="只能编辑用户消息")

        history_messages = messages[:target_index]
        if checkpointer is not None:
            try:
                configurable = _configurable(config)
                thread_id = configurable.get("thread_id")
                if thread_id is not None:
                    checkpointer.delete_thread(str(thread_id))
            except Exception as exc:
                logger.exception("重置编辑会话的 checkpoint 失败")
                raise HTTPException(
                    status_code=503,
                    detail="无法重置会话上下文，请稍后重试",
                ) from exc

        delete_messages_after(db, conversation, req.edit_message_id)
        if history_messages:
            graph.update_state(
                config,
                {
                    "messages": [
                        HumanMessage(content=message["content"])
                        if message["role"] == "user"
                        else AIMessage(content=message["content"])
                        for message in history_messages
                    ]
                },
            )

    user_message = None
    if conversation is not None:
        user_message = append_message(
            db, conversation, "user", _display_question(req, question)
        )

    return StreamingResponse(
        _stream_answer(
            graph,
            final_question,
            config,
            user_id=user_id,
            conversation_id=conversation.id if conversation is not None else None, # type: ignore
            user_msg_id=user_message.id if user_message is not None else None,
            enable_web_search=req.enable_web_search,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _analyze_upload(user_id: int, source: str) -> dict:
    """从已入库块取摘要，避免再次解析(尤其避免扫描版 PDF 二次 OCR)。"""
    try:
        from vector_store import get_user_collection

        db = get_user_collection(user_id)
        data = db.get(where={"source": source}, limit=20, include=["documents"])
        combined = "\n".join(data.get("documents") or [])[:6000]
        if not combined.strip():
            return {}
        return analyze_document(source, combined)
    except Exception:
        logger.exception("文档智能分析失败：%s", source)
        return {}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: int = Depends(get_user_id),
):
    settings = get_settings()
    try:
        ext = validate_extension(file.filename, settings.allowed_upload_exts)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    display_name = sanitize_filename(file.filename, "document")
    destination = unique_path(user_dir("documents", user_id), display_name)

    try:
        await run_in_threadpool(
            save_stream_limited,
            file.file,
            destination,
            settings.max_upload_mb * MB,
        )
    except UploadValidationError as exc:
        status_code = 413 if "过大" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from None
    finally:
        await file.close()

    try:
        ok, message = await run_in_threadpool(
            build_index, str(destination), display_name, ext, user_id
        )
    except EmbeddingModelMismatch as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except Exception:
        logger.exception("索引入库失败：%s", display_name)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="文档入库失败") from None

    if not ok:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=message)

    invalidate_cache(user_id)

    analysis = {}
    if ext in {item.lstrip(".") for item in TEXT_EXTENSIONS}:
        analysis = await run_in_threadpool(_analyze_upload, user_id, display_name)

    logger.info("用户 %s 上传入库：%s(%s)", user_id, display_name, ext)
    return {
        "message": message,
        "doc": display_name,
        "type": ext,
        "analysis": analysis,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: int = Depends(get_user_id),
):
    settings = get_settings()
    try:
        ext = validate_extension(file.filename, settings.allowed_upload_exts)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if f".{ext}" not in TEXT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="该接口仅支持文本文档")

    display_name = sanitize_filename(file.filename, "document")
    temp_path = unique_path(user_dir("tmp", user_id), display_name)

    try:
        await run_in_threadpool(
            save_stream_limited, file.file, temp_path, settings.max_upload_mb * MB
        )
        try:
            blocks = await run_in_threadpool(load_document, str(temp_path))
        except DocumentParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        combined = "\n".join(block["content"] for block in blocks)[:6000]
        if not combined.strip():
            raise HTTPException(status_code=400, detail="文档内容为空")
        return await run_in_threadpool(analyze_document, display_name, combined)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    finally:
        await file.close()
        temp_path.unlink(missing_ok=True)


@app.post("/learn")
async def learn(req: LearnRequest, user_id: int = Depends(get_user_id)):
    try:
        ok, message = await run_in_threadpool(
            _learn_url, req.url, req.title, user_id
        )
    except FetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("网页学习失败(user_id=%s)", user_id)
        raise HTTPException(status_code=500, detail="学习失败，请稍后重试") from None
    if not ok:
        raise HTTPException(status_code=400, detail=message) from None
    return {"ok": True, "message": message}


# ---------------------------------------------------------------------------
# 文档管理与统计
# ---------------------------------------------------------------------------


@app.get("/documents")
async def get_documents(user_id: int = Depends(get_user_id)):
    documents = await run_in_threadpool(list_documents, user_id)
    return {"documents": documents, "count": len(documents)}


@app.delete("/documents")
async def delete_document(
    source: str = Query(..., min_length=1, max_length=255),
    user_id: int = Depends(get_user_id),
):
    try:
        result = await run_in_threadpool(remove_document, user_id, source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("删除文档失败(user_id=%s, source=%s)", user_id, source)
        raise HTTPException(status_code=500, detail="删除失败，请稍后重试") from None
    invalidate_cache(user_id)
    return {"ok": True, **result}


@app.get("/papers/{filename}")
async def download_paper(filename: str, user_id: int = Depends(get_user_id)):
    directory = (user_dir("documents", user_id) / "papers").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    candidate = (directory / filename).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError:
        raise HTTPException(status_code=404, detail="文件不存在") from None
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        candidate,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        filename=candidate.name,
    )


@app.get("/stats")
async def stats(user_id: int = Depends(get_user_id)):
    try:
        total = await run_in_threadpool(count_documents, user_id)
        documents = await run_in_threadpool(list_documents, user_id)
        embedding_model = await run_in_threadpool(get_embedding_model_name)
    except EmbeddingModelMismatch as exc:
        logger.warning(
            "统计失败：知识库索引与当前嵌入模型不一致(user_id=%s): %s",
            user_id,
            exc,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except Exception:
        logger.exception("统计失败(user_id=%s)", user_id)
        raise HTTPException(status_code=500, detail="统计失败") from None

    return {
        "total_chunks": total,
        "documents": len(documents),
        "embedding_model": embedding_model,
        "retrieval": "hybrid(vector+bm25)+cross-encoder",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
