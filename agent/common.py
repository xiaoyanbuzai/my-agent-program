"""通用工具元信息与请求上下文。"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_config


_SUPPORTED_HERMES_READ_TOOLS = frozenset(
    {
        "channels_list",
        "conversations_list",
        "events_poll",
        "permissions_list_open",
    }
)


# PubMed 生物医学检索 MCP（pubmed-search-mcp）导出的全部工具名。
# 语义化学术检索 MCP（s2-mcp-server）的工具一律带 ``semantic_scholar_`` 前缀，
# 在 _is_supported_extra_tool 中按前缀放行。
_SUPPORTED_PUBMED_TOOLS = frozenset(
    {
        "unified_search",
        "analyze_search_query",
        "validate_pico_plan",
        "generate_search_queries",
        "find_related_articles",
        "find_citing_articles",
        "get_article_references",
        "fetch_article_details",
        "get_citation_metrics",
        "verify_reference_list",
        "prepare_export",
        "save_literature_notes",
        "get_fulltext",
        "get_text_mined_terms",
        "get_article_figures",
        "search_gene",
        "get_gene_details",
        "get_gene_literature",
        "search_compound",
        "get_compound_details",
        "get_compound_literature",
        "search_clinvar",
        "build_citation_tree",
        "build_research_chronicle",
        "read_research_chronicle",
        "prepare_figure_search",
        "configure_institutional_access",
        "get_institutional_link",
        "list_resolver_presets",
        "test_institutional_access",
        "diagnose_institutional_access",
        "convert_icd_mesh",
        "search_biomedical_images",
        "save_pipeline",
        "list_pipelines",
        "load_pipeline",
        "delete_pipeline",
        "get_pipeline_history",
        "schedule_pipeline",
        "unschedule_pipeline",
        "read_session",
    }
)


def _tool_name(tool) -> str:
    return str(getattr(tool, "__name__", getattr(tool, "name", "")))


def _is_supported_extra_tool(tool) -> bool:
    """判定 MCP 工具是否可被 Agent 使用。

    - Hermes 只暴露只读工具（频道/会话/事件/权限查询），不暴露消息读写、权限审批等写操作；
    - Semantic Scholar（``semantic_scholar_`` 前缀）与 PubMed（白名单集合）的检索工具全部放行。
    """
    name = _tool_name(tool).strip()
    if not name:
        return False
    if name in _SUPPORTED_HERMES_READ_TOOLS or name in _SUPPORTED_PUBMED_TOOLS:
        return True
    lowered = name.lower()
    return lowered.startswith("semantic_scholar_")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _configurable(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    value = config.get("configurable")
    return value if isinstance(value, dict) else {}


def _current_user_id() -> int:
    config = get_config()
    configurable = _configurable(config)
    return int(configurable.get("user_id", 0) or 0)
