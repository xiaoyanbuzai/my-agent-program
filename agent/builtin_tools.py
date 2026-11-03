"""内置工具：知识库检索、PubMed/arXiv、联网、网页入库、生成 Word 文档。"""

from __future__ import annotations

import logging

import os
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

from agent.common import _current_user_id
from config import get_settings, user_dir
from indexer import build_index
from kb_retriever import invalidate_cache, search_kb
from paper_writer import generate_paper_docx
from search_online import (
    FetchError,
    download_binary,
    fetch_page_text,
    search_arxiv,
    search_pubmed,
    validate_public_url,
    web_search,
)
from uploads import sanitize_filename, unique_path
from vector_store import EmbeddingModelMismatch

logger = logging.getLogger("web_app")


def kb_search_tool(query: str) -> str:
    """在用户知识库中检索相关内容。问题涉及已入库文献/网页时调用。"""
    try:
        user_id = _current_user_id()
        results = search_kb(query, user_id)
    except EmbeddingModelMismatch:
        logger.exception("知识库索引与当前嵌入模型不一致")
        return "知识库索引需要重建，请联系管理员运行重建脚本。"
    except Exception:
        logger.exception("知识库检索工具失败")
        return "知识库检索暂时不可用，请稍后重试。"

    if not results:
        return "知识库中没有检索到与该问题相关的内容。"

    parts = []
    for item in results:
        page = f" p.{item['page']}" if item["page"] else ""
        content = _compress_retrieval_content(query, item["content"])
        parts.append(
            f"[来源: {item['source']}{page} | 类型: {item['doc_type']} | 相关度: {item['score']:.3f}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(parts)


def _compress_retrieval_content(
    query: str,
    content: str,
    max_sentences: int = 3,
    max_chars: int = 600,
) -> str:
    """保留与查询词重合度最高的句子，供模型使用时仍保留原文与来源标注。"""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\r?\n+", content)
        if sentence.strip()
    ]
    if len(sentences) <= max_sentences and len(content) <= max_chars:
        return content.strip()

    query_lower = query.lower()
    query_terms = set(re.findall(r"[a-z0-9_]+", query_lower))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", query_lower):
        query_terms.update(
            phrase[index : index + 2] for index in range(max(1, len(phrase) - 1))
        )
    if not query_terms:
        return " ".join(sentences[:max_sentences])[:max_chars]

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = sum(term in sentence_lower for term in query_terms)
        ranked.append((score, index, sentence))
    selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected.sort(key=lambda item: item[1])

    compressed = " ".join(sentence for _, _, sentence in selected)
    if len(compressed) <= max_chars:
        return compressed
    return compressed[: max_chars - 1].rstrip() + "…"


def search_papers_tool(topic: str) -> str:
    """检索 arXiv 学术论文，返回真实标题、作者、链接和摘要。"""
    try:
        papers = search_arxiv(topic, max_results=5)
    except FetchError as exc:
        return f"学术检索失败：{exc}"
    except Exception:
        logger.exception("arXiv 检索工具失败")
        return "学术检索服务暂时不可用。"

    if not papers:
        return "未检索到相关论文。"

    parts = []
    for paper in papers:
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " 等"
        parts.append(
            f"- **{paper['title']}**\n  作者：{authors or '未知'}\n"
            f"  链接：{paper['url']}\n  摘要：{paper['summary']}"
        )
    return "\n\n".join(parts)


def search_pubmed_tool(query: str) -> str:
    """检索 PubMed 生物医学文献，返回真实标题、作者、期刊、年份、PMID 链接与摘要。

    适用于基因突变、癌症预后、临床研究、药物、疾病等生物医学主题。
    参数 query 建议用英文检索词（如 "TP53 mutation cancer prognosis"）。
    """
    try:
        papers = search_pubmed(query, max_results=5)
    except FetchError as exc:
        return f"PubMed 检索失败：{exc}"
    except Exception:
        logger.exception("PubMed 检索工具失败")
        return "PubMed 检索服务暂时不可用。"

    if not papers:
        return "PubMed 未检索到相关文献，请尝试调整检索词或英文关键词。"

    parts = []
    for paper in papers:
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " 等"
        meta = " | ".join(
            part
            for part in (paper.get("journal"), paper.get("year"))
            if part
        )
        parts.append(
            f"- **{paper['title']}**\n"
            f"  作者：{authors or '未知'}\n"
            f"  期刊/年份：{meta or '未知'}\n"
            f"  链接：{paper['url']}"
        )
    return "\n\n".join(parts)


def generate_paper_docx_tool(
    title: str,
    content: str,
    author: str = "",
    affiliation: str = "",
) -> str:
    """当用户明确要求“生成论文 doc / 导出 Word / 生成论文文档”时调用。

    参数：
    - title:论文标题;
    - content:完整论文 Markdown,必须包含标题、摘要、关键词、正文各章节和参考文献；
      正文不要带“[来源: …]”检索标记，文内引用改用 [1]、[2] 并在文末列出；
    - author / affiliation:作者与单位,未知时留空,不要编造。

    工具会把内容排成中文论文格式并保存为当前账号可下载的 .docx。
    """
    user_id = _current_user_id()
    if user_id <= 0:
        return "无法识别当前用户，请重新登录。"

    if not (content or "").strip():
        return "论文内容为空：请先根据知识库撰写完整论文正文，再调用本工具。"

    try:
        _, filename = generate_paper_docx(
            user_id,
            title=title,
            content=content,
            author=author,
            affiliation=affiliation,
        )
    except ValueError as exc:
        return f"论文生成失败：{exc}"
    except Exception:
        logger.exception("论文 Word 生成失败（user_id=%s）", user_id)
        return "论文 Word 生成失败，请稍后重试。"

    link = "/papers/" + quote(filename)
    return (
        f"论文 Word 已生成：**《{title.strip() or '学术论文'}》**\n\n"
        f"[下载 .docx]({link})\n\n"
        f"文件：`{filename}`，已按中文论文格式排版（标题、作者/单位、摘要、关键词、"
        f"章节标题、正文缩进、参考文献、页码）。\n\n"
        "请直接把上面的下载链接回复给用户，不要再调用任何工具。"
    )


def _learn_url(url: str, title: str, user_id: int) -> tuple[bool, str]:
    """抓取 URL 并入库；返回 (成功, 面向用户的提示)。"""
    try:
        safe_url = validate_public_url(url)
    except FetchError as exc:
        return False, str(exc)

    try:
        text = fetch_page_text(safe_url)
    except FetchError as exc:
        return False, str(exc)

    if not text:
        return False, "网页抓取失败或内容为空。"

    directory = user_dir("web", user_id)
    parsed = urlparse(safe_url)
    fallback_title = Path(parsed.path).stem or parsed.netloc or "webdoc"
    safe_title = sanitize_filename(title or fallback_title, "webdoc")
    stem = Path(safe_title).stem or "webdoc"

    if text == "__IS_PDF__":
        try:
            data = download_binary(safe_url)
        except FetchError as exc:
            return False, str(exc)

        pdf_path = unique_path(directory, f"{stem}.pdf")
        try:
            pdf_path.write_bytes(data)
            ok, message = build_index(
                str(pdf_path), pdf_path.name, "pdf", user_id=user_id
            )
            if not ok:
                pdf_path.unlink(missing_ok=True)
                return False, message
        except Exception:
            logger.exception("PDF 学习入库失败：%s", safe_url)
            pdf_path.unlink(missing_ok=True)
            return False, "PDF 解析或入库失败。"

        invalidate_cache(user_id)
        return True, f"已学习并入库 PDF:{pdf_path.name}"

    md_path = unique_path(directory, f"{stem}.md")
    try:
        md_path.write_text(
            f"# {title or stem}\n来源: {safe_url}\n\n{text}", encoding="utf-8"
        )
        ok, message = build_index(str(md_path), md_path.name, "md", user_id=user_id)
        if not ok:
            md_path.unlink(missing_ok=True)
            return False, message
    except Exception:
        logger.exception("网页学习入库失败：%s", safe_url)
        md_path.unlink(missing_ok=True)
        return False, "网页解析或入库失败。"

    invalidate_cache(user_id)
    return True, f"已学习并入库网页：{md_path.name}"


def learn_from_url_tool(url: str, title: str = "") -> str:
    """抓取网页正文并保存到当前用户的知识库，之后可以检索。"""
    try:
        user_id = _current_user_id()
    except Exception:
        return "无法确定当前用户，请重新登录。"

    ok, message = _learn_url(url, title, user_id)
    return message if ok else f"学习失败：{message}"


def web_news_tool(query: str) -> str:
    """联网搜索最新资讯(Tavily 优先,DuckDuckGo 兜底),最先使用。"""
    settings = get_settings()
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()

    if tavily_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_raw_content": True, 
                    "max_results": 3,
                },
                timeout=settings.fetch_timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                parts = []
                for item in results:
                    title = item.get('title', '无标题')
                    url = item.get('url', '')
                    full_text = item.get('raw_content') or item.get('content') or ''
                    parts.append(
                        f"### [{title}]({url})\n"
                        f"**正文内容**:\n{full_text[:8000]}\n"
                    )
                return "以下为联网搜索结果：\n\n" + "\n\n".join(parts)
        except Exception as exc:
            logger.warning("Tavily 搜索失败，降级到 DuckDuckGo:%s", exc)

    try:
        results = web_search(query, max_results=5)
    except Exception:
        logger.exception("联网搜索失败")
        return "联网搜索暂时不可用，请稍后重试。"

    if not results:
        return "联网搜索没有返回结果，请提供具体网页 URL 让知识库学习。"

    parts = [
        f"- {item.get('title', '')}\n  {item.get('url', '')}\n  {item.get('snippet', '')[:200]}"
        for item in results
    ]
    return "以下为联网搜索结果(DuckDuckGo):\n\n" + "\n\n".join(parts)
