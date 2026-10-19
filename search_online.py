"""联网能力：安全抓取、通用网页搜索、arXiv 学术检索。

安全约束：
- 仅允许 http/https，禁止 URL 内嵌凭证与非常规端口；
- 默认解析 DNS 后拒绝内网 / 回环 / 链路本地等非公网地址（可配置 FETCH_ALLOWED_HOSTS 收紧）；
- 所有下载都有超时与字节上限，PDF 需通过 ``%PDF`` 魔数校验。

说明：``validate_public_url`` 是"解析后校验"，对 DNS rebinding 只能降低风险、无法完全消除；
面向公网部署时应同时配置出口防火墙。
"""

from __future__ import annotations

import html as _html
import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from config import get_settings

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LiteratureAgent/1.0"
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}


class FetchError(RuntimeError):
    """抓取失败或目标 URL 不被允许。"""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向目标也必须通过公网校验，避免用 302 绕过 SSRF 防护。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def _host_matches_allowlist(host: str, allowed: frozenset[str]) -> bool:
    if not allowed:
        return True
    host = host.lower().rstrip(".")
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


def validate_public_url(url: str) -> str:
    """校验 URL 可安全抓取，返回规范化后的 URL。"""
    settings = get_settings()
    if not url or not url.strip():
        raise FetchError("URL 不能为空")

    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise FetchError("仅支持 http/https 链接")
    if parsed.username or parsed.password:
        raise FetchError("URL 不能包含用户名或密码")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise FetchError("URL 缺少主机名")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".local", ".internal", ".localhost")
    ):
        raise FetchError("不允许抓取本机或内网地址")

    try:
        port = parsed.port
    except ValueError as exc:
        raise FetchError("URL 端口不合法") from exc
    if port is not None and port not in ALLOWED_PORTS:
        raise FetchError("仅允许 80/443 端口")

    if not _host_matches_allowlist(host, settings.fetch_allowed_hosts):
        raise FetchError("该域名不在 FETCH_ALLOWED_HOSTS 允许列表内")

    # 未配置白名单时，解析 DNS 并拒绝一切非公网地址
    if not settings.fetch_allowed_hosts:
        resolve_port = port or (443 if parsed.scheme.lower() == "https" else 80)
        try:
            infos = socket.getaddrinfo(
                host, resolve_port, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror as exc:
            raise FetchError("域名解析失败") from exc
        for info in infos:
            address = info[4][0].split("%", 1)[0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise FetchError("域名解析结果异常") from exc
            if not ip.is_global:
                raise FetchError("不允许抓取内网 / 保留地址")

    return urllib.parse.urlunsplit(parsed)


def _read_limited(response, max_bytes: int) -> bytes:
    data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FetchError(f"响应体超过 {max_bytes // (1024 * 1024)} MB 限制")
    return data


def _open(request: urllib.request.Request, timeout: float):
    try:
        return _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        logger.warning("抓取返回 HTTP %s：%s", exc.code, request.full_url)
        raise FetchError(f"目标站点返回 HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("抓取失败：%s（%s）", request.full_url, exc)
        raise FetchError("网络请求失败或超时") from exc


def download_binary(url: str, *, max_bytes: int | None = None) -> bytes:
    """下载二进制内容（用于 PDF），带大小与魔数校验。"""
    settings = get_settings()
    safe_url = validate_public_url(url)
    limit = max_bytes or settings.max_pdf_mb * 1024 * 1024
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})

    with _open(request, settings.fetch_timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > limit:
            raise FetchError(f"文件超过 {limit // (1024 * 1024)} MB 限制")
        data = _read_limited(response, limit)

    if not data.startswith(b"%PDF"):
        raise FetchError("返回内容不是有效的 PDF")
    return data


def fetch_page_text(url: str, max_chars: int | None = None) -> str | None:
    """抓取网页正文（经 Jina Reader 提纯）。PDF 返回 ``__IS_PDF__`` 交给上层下载。"""
    settings = get_settings()
    safe_url = validate_public_url(url)
    if safe_url.lower().endswith(".pdf"):
        return "__IS_PDF__"

    limit = settings.fetch_max_mb * 1024 * 1024
    reader_url = f"https://r.jina.ai/{safe_url}"
    request = urllib.request.Request(reader_url, headers={"User-Agent": USER_AGENT})

    try:
        with _open(request, settings.fetch_timeout) as response:
            raw = _read_limited(response, limit)
    except FetchError:
        raise
    except Exception as exc:  # pragma: no cover - 兜底
        logger.exception("网页正文抓取失败：%s", safe_url)
        raise FetchError("网页正文抓取失败") from exc

    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    return text[: max_chars or settings.fetch_max_chars]


# ---------- 通用网页搜索 ----------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """多通道搜索：duckduckgo-search 优先，失败降级到 HTML 端点。"""
    if not query or not query.strip():
        return []

    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=10) as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        if raw:
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "") or item.get("snippet", ""),
                }
                for item in raw
                if item.get("href")
            ]
    except Exception as exc:
        logger.warning("DuckDuckGo SDK 搜索失败，降级到 HTML 端点：%s", exc)

    return _web_search_html(query, max_results)


def _web_search_html(query: str, max_results: int = 5) -> list[dict]:
    settings = get_settings()
    try:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with _open(request, settings.fetch_timeout) as response:
            content = _read_limited(response, settings.fetch_max_mb * 1024 * 1024).decode(
                "utf-8", "ignore"
            )
        matches = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', content
        )
        results = []
        for href, text in matches[:max_results]:
            title = re.sub(r"<[^>]+>", "", _html.unescape(text)).strip()
            results.append({"title": title, "url": href, "snippet": ""})
        return results
    except Exception as exc:
        logger.warning("HTML 搜索端点失败：%s", exc)
        return []


# ---------- 学术检索 ----------

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """通过 arXiv Atom API 检索论文，返回真实标题、链接、作者与摘要。"""
    if not query or not query.strip():
        return []

    settings = get_settings()
    api_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    request = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})

    try:
        with _open(request, settings.fetch_timeout + 5) as response:
            data = _read_limited(response, 5 * 1024 * 1024)
        root = ET.fromstring(data)
    except (FetchError, ET.ParseError) as exc:
        logger.warning("arXiv 检索失败：%s", exc)
        raise FetchError("学术检索服务暂时不可用") from exc

    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        title_node = entry.find("a:title", ns)
        id_node = entry.find("a:id", ns)
        summary_node = entry.find("a:summary", ns)
        results.append(
            {
                "title": " ".join((title_node.text or "").split()) if title_node is not None else "",
                "url": (id_node.text or "").strip() if id_node is not None else "",
                "summary": (
                    " ".join((summary_node.text or "").split())[:400]
                    if summary_node is not None
                    else ""
                ),
                "authors": [
                    (author.find("a:name", ns).text or "")
                    for author in entry.findall("a:author", ns)
                    if author.find("a:name", ns) is not None
                ],
                "type": "arxiv",
            }
        )
    return results


def search_topic_trending(query: str, max_results: int = 5) -> dict:
    """主题趋势：arXiv 最新论文 + 网页结果，单侧失败不影响另一侧。"""
    papers: list[dict] = []
    try:
        papers = search_arxiv(query, max_results)
    except FetchError:
        pass
    return {"papers": papers, "web": web_search(query, max_results)}
