"""联网能力模块：通用网页搜索 + 学术检索 + 网页正文抓取。

设计原则：
- 通用搜索采用多通道 + 优雅降级：优先 duckduckgo-search，失败退到 HTML 端点。
- 学术检索走 arXiv API（免费、稳定、无需 key），返回真实标题+链接。
- 网页正文抓取用于"学习新文章并入库存档"，带超时与大小限制。
"""
import urllib.request
import urllib.parse
import re
import html as _html
import json

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# ---------- 通用网页搜索 ----------

def web_search(query: str, max_results: int = 5):
    """通用网页搜索，返回 [{title, url, snippet}]。多通道降级。"""
    # 通道1：duckduckgo-search 库
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        if raw:
            return [
                {"title": x.get("title", ""), "url": x.get("href", ""),
                 "snippet": x.get("body", "") or x.get("snippet", "")}
                for x in raw if x.get("href")
            ]
    except Exception:
        pass
    # 通道2：DuckDuckGo HTML 端点（备选）
    return _web_search_html(query, max_results)


def _web_search_html(query: str, max_results: int = 5):
    try:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
            content = r.read().decode("utf-8", "ignore")
        results = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', content)
        out = []
        for href, txt in results[:max_results]:
            title = re.sub(r"<[^>]+>", "", _html.unescape(txt)).strip()
            out.append({"title": title, "url": href, "snippet": ""})
        return out
    except Exception:
        return []


# ---------- 学术期刊检索 ----------

def search_arxiv(query: str, max_results: int = 5):
    """通过 arXiv API 检索学术论文/预印本，返回真实标题+链接+摘要。"""
    api_url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results})
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = r.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    from xml.etree import ElementTree as ET
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.find("a:title", ns).text or "").split())
        link = (e.find("a:id", ns).text or "").strip()
        summary = " ".join((e.find("a:summary", ns).text or "").split())
        authors = [
            (a.find("a:name", ns).text or "") for a in e.findall("a:author", ns)
        ]
        out.append({
            "title": title,
            "url": link,
            "summary": summary[:400],
            "authors": authors,
            "type": "arxiv",
        })
    return out


def search_topic_trending(query: str, max_results: int = 5):
    """主题趋势检索：优先 arXiv 最新相关论文，辅以通用网页新闻。"""
    papers = []
    try:
        papers = search_arxiv(query, max_results)
    except Exception:
        pass
    web = web_search(query, max_results)
    return {"papers": papers, "web": web}


# ---------- 网页正文抓取（用于学习并入库存档） ----------

def fetch_page_text(url: str, max_chars: int = 8000):
    """抓取网页正文纯文本，限制大小，失败返回 None。"""
    try:
        from bs4 import BeautifulSoup
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            html_data = r.read(500000)
        soup = BeautifulSoup(html_data, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:max_chars] if text else None
    except Exception:
        return None