"""文档智能分析：对上传/入库的文档内容生成摘要、关键词、主题分类。

复用项目的 deepseek LLM（通过 ChatOpenAI）完成分析，输出结构化 JSON。
"""
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="deepseek-flash", temperature=0.2)
    return _llm


def analyze_document(title: str, content: str, max_len: int = 4000) -> dict:
    """分析文档内容，返回 {title, summary, keywords, category}。

    失败时返回降级结果，不让上传流程中断。
    """
    snippet = content[:max_len]
    prompt = (
        "你是一个文献分析助手。分析下面这篇文档的要点，并用 JSON 输出，"
        "不要输出其他内容，格式如下：\n"
        "{\"summary\": \"150字以内的核心内容概括\", "
        "\"keywords\": [\"关键词1\", \"关键词2\", \"关键词3\"], "
        "\"category\": \"文档所属领域/技术类别\"}\n\n"
        f"文档标题：{title}\n文档内容：\n{snippet}"
    )
    try:
        resp = _get_llm().invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 提取 JSON 块
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("无 JSON 输出")
        data = json.loads(text[start:end + 1])
        return {
            "title": title,
            "summary": data.get("summary", ""),
            "keywords": data.get("keywords", []),
            "category": data.get("category", ""),
        }
    except Exception:
        return {
            "title": title,
            "summary": "",
            "keywords": [],
            "category": "",
        }