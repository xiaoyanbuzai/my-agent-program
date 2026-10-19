import os
os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import Depends
from sqlalchemy.orm import Session
from db import get_db, init_db
from models import User
from auth import AuthRequest, get_user_id, register_user, login_user, create_token

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil

import base64
from zhipuai import ZhipuAI
from langgraph.config import get_config , get_store
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from contextlib import ExitStack

from kb_retriever import search_kb, count_documents,invalidate_cache
from indexer import build_index
from search_online import search_arxiv, fetch_page_text, web_search
from knowledge import analyze_document

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 网页正文暂存目录：URL 学习入库前先落盘
os.makedirs(os.path.join(BASE_DIR, "knowledge", "web"), exist_ok=True)



class State(TypedDict):
    messages: Annotated[list, add_messages]


def kb_search_tool(query: str) -> str:
    """在知识库中检索相关内容。用户问题涉及知识库文档时，必须先调用本工具。"""
    config = get_config()
    user_id = config["configurable"]["user_id"]
    results = search_kb(query, user_id)
    if not results:
        return "未找到相关内容"
    return "\n\n---\n\n".join(
        f"[来源:{r['source']}|类型:{r['doc_type']}|小节:{r['title']}]\n{r['content']}"
        for r in results
    )


def search_papers_tool(topic: str) -> str:
    """检索学术论文(arXiv),返回真实论文标题、链接和摘要。用于引用期刊/文献回答问题。"""
    try:
        papers = search_arxiv(topic, max_results=5)
    except Exception as e:
        return f"学术检索失败：{e}"
    if not papers:
        return "未检索到相关论文"
    parts = []
    for p in papers:
        authors = ", ".join(p["authors"][:3]) + (" 等" if len(p["authors"]) > 3 else "")
        parts.append(
            f"- **{p['title']}**\n  作者：{authors}\n  链接：{p['url']}\n  摘要：{p['summary']}"
        )
    return "\n\n".join(parts)


def learn_from_url_tool(url: str, title: str = "", user_id: int = None) -> str:
    """抓取网页正文，学习并保存到知识库，之后可检索。参数 url 为网页地址,title 可选。"""
    # 1. 获取 user_id（优先用传进来的，没有就去 Graph 上下文拿）
    if user_id is None:
        try:
            config = get_config()
            user_id = config["configurable"]["user_id"]
        except Exception as e:
            return f"获取用户信息失败：{e}，请确保已登录。"

    text = fetch_page_text(url)
    if not text:
        return f"网页抓取失败或内容为空：{url}"
    if not title:
        import urllib.parse
        title = urllib.parse.urlparse(url).netloc or "webdoc"
    # 落盘为 md 格式入库
    web_dir = os.path.join(BASE_DIR, "knowledge", "web", str(user_id))
    os.makedirs(web_dir, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in title)[:60] or "webdoc"
    md_path = os.path.join(web_dir, f"{safe}_{abs(hash(url)) % 100000}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n来源:{url}\n\n{text}")
    ok, msg = build_index(md_path, safe, "md",user_id=user_id)
    if not ok:
        return msg
    invalidate_cache(user_id)
    return f"已学习并入库来源 {url}({title})"

# 预留 Coze 官方联网接口：可配置后端。默认设有 OPENAI 无需搜索 key，
# 若配置了 COZE_WEB_SEARCH_ENDPOINT 则走该通道，否则回退到 search_online 的 web_search。
def web_news_tool(query: str) -> str:
    """联网搜索相关网页资讯（尽力而为）。当知识库无相关内容或用户要求最新信息时使用。"""
    endpoint = os.getenv("COZE_WEB_SEARCH_ENDPOINT", "")
    blocked_url = None
    if endpoint:
        # 预留：Coze 官方联网端点（需配置后启用）
        return f"（联网通道未开启：请在环境变量 COZE_WEB_SEARCH_ENDPOINT 配置 Coze 搜索端点）"
    try:
        results = web_search(query, max_results=5)
        if not results:
            return "未能获取联网搜索结果，请直接给出网页 URL 让知识库学习，或换一种问法。"
        parts = []
        for r in results[:5]:
            parts.append(f"- {r['title']}\n  {r['url']}\n  {r['snippet'][:150]}")
        return "以下为联网搜索结果（关键词匹配）：\n\n" + "\n\n".join(parts)
    except Exception as e:
        return f"联网搜索失败：{e}"


def build_graph(checkpointer=None, store=None):
    tools = [kb_search_tool, search_papers_tool, learn_from_url_tool, web_news_tool]
    tool_map = {t.__name__: t for t in tools}

    llm = ChatOpenAI(model="deepseek-flash")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        config = get_config()
        user_id = config["configurable"].get("user_id", "default")

        store = get_store()
        memories = store.search((str(user_id),"memories"), limit=5)
        memory_text = "\n".join([m.value['text'] for m in memories])
        system = {
            "role": "system",
            "content": (
                "你是一个知识库助手，具备以下能力：\n"
                "1. 用知识库检索工具查找已入库内容；\n"
                "2. 用户要求引用学术期刊/论文时，用论文检索工具获取真实 arXiv 论文并给出标题+链接；\n"
                "3. 用户可以给你一个网页链接让你学习并入库（调用抓取学习工具）；\n"
                "4. 用户需要最新资讯时，可尝试联网搜索工具；\n"
                "回答时应优先基于知识库检索结果，并按需结合论文链接，标注来源便于核对。不要编造论文或链接。"
            ),
        }

        response = llm_with_tools.invoke([system] + state["messages"])
        store.put(
            (str(user_id), "memories"),
            key = f"turn_{abs(hash(state['messages'][-1].content))}",
            value = {"text": f"用户问:{state['messages'][-1].content},AI答:{response.content[:100]}"},
        )
        return {"messages": [response]}

    def tools_node(state: State):
        last = state["messages"][-1]
        outputs = []
        for call in last.tool_calls:
            fn = tool_map.get(call["name"])
            if fn is None:
                result = f"错误:没有名为{call['name']}的工具"
            else:
                try:
                    result = fn(**call["args"])
                except Exception as e:
                    result = f"工具执行失败:{e}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    def should_continue(state: State):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "chatbot")

    # return graph_builder.compile(checkpointer=MemorySaver())
    return graph_builder.compile(checkpointer=checkpointer, store=store)


import os
# 确保存放数据库的目录存在
os.makedirs("data", exist_ok=True)

# 用 ExitStack 管理生命周期，这样 Uvicorn 关闭时能自动释放数据库连接
stack = ExitStack()

# 1. 初始化对话历史记忆库（短期）
checkpointer = stack.enter_context(SqliteSaver.from_conn_string("data/conversations.db"))

# 2. 初始化长期记忆库（跨会话）
store = stack.enter_context(SqliteStore.from_conn_string("data/memory.db"))

# 3. 把它们传给图
graph = build_graph(checkpointer=checkpointer, store=store)


class AskRequest(BaseModel):
    question: str
    thread_id: str = 'web-user'
    image_base64: str | None = None  # 可选，用户上传的图片，base64 编码


# 启动时建表
init_db()


@app.post("/register")
async def register(req: AuthRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, req.username, req.password)
    except ValueError as e:
        return {"error": str(e)}
    return {"token": create_token(user.id), "username": user.username}


@app.post("/login")
async def login(req: AuthRequest, db: Session = Depends(get_db)):
    try:
        user = login_user(db, req.username, req.password)
    except ValueError as e:
        return {"error": str(e)}
    return {"token": create_token(user.id), "username": user.username}


@app.get("/me")
async def me(user_id: int = Depends(get_user_id), db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        return {"error": "用户不存在"}
    return {"id": user.id, "username": user.username}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    total = 0
    try:
        total = request.headers.get("authorization", "")
        if total.startswith("Bearer "):
            from auth import decode_token
            user_id = decode_token(total[7:])
            total = count_documents(user_id)
    except Exception:
        pass    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"total_chunks": total},
    )

@app.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
    )

@app.post('/ask')
async def ask(req: AskRequest,user_id: int = Depends(get_user_id)):
    final_question = req.question

    if req.image_base64:
        try:
            zhipu_key = os.getenv("ZHIPU_API_KEY")
            # print(f"DEBUG: 拿到的 Zhipu Key 是 -> [{zhipu_key}]")
            
            client = ZhipuAI(api_key=f'{zhipu_key}')
            base64_data = req.image_base64.split(",")[-1] if "," in req.image_base64 else req.image_base64

            response = client.chat.completions.create(
                model = "glm-4.6v-flash",
                messages = [{
                    "role": "user",
                    "content": [
                        {"type":"image_url","image_url": {"url":base64_data}},
                        {"type":"text","text": "请详细提取并描述这张图片中的内容。如果是图表或公式，请尽可能输出结构化文本。"}
                    ]
                }]
            )
            image_desc = response.choices[0].message.content
            final_question = f"用户上传了一张图片，内容描述如下：{image_desc}\n\n【用户基于此图的问题是】:{req.question}"

        except Exception as e:  
            return {"answer": f"图片识别失败，请检查智谱 API Key:{e}"}

    config = {'configurable': {'thread_id': f"u{user_id}-{req.thread_id}","user_id": user_id}}
    result = graph.invoke(
        {'messages': [{'role': 'user', 'content': final_question}]},
        config=config,
    )
    return {'answer': result['messages'][-1].content}


@app.post('/upload')
async def upload(file: UploadFile = File(...), user_id: int = Depends(get_user_id)):
    doc_dir = os.path.join(BASE_DIR, 'knowledge', 'documents', str(user_id))
    os.makedirs(doc_dir, exist_ok=True)


    safe_name = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in file.filename) or 'document'
    file_path = os.path.join(doc_dir, safe_name)

    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    ext = os.path.splitext(safe_name)[1].lstrip('.').lower() 

    # if ext in ['png', 'jpg', 'jpeg', 'webp']:
    #     try:
    #         with open(file_path, 'rb') as f:
    #             base64_data = base64.b64encode(f.read()).decode('utf-8')

    #         client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
    #         response = client.chat.completions.create(
    #             model = "glm-4.6v-flash",
    #             messages = [{
    #                 "role": "user",
    #                 "content": [
    #                     {"type":"image_url","image_url": {"url":base64_data}},
    #                     {"type":"text","text": "请详细提取并描述这张图片中的内容。如果是图表或公式,请尽可能输出Markdown结构化文本。"}
    #                 ]
    #             }]
    #         )
    #         image_text = response.choices[0].message.content

    #     except Exception as e:
    #         return {"error": f"图片识别失败:{e}"}

    #     md_name = f"image_{abs(hash(safe_name)) % 100000}.md"
    #     md_path = os.path.join(doc_dir, md_name)
    #     with open(md_path, 'w', encoding='utf-8') as f:
    #         f.write(f"# 图片内容提取：{safe_name}\n\n{image_text}")

    ok, msg = build_index(file_path, safe_name, ext, user_id=user_id)
    if not ok:
        return {"error": msg}
    invalidate_cache(user_id)
    return {"message": msg, "doc": safe_name, "type": ext}

    ok, msg = build_index(file_path, safe_name, ext,user_id=user_id)
    if not ok:
        return {"error": msg}
    # 分析文档内容（从文档提取文本做摘要）
    invalidate_cache(user_id)
    analysis = {}
    try:
        from kb_loader import load_document
        blocks = load_document(file_path)
        combined = "\n".join(b["content"] for b in blocks)
        if combined.strip():
            analysis = analyze_document(safe_name, combined)
    except Exception:
        analysis = {}
    return {"message": msg, "doc": safe_name, "type": ext, "analysis": analysis}


@app.post('/analyze')
async def analyze(file: UploadFile = File(...), user_id: int = Depends(get_user_id)):
    """分析但不入库：返回摘要、关键词、分类。"""
    tmp = os.path.join(BASE_DIR, "knowledge", "tmp", str(user_id))
    os.makedirs(tmp, exist_ok=True)
    safe_name = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in file.filename) or "doc"
    tmp_path = os.path.join(tmp, safe_name)
    with open(tmp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        from kb_loader import load_document
        blocks = load_document(tmp_path)
        combined = "\n".join(b["content"] for b in blocks)
        analysis = analyze_document(safe_name, combined)
        return analysis
    except Exception as e:
        return {"title": safe_name, "error": str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class LearnRequest(BaseModel):
    url: str
    title: str = ""


@app.post('/learn')
async def learn(req: LearnRequest, user_id: int = Depends(get_user_id)):
    """通过 URL 抓取网页内容入库（联网学习）。"""
    result = {}
    try:
        msg = learn_from_url_tool(req.url, req.title,user_id=user_id)
        ok = "已学习并入库" in msg
        result = {"ok": ok, "message": msg}
    except Exception as e:
        result = {"ok": False, "message": f"学习失败：{e}"}
    return result


@app.get('/stats')
async def stats(user_id: int = Depends(get_user_id)):
    try:
        total = count_documents(user_id = user_id)
    except Exception:
        import traceback
        traceback.print_exc()
        total = 0
    return {
        "total_chunks": total,
        "knowledge_dir": os.path.join(BASE_DIR, 'knowledge', 'documents',str(user_id)),
    }