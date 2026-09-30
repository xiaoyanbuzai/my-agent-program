import os
os.environ["HF_HUB_OFFLINE"] = "1"

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print("DEBUG cwd:", os.getcwd())

from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb

# print("DEBUG chromadb version:", chromadb.__version__)
# print("DEBUG cwd:", os.getcwd())
# print("DEBUG chroma path:", os.path.join(os.getcwd(), "chroma_db"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

from tools import make_tools
from retriever import search_paper

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

client = chromadb.PersistentClient(path = "./chroma_db")

print("DEBUG client:", client)


collections = client.list_collections()
# print("DEBUG collections:", collections)
# print("DEBUG papers:", [c.name for c in collections])

papers = [c.name for c in client.list_collections()]
# print("DEBUG papers:", papers)   # 加这行

graphs = {}

for paper_name in papers:
    Tools,Tool_MAP = make_tools(paper_name)
    llm = ChatOpenAI(model = 'deepseek-flash')
    llm_with_tools = llm.bind_tools(Tools)

    class State(TypedDict):
        messages:Annotated[list,add_messages]

    def make_chatbot(llm_with_tools):
        def chatbot(state:State):
            system = {"role":'system',
                      "content":"你是一个文献助手。用户的问题如果涉及论文内容，你必须先调用工具检索，再基于检索结果回答。",}
            return {"messages":[llm_with_tools.invoke([system] + state["messages"])]}
        return chatbot

    def make_tools_node(Tool_MAP):  
        def tools_node(state:State):
            last = state["messages"][-1]
            outputs = []
            for call in last.tool_calls:
                fn = Tool_MAP.get(call['name'])
                if fn is None:
                    result = f"错误:没有名为{call['name']}的工具"
                else:
                    try:
                        result = fn(**call['args'])
                    except Exception as e:
                        result = f"工具执行失败 :{e}"
                outputs.append(ToolMessage(content=str(result),tool_call_id = call['id']))
            return {"messages":outputs}
        return tools_node

    def should_continue(state:State):
        last = state["messages"][-1]
        if getattr(last,"tool_calls",None):
            return "tools"
        return END


    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot",make_chatbot(llm_with_tools))
    graph_builder.add_node("tools",make_tools_node(Tool_MAP))
    graph_builder.add_edge(START,'chatbot')
    graph_builder.add_conditional_edges('chatbot',should_continue,{'tools':'tools',END:END})
    graph_builder.add_edge('tools','chatbot')


    checkpointer = MemorySaver()
    graphs[paper_name] = graph_builder.compile(checkpointer=checkpointer)


class AskRequest(BaseModel):
    paper:str
    question:str
    thread_id: str = 'web-user'

@app.get('/',response_class = HTMLResponse)
async def index(request:Request):
    # return templates.TemplateResponse('index.html',{'request':request,'papers':papers})
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"papers": papers},
)

@app.post('/ask')
async def ask(req:AskRequest):
    if req.paper not in graphs:
        return {'answer':f'错误,找不到论文{req.paper}'}

    graph = graphs[req.paper]
    config = {'configurable':{'thread_id':f"{req.paper}-{req.thread_id}"}}

    result = graph.invoke({'messages':[{'role':'user','content':req.question}]},
                          config = config)
    return {'answer':result['messages'][-1].content}