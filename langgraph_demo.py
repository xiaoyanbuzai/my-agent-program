from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
#    记忆 字典 

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
# 图构建器 起点 终点 追加列表信息   

from langchain_openai import ChatOpenAI
#  ai接口   

from dotenv import load_dotenv
# 读取 .env 文件里的 API Key 和 base_url

#   读取工具    
from langgraph_tools_demo import Tool_MAP,Tools
from retriever import search_paper
#   工具信息    
from langchain_core.messages import ToolMessage

#   读取 .env 文件里的 API Key 和 base_url
load_dotenv()

#   建模型  
llm = ChatOpenAI( model="deepseek-flash",
    base_url="https://api.deepseek.com/v1",)

#   定义状态结构
class State(TypedDict):
    messages: Annotated[list,add_messages]

llm_with_tools = llm.bind_tools(Tools)

#   机器人回应
def chatbot(state: State):
    system = {
        "role":"system",
        "content":(
            "你是一个文献助手。用户的问题如果涉及论文内容，"
            "你必须先调用 search_paper 工具检索，再基于检索结果回答。"
            "如果用户没有明确指哪篇论文，默认指最近讨论的那篇。"
        )
    }
    messages = [system] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

# def chatbot(state:State):
#     return {"mseeages":[llm_with_tools.invoke(state["messages"])]}

#   工具调用
def tool_node(state:State):
    last = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        fn = Tool_MAP[call["name"]]
        result = fn(**call["args"])
        outputs.append(ToolMessage(content = str(result),tool_call_id = call["id"]))
    return {"messages": outputs}

#   判断
def should_continue(state: State):
    last = state["messages"][-1]
    if  getattr(last,"tool_calls",None):
        return "tools"
    return END

#   总结节点
def summarize(state:State):
    queries = ["研究问题","方法","数据集","主要结论","局限性"]
    context_parts = []
    for q in queries:
        context_parts.append(f"【{q}】\n{search_paper(q)}")
    context = "\n\n".join(context_parts)
    system = {
        "role": "system",
        "content": (
            "你是一个文献摘要助手。基于下面的论文片段，"
            "严格按以下格式输出：\n\n"
            "## 研究问题\n\n## 方法\n\n## 数据集\n\n## 主要结论\n\n## 局限性\n\n"
            "如果某部分论文中未提及，写“论文未w提及”。\n\n"
            f"论文片段：\n{context}"
        ),
    }
    recent = state["messages"][-15:]  # 只取最近的15条消息，避免上下文过长
    return {"messages":[llm.invoke([system] + recent)]}

#   批判
def critique(state:State):
    context_parts = []
    for q in ["实验","方法","数据集","结论","局限性"]:
        context_parts.append(f"【{q}】\n{search_paper  (q)}")
    context = "\n\n".join(context_parts)
    system = {
        "role": "system",
        "content": (
            "你是一个严格的论文评审人。基于下面的论文片段，"
            "输出批判性分析，严格按以下格式：\n\n"
            "## 最值得信的一点\n\n## 最值得怀疑的一点\n\n"
            "## 方法上的潜在问题\n\n## 如果我要用它的方法\n\n## 后续可以做什么\n\n"
            "每个部分要有具体依据，引用论文中的数字或设计细节。\n\n"
            f"论文片段：\n{context}"
        ),
    }
    recent = state["messages"][-15:]  # 只取最近的15条消息，避免上下文过长
    return {"messages":[llm.invoke([system] + recent)]}

#   条件边
def route_entry(state:State):
    content = state["messages"][-1].content
    if any(keyword in content for keyword in ["批判","评审","review", "值得怀疑"]):
        return "critique"
    if any(keyword in content for keyword in ["总结","总结一下","总结下","总结论文","摘要","概括"]):
        return "summarize"
    return "chatbot"

#   建图
graph_builder = StateGraph(State)
graph_builder.add_node("tools",tool_node)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("critique", critique)

graph_builder.add_conditional_edges(START, route_entry, {
    "summarize": "summarize",
    "critique": "critique",
    "chatbot": "chatbot",
})
graph_builder.add_conditional_edges("chatbot", should_continue, {
    "tools": "tools",
    END: END,
})
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("summarize", END)
graph_builder.add_edge("critique", END)

with SqliteSaver.from_conn_string("conversation.db") as checkpointer:
    graph = graph_builder.compile()
    config = {"configurable": {"thread_id" : "user-001"}}
    
    #   交互式  
    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ['exit','quit','q']:
                print('退出对话!')
                break
            result = graph.invoke({"messages":[{"role":"user","content":user_input}]} , config = config)
            print("AI:",result["messages"][-1].content)
        except Exception as e:
            print("Error:", e)
            break

# result = graph.invoke({"messages":[{"role":"user","content":"你好"}]})
# print(result["messages"][-1].content)