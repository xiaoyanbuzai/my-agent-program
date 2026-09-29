import os
os.environ["HF_HUB_OFFLINE"] = "1"

from retriever import search_paper
from tools import make_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = chromadb.PersistentClient(path = "./chroma_db")
papers = [c.name for c in client.list_collections()]
paper_name = papers[0]
print(f"评估论文: {paper_name}\n")

Tools,Tool_MAP = make_tools(paper_name)


llm = ChatOpenAI(model = "deepseek-flash")
llm_with_tools = llm.bind_tools(Tools)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def chatbot(state:State):
    system = {
        "role":"system",
        "content":("你是一个文献助手。用户的问题如果涉及论文内容，"
            "你必须先调用工具检索，再基于检索结果回答。")
    }
    return {"messages":[llm_with_tools.invoke([system] + state["messages"])]}

def tool_node(state:State):
    last = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        fn = Tool_MAP.get(call["name"])
        if fn is None:
            result = f"错误:没有名为{call["name"]}的工具"
        else:
            try:
                result = fn(**call["args"])
            except Exception as e:
                result = f"工具执行失败:{e}"
        outputs.append(ToolMessage(content=str(result),tool_call_id = call["id"]))
    return {"messages": outputs}

def should_continue(state:State):
    last = state["messages"][-1]
    if getattr(last,"tool_calls",None):
        return "tools"
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges("chatbot",should_continue,{"tools":"tools",END:END})
graph_builder.add_edge("tools", "chatbot")

checkpointer =  MemorySaver()
graph = graph_builder.compile(checkpointer = checkpointer)

# questions = ["这篇论文要解决的核心问题是什么？",
#     "为什么现有方法难以生成带纹理的 3D 物体？",
#     "论文提出的训练范式与传统方法有什么根本不同？",
#     "论文的生成模型包含哪些隐变量？各自的作用是什么？",
#     "为什么要把 z_shape 和 z_color 分开？",
#     "PUSHING 参数化的核心思想是什么？",
#     "PUSHING 参数化中，线性规划的目标函数和约束是什么？",
#     "论文在合成数据上用了哪些数据集和类别？",
#     "论文在自然图像上用了哪些数据集？",
#     "论文的两种监督设定 (MASK) 和 (NO-MASK) 有什么区别？",
#     "PUSHING 参数化在消除自相交方面的效果如何？",
#     "PUSHING 参数化的代价是什么？",
#     "在自然图像上，(NO-MASK) 设定对 bird 和 car 的表现有什么不同？",
#     "论文自己承认的主要局限性是什么？",
#     "论文在 (NO-MASK) 设定下有什么未解决的问题？",]
questions = [
    "这篇论文要解决的核心问题是什么？",
    "这篇论文提出了什么方法？核心思路是什么？",
    "这篇论文的主要贡献是什么？",
    "这篇论文用了什么实验设置或数据集？",
    "这篇论文的主要结论是什么？",
    "这篇论文有什么局限性？",
    "这篇论文的创新点在哪里？",
    "这篇论文和已有方法相比有什么优势？",
    "这篇论文的方法有什么潜在问题？",
    "这篇论文适合什么背景的人读？",
    "这篇论文最值得信的一点是什么？",
    "这篇论文最值得怀疑的一点是什么？",
    "如果我要用它的方法，最大的障碍是什么？",
    "这篇论文留下了什么未解决的问题？",
    "这篇论文的总体评价是什么？值不值得跟？",
]

results = []
for i,q in enumerate(questions,1):
    print(f"---Q{i}:{q}")
    config = {"configurable":{"thread_id":f"eval-{i}"}}
    try:
        result = graph.invoke(
            {"messages":[{"role":"user","content":q}]},
            config = config)
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"ERROR:{e}"
    print(f"A{i}:{answer[:200]}...\n")
    results.append({"q":q,"a":answer})

with open("eval_results.md","w",encoding = "utf-8") as f:
    for i, r in enumerate(results,1):
        f.write(f"##Q{i}:{r["q"]}\n\n{r['a']}\n\n--\n\n")

print("评估完成，结果已保存到 eval_results.md")