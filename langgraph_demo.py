from typing import Annotated
from typing_extensions import TypedDict
#    记忆 字典 

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
# 图构建器 起点 终点 追加列表信息   

from langchain_openai import ChatOpenAI
#  ai接口   

from dotenv import load_dotenv
# 读取 .env 文件里的 API Key 和 base_url

#   读取 .env 文件里的 API Key 和 base_url
load_dotenv()

#   建模型  
llm = ChatOpenAI(model = "deepseek-flash")

#   定义状态结构
class State(TypedDict):
    messages: Annotated[list,add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

#   建图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

# result = graph.invoke({"messages":[{"role":"user","content":"你好"}]})
# print(result["messages"][-1].content)

#   交互式  
while True:
    try:
        user_input = input("你: ")
        if user_input.lower() in ['exit','quit','q']:
            print('退出对话!')
            break
        result = graph.invoke({"messages":[{"role":"user","content":user_input}]})
        print("AI:",result["messages"][-1].content)
    except Exception as e:
        print("Error:", e)
        break