import os
from retriever import search_paper
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# load_dotenv()
def make_tools(paper_name:str):
    """根据当前论文名，生成工具列表和映射表。"""
    def search_current(query:str) -> str:
        """搜索当前论文中与查询相关的段落。"""
        # print("DEBUG paper_name:", paper_name, type(paper_name))
        return search_paper(query, paper_name)
    tools = [search_current]
    tool_map = {t.__name__: t for t in tools}
    return tools, tool_map

# Tools = [search_paper]
# Tool_MAP = {t.__name__: t for t in Tools}

# llm = ChatOpenAI( model="deepseek-v4-flash",
#     base_url="https://api.deepseek.com/v1",)

# llm_with_tools = llm.bind_tools(Tools)