import os
from retriever import search_paper
from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

Tools = [search_paper]
Tool_MAP = {t.__name__: t for t in Tools}

# llm = ChatOpenAI( model="deepseek-v4-flash",
#     base_url="https://api.deepseek.com/v1",)

# llm_with_tools = llm.bind_tools(Tools)