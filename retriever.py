import os
from langchain_chroma import Chroma
from langchain_huggingface import  HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# vectorstore = Chroma(persist_directory = "./chroma_db", embedding_function = embeddings)

# retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

def search_paper(query:str, paper_name: str):
    """搜索论文中与查询相关的段落,返回相关内容"""
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=paper_name,
    )
    docs = db.similarity_search(query, k=5)
    if not docs:
        return "未找到相关内容"
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# def search_paper(query: str) -> str:
#     """搜索论文中与查询相关的段落。"""
#     return (
#         "论文方法部分：本文提出了一种基于注意力机制的序列到序列模型。"
#         "编码器由 6 层堆叠的自注意力层和前馈网络组成，解码器同样为 6 层，"
#         "并在每层加入交叉注意力。训练使用 Adam 优化器，学习率采用 warmup 策略。"
#     )