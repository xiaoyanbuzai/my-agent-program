import os
os.environ["HF_HUB_OFFLINE"] = "1" #已下载模型 直接用本地的 未下载的话就要先下载了

from langchain_chroma import Chroma
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
# from langchain_huggingface import HuggingFaceCrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import jieba

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)

# vectorstore = Chroma(persist_directory = "./chroma_db", embedding_function = embeddings)

# retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

def search_paper(query:str, paper_name: str):
    """搜索论文中与查询相关的段落,返回相关内容,混合搜索：向量检索（语义）+ BM25 关键词检索（字面匹配）"""
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=paper_name,
    )
    base_retriever = db.as_retriever(search_kwargs={"k": 20})

    all_docs = db.get(limit=10000)['documents']
    bm25_retriever = BM25Retriever.from_texts(all_docs,preprocess_func=lambda x: list(jieba.cut(x)))  # 中文分词关键
    bm25_retriever.k = 20
    
    ensemble = EnsembleRetriever(retrievers=[base_retriever, bm25_retriever], weights=[0.5, 0.5])
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
    # docs = db.similarity_search(query, k=5)
    docs = compression_retriever.invoke(query)  # 精排出 3 个
    
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