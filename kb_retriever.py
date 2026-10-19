"""通用知识库全局检索器：对统一的 kb_documents collection 做混合检索（向量 + BM25 + Rerank）。

与应用层解耦，供问答、评估等场景复用。缓存单例，避免重复构建索引。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 已下载模型，直接用本地

from vector_store import get_user_collection, embeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import jieba

# COLLECTION = "kb_documents"

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)

_context_cache = {}


def _get_context(user_id: int):
    """懒加载并缓存全局知识库的检索上下文(db、bm25、compression)。"""
    if user_id not in _context_cache:
        db = get_user_collection(user_id)   

        base_retriever = db.as_retriever(search_kwargs={"k": 20})

        all_docs = db.get(limit=100000)["documents"]
        if not all_docs:
            _context_cache[user_id] = {"db": db, "compression": None}
            return _context_cache[user_id]
        bm25_retriever = BM25Retriever.from_texts(
            all_docs,
            preprocess_func=lambda x: list(jieba.cut(x)),
        )
        bm25_retriever.k = 20

        ensemble = EnsembleRetriever(
            retrievers=[base_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        compression= ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble,
        )
        _context_cache[user_id] = {
            "db": db,
            "compression": compression,
        }
    # global _context_cache
    # if _context_cache is None:
    #     db = Chroma(
    #         persist_directory="./chroma_db",
    #         embedding_function=embeddings,
    #         collection_name=COLLECTION,
    #     )
    #     base_retriever = db.as_retriever(search_kwargs={"k": 20})

    #     all_docs = db.get(limit=100000)["documents"]
    #     bm25_retriever = BM25Retriever.from_texts(
    #         all_docs,
    #         preprocess_func=lambda x: list(jieba.cut(x)),
    #     )
    #     bm25_retriever.k = 20

    #     ensemble = EnsembleRetriever(
    #         retrievers=[base_retriever, bm25_retriever],
    #         weights=[0.5, 0.5],
    #     )
    #     compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=reranker,
    #         base_retriever=ensemble,
    #     )
    #     _context_cache = {
    #         "db": db,
    #         "bm25": bm25_retriever,
    #         "compression": compression_retriever,
    #     }
    return _context_cache[user_id]


def search_kb(query: str, user_id: int, top_k: int = 4) -> list[dict]:
    """按用户检索，返回完整父块 [{source, doc_type, title, content}, ...]。

    Args:
        query:    查询语句
        user_id:  用户ID
        top_k:    返回的块数量
    """
    ctx = _get_context(user_id)
    if ctx["compression"] is None:
        return []
    docs = ctx["compression"].invoke(query)

    results = []

    seen_parents = set() # 用于父块去重指纹
    for d in docs[:top_k]:
        meta = d.metadata or {}
        parent_text = meta.get("parent_content", d.page_content)
        fingerprint = hash(parent_text[:100])
        if fingerprint in seen_parents:
            continue
        seen_parents.add(fingerprint)

        source = meta.get("source", "未知来源")
        d_type = meta.get("doc_type", "")
        title = meta.get("title", "")
        # if doc_type and d_type and d_type != doc_type:
        #     continue
        results.append({
            "source": source,
            "doc_type": d_type,
            "title": title,
            "content": parent_text,
        })
        # 达到需要的父块数量，停止提取
        if len(results) >= top_k:
            break
    return results


def count_documents(user_id: int) -> int:
    """返回该用户知识库当前已入库的块数量。"""
    try:
        # return _get_context(user_id)["db"]._collection.count()
        ctx = get_user_collection(user_id)
        result = ctx.get(limit=100000)
        print(f"DEBUG count_documents: user_id={user_id}, ids={len(result.get('ids', []))}")
        return len(result.get("ids", []))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0

def invalidate_cache(user_id: int):
    """上传新文档后调用，让检索器重新构建索引。"""
    _context_cache.pop(user_id, None)