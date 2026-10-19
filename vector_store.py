"""向量库封装：按 user_id 隔离存储。"""
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

# HF 离线模式，用本地缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data","chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 每个用户一个 Chroma 实例，缓存住，避免重复加载
_cache = {}

def _get_chroma(user_id: int) -> Chroma:
    if user_id not in _cache:
        path = os.path.join(DATA_DIR, str(user_id))
        os.makedirs(path, exist_ok=True)
        _cache[user_id] = Chroma(
            persist_directory=path, 
            embedding_function=embeddings,
            collection_name="kb_documents",)
    return _cache[user_id]

def get_user_collection(user_id: int) -> Chroma:
    """供外部调用的入口。"""
    return _get_chroma(user_id)

def delete_user_collection(user_id: int):
    """删除用户的向量库。"""
    path = os.path.join(DATA_DIR, str(user_id))
    if os.path.exists(path):
        shutil.rmtree(path)
    _cache.pop(user_id, None)