from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["HF_HUB_OFFLINE"] = "1" #已下载模型 直接用本地的 未下载的话就要先下载了
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=emb,
    collection_name="semantic_grayscale_3d_texture",
)

for q in ["cosθ", "d²", "权重函数", "W ="]:
    docs = db.similarity_search(q, k=3)
    print(f"=== 查询: {q} ===")
    for i, d in enumerate(docs):
        print(f"  {i}: {d.page_content[:150]}")
    print()