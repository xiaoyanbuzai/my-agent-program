import os
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

def build_index(pdf_path: str, paper_name: str):
    #   读取 PDF
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}")
        return

    #   检查是否存在同名
    client = chromadb.PersistentClient(path = "./chroma_db")
    exisiting = [c.name for c in client.list_collections()]
    if paper_name in exisiting:
        print(f"错误：论文《{paper_name}》的索引已存在，拒绝写入。")
        print(f"如需重建，请先删除：")
        print(f"  uv run python -c \"import chromadb; chromadb.PersistentClient(path='./chroma_db').delete_collection('{paper_name}')\"")
        return

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    if not full_text.strip():
        print("错误: PDF中未提取到文字,可能是扫描版图片")
        return

    print(f"PDF 读取完成,总字符数: {len(full_text)}")

    # 切块  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    print(f"《{paper_name}》读取完成，共 {len(chunks)} 个块")

    # 创建向量存储Chroma
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    # ids = [f"{paper_name}_{i}" for i in range(len(chunks))]
    db = Chroma.from_texts(
        texts = chunks,
        embedding = embeddings,
        persist_directory = "./chroma_db",
        collection_name=paper_name,   # 关键：每篇论文一个 collection
        # ids = ids,
    )

    print(f"《{paper_name}》索引已保存")

if __name__ == "__main__":
    pdf_path = input("请输入PDF文件路径(例如 papers/xxx.pdf): ").strip()
    paper_name = input("论文名称(英文/拼音,用作collection名): ").strip()
    build_index(pdf_path , paper_name)