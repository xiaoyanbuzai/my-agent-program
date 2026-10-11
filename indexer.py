import os
import chromadb
# from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pymupdf

os.environ["HF_HUB_OFFLINE"] = "1"

def build_chunks(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()

    if toc:
        chunks = split_by_toc(doc,toc)
    else:
        chunks = split_by_chars(doc)
    return chunks

def split_by_chars(doc):
    """无书签，全文按字符切"""
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    return [{"title": "unknown", "content": c} for c in chunks]

    
def split_by_toc(doc,toc):
    """按书签切块，每个章节一个块（太长则内部再切）"""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    level1_toc = [(title, page) for level, title, page in toc if level == 1]

    for i, (title, page_num) in enumerate(toc):
        start_page = page_num - 1
        end_page = level1_toc[i + 1][1] - 1 if i + 1 < len(level1_toc) else doc.page_count

        text = ""
        for page in range(start_page, end_page):
            text += doc[page].get_text()

         # 章节太长，内部再切
        if len(text) > 2000:
            sub_chunks = splitter.split_text(text)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "title": f"{title} (part {j+1})",
                    "content": sub,
                })
        else:
            chunks.append({
                "title": title,
                "content": text,
            })
    return chunks


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

    # reader = PdfReader(pdf_path)
    # full_text = ""
    # for page in reader.pages:
    #     text = page.extract_text()
    #     if text:
    #         full_text += text + "\n"

    # if not full_text.strip():
    #     print("错误: PDF中未提取到文字,可能是扫描版图片")
    #     return

    # print(f"PDF 读取完成,总字符数: {len(full_text)}")

        # 切块  
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 300,
    #     separators=["\n\n", "\n", "。", " ", ""],
    # )
    # chunks = splitter.split_text(full_text)
    # print(f"《{paper_name}》读取完成，共 {len(chunks)} 个块")
    chunks = build_chunks(pdf_path)
    if not chunks:
        return False, "PDF 中未提取到文字"

    print(f"《{paper_name}》切块完成，共 {len(chunks)} 个块")
    texts = [f"[{c['title']}]\n{c['content']}" for c in chunks]

    # 创建向量存储Chroma
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    # ids = [f"{paper_name}_{i}" for i in range(len(chunks))]
    db = Chroma.from_texts(
        texts = texts,
        embedding = embeddings,
        persist_directory = "./chroma_db",
        collection_name=paper_name,   # 关键：每篇论文一个 collection
        # ids = ids,
    )

    return True, f"《{paper_name}》索引已保存"