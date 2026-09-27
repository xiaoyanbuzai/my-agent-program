import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

def build_index(pdf_path: str, persist_dir: str = "./chroma_db"):
    #   读取 PDF
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}")
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
    )
    chunks = splitter.split_text(full_text)
    print(f"文本切块完成,总块数: {len(chunks)}")

    # 创建向量存储Chroma
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    db = Chroma.from_texts(
        texts = chunks,
        embedding = embeddings,
        persist_directory = persist_dir,
    )
    db.persist()
    print(f"向量存储Chroma创建完成,数据已保存到: {persist_dir}")

if __name__ == "__main__":
    pdf_path = input("请输入PDF文件路径(例如 papers/xxx.pdf): ").strip()
    build_index(pdf_path)