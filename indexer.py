"""统一索引构建：解析文档 → 切块 → 写入用户隔离的向量库。"""

import os
import base64
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import re

load_dotenv()  # 从 .env 文件加载环境变量

# from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pymupdf

from vector_store import get_user_collection

os.environ.setdefault("HF_HUB_OFFLINE", "1")

def build_chunks(pdf_path: str):
    """读取 PDF,返回 [{"title", "content"}, ...]。"""
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
    """按一级章节(level 1)切块，每个章节一个块（太长则内部再切）。"""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    # level1 章节边界（页码为 PDF 中从 1 起始的页码）
    level1 = [(title, page) for level, title, page in toc if level == 1]
    if not level1:
        # 没有一级标题则回退到全文按字符切
        return split_by_chars(doc)

    total_pages = doc.page_count
    for i, (title, page_num) in enumerate(level1):
        start_page = page_num - 1  # 转 0-based
        end_page = (level1[i + 1][1] - 1) if i + 1 < len(level1) else total_pages
        start_page = max(0, min(start_page, total_pages))
        end_page = max(start_page, min(end_page, total_pages))

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

def is_valid_chunk(text: str) -> str:
    """判断切分出的子块是否有效（过滤太短或乱码的碎片）"""
    text = text.strip()

    if len(text) < 30:
        return False

    valid_chars =  len(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]',text))
    if len(text) >0 and (valid_chars / len(text)) < 0.4:
        return False

    return True

def build_index(file_path: str, doc_name: str, doc_type: str , user_id: int) -> tuple[bool, str]:
    """把文档建成父子分块索引。
    - 子块(Child)用于向量检索。
    - 父块(Parent)存入 metadata,用于提供给大模型生成回答。

    Args:
        file_path: 待入库文件路径(pdf/docx/md/txt)
        doc_name:  文档显示名（通常取文件名）
        doc_type:  文档类型标签，用于检索过滤，默认取自扩展名
        user_id:  用户ID,用于隔离存储
    """
    if not os.path.exists(file_path):
        return False, f"文件不存在 {file_path}"

    # 根据扩展名推断类型
    if not doc_type:
        doc_type = os.path.splitext(file_path)[1].lstrip(".").lower() or "doc"

    blocks = []

    if doc_type in ["png", "jpg", "jpeg","webp"]:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            return False, "未配置 ZHIPUAI_API_KEY,无法解析图片"

        try:
            with open(file_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

            client = ZhipuAI(api_key=f'{api_key}')
            response = client.chat.completions.create(
                model="glm-4.6v-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type":"image_url","image_url":{"url":base64_data}},
                        {"type":'text',"text":"请详细提取这张图片中的文字、公式或图表数据。请以 Markdown 格式输出，保留结构。"}]}],
                        timeout=30
            )
            image_text = response.choices[0].message.content
            blocks = [{"title":"图片解析内容","content":image_text}]
        except Exception as e:
            return False,f"图片识别失败:{e}"

    else:
        try:
            from kb_loader import load_document
            blocks = load_document(file_path)
        except Exception as e:
            return False, f"文档解析失败：{e}"

    if not blocks:
        return False, "文档或图片为空或无法解析"

    db = get_user_collection(user_id)

    try:
        existing = db.get(where={"source": doc_name})
        if existing["ids"]:
            db.delete(ids=existing["ids"])

    except Exception as e:
        pass

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    child_texts = []
    child_metadatas = []
    for i,block in enumerate(blocks):
        parent_content = block["content"]
        title = block.get("title", "") 

        if(len(parent_content) < 1000):
            children = [parent_content]
        else:
            children = child_splitter.split_text(parent_content)

        for j,child_text in enumerate(children):
            if not is_valid_chunk(child_text):
                continue

            child_texts.append(child_text)
            child_metadatas.append({
                "user_id": user_id,
                "source": doc_name,
                "doc_type": doc_type,
                "title": title,
                "child_index": f"{i}_{j}",
                "parent_content": parent_content  # 截断存储，避免过长
            })

    if child_texts:
        try:
            db.add_texts(texts=child_texts, metadatas=child_metadatas)
        except Exception as e:
            return False, f"写入向量库失败：{e}"
    else:
        return False, "没有提取到有效文本块"

    return True, f"《{doc_name}》已清洗入库包含 {len(blocks)} 个父块，{len(child_texts)} 个子块，类型 {doc_type})"
   

    # # 多格式解析（延迟导入避免与 kb_loader 循环依赖）
    
    # texts = [f"[{c['title']}]\n{c['content']}" for c in chunks]
    # metadatas = [{"user_id": user_id, "source": doc_name, "doc_type": doc_type, "title": c.get("title", "")} for c in chunks]

    # # 写入用户隔离的向量库
    # try:
    #     db = get_user_collection(user_id)
    # except Exception as e:
    #     return False, f"打开向量库失败：{e}"

    # # 先删旧：同一文件重复上传时覆盖
    # try:
    #     existing = db.get(where={"source": doc_name})
    #     if existing["ids"]:
    #         db.delete(ids=existing["ids"])
    # except Exception:
    #     pass

    # try:
    #     db.add_texts(texts=texts, metadatas=metadatas)
    # except Exception as e:
    #     return False, f"写入向量库失败：{e}"

    # return True, f"《{doc_name}》已入库（{len(chunks)} 个块，类型 {doc_type})"


# 兼容旧入口：保留 build_chunks 供教练/测试使用
def build_index_legacy_pdf(pdf_path: str, paper_name: str):
    return build_index(pdf_path, paper_name, "pdf")