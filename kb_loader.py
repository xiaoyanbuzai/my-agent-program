"""多格式文档解析器：将 PDF / Word / Markdown / txt 解析为纯文本。

统一返回 [(title, text), ...]，每份文档可包含多个"章节块"。
"""
import os
import pymupdf
import re

SUPPORTED_EXT = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def load_document(file_path: str):
    """根据扩展名解析文档，返回 [(title, text), ...]。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的文件类型：{ext}（支持 {sorted(SUPPORTED_EXT)}）")
    if ext == ".pdf":
        blocks = _load_pdf(file_path)
    elif ext in (".docx",):
        blocks = _load_docx(file_path)
    elif ext in (".md", ".markdown"):
        blocks = _load_markdown(file_path)
    elif ext == ".txt":
        blocks = _load_txt(file_path)
    else:
        blocks = []  # 不会到达

    for block in blocks:
        block['content'] = clean_raw_text(block['content'])

    return [b for b in blocks if b['content']]


def _load_pdf(file_path: str):
    """PDF:按一级书签切章，无书签则按字符切。"""
    from indexer import split_by_chars, split_by_toc
    doc = pymupdf.open(file_path)
    toc = doc.get_toc()
    if toc:
        try:
            chunks = split_by_toc(doc, toc)
        except Exception:
            chunks = split_by_chars(doc)
    else:
        chunks = split_by_chars(doc)
    # split_by_toc/split_by_chars 都返回 [{"title","content"}]
    return [{"title": c["title"], "content": c["content"]} for c in chunks]


def _load_docx(file_path: str):
    """Word:按章节标题分块，否则整篇返回。"""
    from docx import Document
    doc = Document(file_path)
    sections = []          # (title, [paragraphs])
    current_title = "正文"
    current_paras = []

    def flush():
        nonlocal current_title, current_paras
        if current_paras and any(p.strip() for p in current_paras):
            text = "\n".join([p for p in current_paras if p.strip()])
            sections.append((current_title, text))
        current_paras = []

    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower() if para.style else ""
        text = para.text.strip()
        if not text:
            continue
        # 粗略判断标题（Heading 样式 或 前3级编号引导）
        is_heading = "heading" in style_name or para.style.style_id in (
            "Heading1", "Heading2", "Heading3", "Title")
        if is_heading:
            flush()
            current_title = text
        else:
            current_paras.append(para.text)
    flush()

    if not sections:
        # 无标题结构，整篇作为一个块
        full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"title": "全文", "content": full}] if full.strip() else []
    return [{"title": t, "content": c} for t, c in sections]


def _load_markdown(file_path: str):
    """Markdown:按一级/二级标题分块，保留纯文本内容。"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # 去除代码块围栏，避免把 ``` 也当文本
    sections = []          # (title, [lines])
    current_title = "文档开头"
    current_lines = []
    in_code = False

    def flush():
        nonlocal current_title, current_lines
        if current_lines and any(l.strip() for l in current_lines):
            text = "".join(current_lines).strip()
            if text:
                sections.append((current_title, text))
        current_lines = []

    for raw in lines:
        stripped = raw.strip()
        # 代码块围栏
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        # 标题（行首 # 开头且不在代码块内）
        if not in_code and stripped.startswith("#"):
            flush()
            # 去掉前面的 # 和空格作为标题名
            level = len(stripped) - len(stripped.lstrip("#"))
            current_title = stripped.lstrip("#").strip()
            continue
        current_lines.append(raw)
    flush()

    if not sections:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full = f.read()
        return [{"title": "全文", "content": full}] if full.strip() else []
    return [{"title": t, "content": c} for t, c in sections]


def _load_txt(file_path: str):
    """txt:整篇返回。"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full = f.read()
    return [{"title": "全文", "content": full}] if full.strip() else []

def clean_raw_text(text: str) -> str:
    """通用文本清洗：去页眉页脚、修断行、去参考文献等"""
    if not text:
        return ""
    text = re.sub(r'-\n(? = [a-zA-Z])','',text)
    text = re.sub(r'^\s*\d+\s*$','',text,flags=re.MULTILINE)
    text = re.sub(r'[-=]{4,}', '', text)
    text = re.sub(r'\n{3,}','\n\n',text)

    return text.strip()

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(f"== {p} ==")
        blocks = load_document(p)
        print(f"  chunks: {len(blocks)}")
        for t, c in blocks[:3]:
            print(f"  - [{t}] {len(c)}字")