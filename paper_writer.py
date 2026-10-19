"""论文 Word 生成：把 Markdown 论文稿排成中文期刊/学位论文常用格式。

版面约定：
- A4，上下 2.54cm、左右 2.54cm；
- 标题黑体二号居中，作者/单位宋体居中；
- 摘要、关键词为“标签 + 正文”段落；
- 一级标题黑体四号，二级标题黑体小四；
- 正文宋体小四、1.5 倍行距、首行缩进 2 字符、两端对齐；
- 参考文献宋体五号，悬挂缩进；
- 页脚居中页码。

输入是 Markdown：``# 标题``、``## 摘要``、``## 关键词``、``## 1 引言``、``### 1.1 ...``、
``## 参考文献``；表格用管道表格，列表用 ``-`` / ``1.``。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from config import user_dir
from uploads import sanitize_filename, unique_path

logger = logging.getLogger(__name__)

MAX_PAPER_CHARS = 80_000
BODY_FONT = "宋体"
HEADING_FONT = "黑体"
LATIN_FONT = "Times New Roman"

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^\d+[.)、]\s+(.*)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_INLINE_RE = re.compile(r"(\*\*.+?\*\*|`.+?`|\*[^*]+\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_SOURCE_MARK_RE = re.compile(r"[\[（(]来源[:：][^\]）)]*[\]）)]")


def _set_run_font(
    run,
    *,
    name: str = BODY_FONT,
    size: float = 12,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor | None = None,
) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color

    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), LATIN_FONT if name == BODY_FONT else name)
    rfonts.set(qn("w:hAnsi"), LATIN_FONT if name == BODY_FONT else name)


def _add_inline_runs(
    paragraph,
    text: str,
    *,
    name: str = BODY_FONT,
    size: float = 12,
    bold: bool = False,
    italic: bool = False,
) -> None:
    text = _LINK_RE.sub(lambda match: f"{match.group(1)}（{match.group(2)}）", text)
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        run_bold = bold
        run_italic = italic
        content = part
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            content = part[2:-2]
            run_bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            content = part[1:-1]
            run_bold = False
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            content = part[1:-1]
            run_italic = True
        run = paragraph.add_run(content)
        _set_run_font(
            run,
            name=name,
            size=size,
            bold=run_bold,
            italic=run_italic,
        )


def _paragraph(
    document,
    text: str = "",
    *,
    first_indent: bool = True,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    size: float = 12,
    name: str = BODY_FONT,
    bold: bool = False,
    space_before: float = 0,
    space_after: float = 0,
    line_spacing: float = 1.5,
):
    paragraph = document.add_paragraph()
    paragraph.alignment = align
    fmt = paragraph.paragraph_format
    fmt.line_spacing = line_spacing
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    if first_indent:
        fmt.first_line_indent = Pt(24)
    else:
        fmt.first_line_indent = Pt(0)
    if text:
        _add_inline_runs(paragraph, text, name=name, size=size, bold=bold)
    return paragraph


def _heading(document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style="Heading 1" if level <= 2 else "Heading 2")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = paragraph.paragraph_format
    fmt.first_line_indent = Pt(0)
    fmt.space_before = Pt(14 if level <= 2 else 10)
    fmt.space_after = Pt(6 if level <= 2 else 4)
    fmt.keep_with_next = True
    size = 14 if level <= 2 else 12
    _add_inline_runs(paragraph, text, name=HEADING_FONT, size=size, bold=True)
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)


def _add_page_number(document) -> None:
    footer = document.sections[0].footer
    footer.is_linked_to_previous = False
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    _set_run_font(run, size=9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    field_run = OxmlElement("w:r")
    field_text = OxmlElement("w:t")
    field_text.text = "1"
    field_run.append(field_text)
    field.append(field_run)
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    _set_run_font(run, size=9)


def _setup_document(title: str, author: str) -> Document:
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.5)

    normal = document.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(12)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), BODY_FONT)

    document.core_properties.title = title
    document.core_properties.author = author
    document.core_properties.subject = "学术论文"
    return document


def _add_title_block(document, title: str, author: str, affiliation: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(8)
    _add_inline_runs(paragraph, title, name=HEADING_FONT, size=22, bold=True)

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    _add_inline_runs(paragraph, author or "作者", size=12)

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(16)
    _add_inline_runs(
        paragraph,
        affiliation or "（单位名称，城市 邮编）",
        size=10.5,
        italic=False,
    )
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def _add_labeled_paragraph(document, label: str, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.line_spacing = 1.5
    fmt.first_line_indent = Pt(0)
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(4)
    run = paragraph.add_run(label)
    _set_run_font(run, name=HEADING_FONT, size=12, bold=True)
    _add_inline_runs(paragraph, text, size=12)


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        stripped = line.strip().strip("|")
        if not stripped:
            continue
        rows.append([cell.strip() for cell in stripped.split("|")])
    return rows


def _add_table(document, rows: list[list[str]]) -> None:
    if len(rows) < 2:
        return
    headers = rows[0]
    body = rows[2:] if len(rows) > 1 and _TABLE_SEPARATOR_RE.match("|".join(rows[1])) else rows[1:]
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_inline_runs(paragraph, header, name=HEADING_FONT, size=10.5, bold=True)

    for row in body:
        cells = table.add_row().cells
        for index, value in enumerate(row[: len(headers)]):
            cells[index].text = ""
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _add_inline_runs(paragraph, value, size=10.5)
    document.add_paragraph()


def _render_blocks(document, lines: list[str], *, reference: bool = False) -> None:
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(part.strip() for part in buffer if part.strip())
        buffer.clear()
        if text:
            _paragraph(document, text)

    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush()
            index += 1
            continue

        if reference:
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt = paragraph.paragraph_format
            fmt.line_spacing = 1.25
            fmt.first_line_indent = Pt(0)
            fmt.left_indent = Pt(0)
            fmt.space_after = Pt(2)
            _add_inline_runs(paragraph, stripped, size=10.5)
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[index + 1].strip()):
            flush()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            _add_table(document, _parse_table(table_lines))
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            flush()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt = paragraph.paragraph_format
            fmt.line_spacing = 1.5
            fmt.left_indent = Pt(24)
            fmt.first_line_indent = Pt(-12)
            fmt.space_after = Pt(0)
            run = paragraph.add_run("• ")
            _set_run_font(run, size=12)
            _add_inline_runs(paragraph, bullet.group(1), size=12)
            index += 1
            continue

        numbered = _NUMBER_RE.match(stripped)
        if numbered:
            flush()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt = paragraph.paragraph_format
            fmt.line_spacing = 1.5
            fmt.left_indent = Pt(24)
            fmt.first_line_indent = Pt(-12)
            fmt.space_after = Pt(0)
            marker = stripped[: stripped.find(numbered.group(1))].strip()
            run = paragraph.add_run(marker + " ")
            _set_run_font(run, size=12)
            _add_inline_runs(paragraph, numbered.group(1), size=12)
            index += 1
            continue

        buffer.append(stripped)
        index += 1

    flush()


def _split_sections(content: str) -> list[dict]:
    sections: list[dict] = []
    current = {"level": 0, "title": "", "lines": []}
    for line in content.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            if current["title"] or any(item.strip() for item in current["lines"]):
                sections.append(current)
            current = {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "lines": [],
            }
        else:
            current["lines"].append(line)
    if current["title"] or any(item.strip() for item in current["lines"]):
        sections.append(current)
    return sections


def _section_text(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return _SOURCE_MARK_RE.sub("", text).strip()


def build_paper_docx(
    output_path: str | Path,
    *,
    title: str,
    content: str,
    author: str = "",
    affiliation: str = "",
) -> Path:
    """把 Markdown 论文稿写成 DOCX，返回输出路径。"""
    title = (title or "").strip() or "学术论文"
    content = (content or "").strip()
    if not content:
        raise ValueError("论文内容不能为空")
    if len(content) > MAX_PAPER_CHARS:
        raise ValueError(f"论文内容过长（上限 {MAX_PAPER_CHARS} 字）")

    # 知识库检索标记不属于论文正文；统一清掉，引用改由文末参考文献承担。
    content = _SOURCE_MARK_RE.sub("", content).strip()

    content_lines = content.splitlines()
    # 第一个一级标题通常就是论文标题；避免正文重复渲染。
    if content_lines:
        first_heading = _HEADING_RE.match(content_lines[0].strip())
        if first_heading and len(first_heading.group(1)) == 1:
            if title == "学术论文":
                title = first_heading.group(2).strip() or title
            content_lines = content_lines[1:]

    document = _setup_document(title, author)
    _add_title_block(document, title, author, affiliation)
    sections = _split_sections("\n".join(content_lines))

    rendered_sections = 0
    for section in sections:
        section_title = section["title"].strip()
        lowered = section_title.lower()
        text = _section_text(section["lines"])

        if not section_title:
            _render_blocks(document, section["lines"])
            rendered_sections += 1
            continue

        if "摘要" in section_title or lowered in {"abstract", "summary"}:
            label = "摘要：" if "摘要" in section_title else "Abstract: "
            text = re.sub(r"^(摘要|abstract|summary)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
            _add_labeled_paragraph(document, label, text)
            rendered_sections += 1
            continue

        if "关键词" in section_title or lowered.startswith("keyword"):
            label = "关键词：" if "关键词" in section_title else "Keywords: "
            text = re.sub(r"^(关键词|keywords?)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
            _add_labeled_paragraph(document, label, text)
            rendered_sections += 1
            continue

        _heading(document, section_title, section["level"] or 2)
        _render_blocks(
            document,
            section["lines"],
            reference="参考文献" in section_title or lowered in {"references", "bibliography"},
        )
        rendered_sections += 1

    if rendered_sections == 0:
        _render_blocks(document, content.splitlines())

    _add_page_number(document)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


def generate_paper_docx(
    user_id: int,
    *,
    title: str,
    content: str,
    author: str = "",
    affiliation: str = "",
) -> tuple[Path, str]:
    """生成到租户目录，返回 (路径, 文件名)。"""
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("用户信息无效")
    stem = sanitize_filename(title or "论文", fallback="paper")
    stem = re.sub(r"\.(docx?|pdf)$", "", stem, flags=re.IGNORECASE) or "paper"
    directory = user_dir("documents", user_id) / "papers"
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_path(directory, f"{stem}.docx")
    build_paper_docx(
        path,
        title=title,
        content=content,
        author=author,
        affiliation=affiliation,
    )
    return path, path.name
