"""切块与引用元数据测试。"""

from __future__ import annotations

from chunking import is_valid_chunk, split_by_chars, split_by_toc, truncate_text


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class FakeDoc:
    def __init__(self, pages: list[str]) -> None:
        self._pages = [FakePage(text) for text in pages]
        self.page_count = len(self._pages)

    def __getitem__(self, index: int) -> FakePage:
        return self._pages[index]

    def __iter__(self):
        return iter(self._pages)


def test_split_by_chars_keeps_page_number():
    doc = FakeDoc(["第一页内容" * 40, "第二页内容" * 40])
    chunks = split_by_chars(doc)
    assert chunks
    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert all(chunk["modality"] == "text" for chunk in chunks)


def test_split_by_toc_uses_section_title_and_page():
    doc = FakeDoc(["第一章" * 100, "第二章" * 100])
    toc = [[1, "第一章", 1], [1, "第二章", 2]]
    chunks = split_by_toc(doc, toc)
    assert chunks[0]["title"].startswith("第一章")
    assert chunks[0]["page"] == 1
    assert any(chunk["title"].startswith("第二章") for chunk in chunks)


def test_truncate_text_respects_limit():
    text = "句子。" * 100
    truncated = truncate_text(text, 50)
    assert len(truncated) <= 52
    assert truncated.endswith("……")


def test_is_valid_chunk_filters_noise():
    assert not is_valid_chunk("。，、；：")
    assert not is_valid_chunk("短")
    assert is_valid_chunk("这是 一段 有效的 中文 文本，包含 足够 的 字数 用于 检索。" * 2)
