"""上传与文件名安全测试。"""

from __future__ import annotations

import io

import pytest

from uploads import (
    UploadValidationError,
    decode_image_payload,
    sanitize_filename,
    save_stream_limited,
    sniff_image_type,
    validate_extension,
    validate_image_bytes,
)

ALLOWED = frozenset({"pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"})
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def test_sanitize_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename(r"C:\Windows\evil.pdf") == "evil.pdf"


def test_sanitize_keeps_chinese_and_rejects_reserved_names():
    assert sanitize_filename("论文 最终版.pdf") == "论文 最终版.pdf"
    assert sanitize_filename("CON.txt") == "document.txt"


def test_validate_extension():
    assert validate_extension("a.PDF", ALLOWED) == "pdf"
    with pytest.raises(UploadValidationError):
        validate_extension("a.exe", ALLOWED)
    with pytest.raises(UploadValidationError):
        validate_extension("no-extension", ALLOWED)


def test_save_stream_limited_enforces_size(tmp_path):
    destination = tmp_path / "big.bin"
    with pytest.raises(UploadValidationError):
        save_stream_limited(io.BytesIO(b"x" * 100), destination, max_bytes=10)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


def test_save_stream_limited_writes(tmp_path):
    destination = tmp_path / "ok.bin"
    written = save_stream_limited(io.BytesIO(b"hello"), destination, max_bytes=10)
    assert written == 5
    assert destination.read_bytes() == b"hello"


def test_image_sniffing_and_validation():
    assert sniff_image_type(PNG) == "png"
    assert sniff_image_type(JPEG) == "jpg"
    assert sniff_image_type(b"not-an-image") is None

    assert validate_image_bytes(PNG, max_bytes=1024) == "png"
    with pytest.raises(UploadValidationError):
        validate_image_bytes(b"GIF89a", max_bytes=1024)


def test_decode_image_payload():
    import base64

    payload = "data:image/png;base64," + base64.b64encode(PNG).decode()
    data, ext = decode_image_payload(payload, max_bytes=1024)
    assert data == PNG
    assert ext == "png"

    with pytest.raises(UploadValidationError):
        decode_image_payload("data:text/plain;base64,aGVsbG8=", max_bytes=1024)
    with pytest.raises(UploadValidationError):
        decode_image_payload("不是 base64", max_bytes=1024)
