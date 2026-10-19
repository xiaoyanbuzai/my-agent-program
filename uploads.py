"""上传与文件名安全处理：扩展名校验、文件名清洗、限流写盘、图片嗅探。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import BinaryIO

_SAFE_CHARS = re.compile(r"[^\w\u4e00-\u9fff.\- ]+", re.UNICODE)
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
)


class UploadValidationError(ValueError):
    """上传内容不合法（类型、大小、编码）。"""


def sanitize_filename(raw_name: str | None, fallback: str = "document") -> str:
    """保留中文、字母数字和少量安全符号，剥离目录与 Windows 保留名。"""
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKC", name).strip()
    name = _SAFE_CHARS.sub("_", name).strip(" ._")

    stem = Path(name).stem.strip(" ._")
    suffix = Path(name).suffix.lower()
    if suffix and not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ""

    if not stem or stem.upper() in _WINDOWS_RESERVED:
        stem = fallback

    return f"{stem[:80]}{suffix}"


def file_extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def validate_extension(filename: str | None, allowed: frozenset[str]) -> str:
    ext = file_extension(filename)
    if not ext or ext not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise UploadValidationError(
            f"不支持的文件类型：{ext or '（无扩展名）'}；支持：{allowed_text}"
        )
    return ext


def unique_path(directory: Path, filename: str) -> Path:
    """同名文件不覆盖：追加短随机串。"""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    return directory / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def save_stream_limited(
    source: BinaryIO,
    destination: Path,
    max_bytes: int,
    *,
    chunk_size: int = 1024 * 1024,
) -> int:
    """分块写盘并强制大小上限；失败时清理半成品，返回字节数。"""
    if max_bytes <= 0:
        raise UploadValidationError("大小上限配置错误")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")
    total = 0

    try:
        with temp_path.open("wb") as target:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadValidationError(
                        f"文件过大：超过 {max_bytes // (1024 * 1024)} MB 限制"
                    )
                target.write(chunk)
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return total


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sniff_image_type(data: bytes) -> str | None:
    """通过魔数判断图片类型，不信任扩展名 / Content-Type。"""
    for signature, ext in IMAGE_SIGNATURES:
        if data.startswith(signature):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_bytes(data: bytes, max_bytes: int) -> str:
    if not data:
        raise UploadValidationError("图片内容为空")
    if len(data) > max_bytes:
        raise UploadValidationError(
            f"图片过大：超过 {max_bytes // (1024 * 1024)} MB 限制"
        )
    ext = sniff_image_type(data)
    if ext is None:
        raise UploadValidationError("仅支持 PNG / JPEG / WebP 图片")
    return ext


def decode_image_payload(payload: str, max_bytes: int) -> tuple[bytes, str]:
    """解析 data URI 或纯 base64，返回 (bytes, 扩展名)。"""
    if not payload or not payload.strip():
        raise UploadValidationError("图片内容为空")

    text = payload.strip()
    if text.startswith("data:"):
        try:
            header, text = text.split(",", 1)
        except ValueError as exc:
            raise UploadValidationError("图片 data URI 格式错误") from exc
        if ";base64" not in header:
            raise UploadValidationError("图片 data URI 必须使用 base64 编码")

    # 先按长度粗判，避免为超大 payload 分配内存
    if len(text) > (max_bytes * 4 // 3) + 32:
        raise UploadValidationError(
            f"图片过大：超过 {max_bytes // (1024 * 1024)} MB 限制"
        )

    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UploadValidationError("图片 base64 解码失败") from exc

    return data, validate_image_bytes(data, max_bytes)
