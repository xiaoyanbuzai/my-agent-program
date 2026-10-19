"""单文件入库 CLI（调试 / 批量补录用）。

用法：
    uv run python build_index.py papers/1.pdf --user 1
    uv run python build_index.py papers/1.pdf --user 1 --name "论文标题"

批量重建请用 ``scripts/reindex.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import ensure_directories, get_settings
from indexer import build_index
from logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="把单个文件写入指定租户的知识库")
    parser.add_argument("file", type=Path, help="文件路径")
    parser.add_argument("--user", type=int, required=True, help="租户 ID")
    parser.add_argument("--name", default="", help="显示名称，默认取文件名")
    parser.add_argument("--type", default="", help="文档类型，默认按扩展名推断")
    args = parser.parse_args()

    setup_logging()
    ensure_directories()

    if not args.file.is_file():
        print(f"文件不存在：{args.file}")
        return 1

    ext = args.type or args.file.suffix.lstrip(".").lower()
    if ext not in get_settings().allowed_upload_exts:
        print(f"不支持的文件类型：{ext}")
        return 1

    ok, message = build_index(
        str(args.file),
        args.name or args.file.name,
        ext,
        args.user,
    )
    print(message)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
