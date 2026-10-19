"""重建租户索引：扫描 knowledge/ 下的原始文件，重新切块并写入向量库。

用法：
    uv run python scripts/reindex.py                 # 增量重建所有用户
    uv run python scripts/reindex.py --user 1        # 只重建用户 1
    uv run python scripts/reindex.py --rebuild       # 先删除向量库再全量重建（换嵌入模型后必须用）
    uv run python scripts/reindex.py --dry-run       # 只列出将处理的文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ensure_directories, get_settings, user_dir  # noqa: E402
from indexer import build_index  # noqa: E402
from logging_utils import setup_logging  # noqa: E402
from vector_store import delete_user_collection, get_embedding_model_name  # noqa: E402

SKIP_SUFFIXES = {".part", ".tmp", ".log", ".json"}


def discover_users() -> list[int]:
    settings = get_settings()
    users: set[int] = set()
    for folder in ("documents", "web"):
        base = settings.knowledge_dir / folder
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.isdigit():
                users.add(int(child.name))
    chroma = settings.chroma_dir
    if chroma.is_dir():
        for child in chroma.iterdir():
            if child.is_dir() and child.name.isdigit():
                users.add(int(child.name))
    return sorted(users)


def iter_files(user_id: int):
    for folder in ("documents", "web"):
        directory = user_dir(folder, user_id)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES:
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="重建知识库索引")
    parser.add_argument("--user", type=int, action="append", help="指定用户 ID，可重复")
    parser.add_argument("--rebuild", action="store_true", help="先删除向量库再重建")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件")
    args = parser.parse_args()

    setup_logging()
    ensure_directories()

    users = args.user or discover_users()
    if not users:
        print("没有发现任何用户目录（knowledge/documents/* 或 knowledge/web/*）。")
        return 1

    print(f"嵌入模型：{get_embedding_model_name()}")
    print(f"待处理用户：{users}")

    total_ok = 0
    total_failed = 0

    for user_id in users:
        files = list(iter_files(user_id))
        if not files:
            print(f"[user {user_id}] 没有找到原始文件，跳过")
            continue

        # 先确认有文件可重建，再删除旧索引，避免误删后无法恢复
        if args.rebuild and not args.dry_run:
            delete_user_collection(user_id)

        for path in files:
            ext = path.suffix.lower().lstrip(".")
            if args.dry_run:
                print(f"[dry-run] user={user_id} {path}")
                continue

            ok, message = build_index(str(path), path.name, ext, user_id)
            if ok:
                total_ok += 1
                print(f"[ok] {message}")
            else:
                total_failed += 1
                print(f"[fail] user={user_id} {path.name}: {message}")

    print(f"\n完成：成功 {total_ok} 个文件，失败 {total_failed} 个。")
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
