"""下载本项目使用的本地模型到 Hugging Face 缓存。

默认下载中文嵌入模型与中文精排模型；下载完成后可保持 HF_HUB_OFFLINE=1 离线运行。

用法：
    uv run python scripts/download_models.py
    uv run python scripts/download_models.py --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402

DEFAULT_MODELS = [
    "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-reranker-v2-m3",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="下载嵌入 / 精排模型")
    parser.add_argument(
        "--model",
        action="append",
        help="模型 ID，可重复；默认下载 bge-base-zh-v1.5 与 bge-reranker-v2-m3",
    )
    parser.add_argument("--cache-dir", default="", help="模型缓存目录（默认 HF_HOME）")
    args = parser.parse_args()

    # 下载脚本必须联网
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    settings = get_settings()
    cache_dir = args.cache_dir or settings.model_cache_dir or None
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub，请先执行 `uv sync`。")
        return 1

    models = args.model or DEFAULT_MODELS
    failed = 0
    for model in models:
        print(f"正在下载：{model}")
        try:
            path = snapshot_download(model, cache_dir=cache_dir)
            print(f"  -> {path}")
        except Exception as exc:
            failed += 1
            print(f"  !! 下载失败：{exc}")

    if failed:
        print(f"\n{failed} 个模型下载失败，请检查网络或镜像配置。")
        return 2

    print("\n全部完成。建议在 .env 中设置：")
    print("  EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5")
    print("  RERANKER_MODEL=BAAI/bge-reranker-v2-m3")
    print("  HF_HUB_OFFLINE=1")
    print("然后运行：uv run python scripts/reindex.py --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
