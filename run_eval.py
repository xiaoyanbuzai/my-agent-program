"""检索 / 引用评测：用金标准问题集量化混合检索质量。

用法：
    uv run python run_eval.py --user 1                     # 只评测检索（不调用 LLM)
    uv run python run_eval.py --user 1 --answers           # 额外生成回答并检查引用
    uv run python run_eval.py --user 1 --golden eval/golden_set.json --top-k 5

金标准格式见 eval/golden_set.example.json:
    {
      "question": "……",
      "expected_sources": ["paper.pdf"],
      "expected_pages": [3],
      "keywords": ["……"]
    }

输出指标：
- hit@k:top-k 中是否命中期望文档；
- MRR:第一个命中结果的倒数排名；
- page_hit:命中结果的页码是否落在期望页码；
- citation_rate / keyword_coverage(--answers 时)。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from kb_retriever import search_kb  # noqa: E402
from logging_utils import setup_logging  # noqa: E402

DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.json"


def load_golden(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"找不到金标准文件：{path}\n"
            "请复制 eval/golden_set.example.json 为 eval/golden_set.json 并填写。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("金标准文件必须是非空 JSON 数组")
    for item in data:
        if not item.get("question"):
            raise SystemExit("每个评测项都必须包含 question")
    return data


def evaluate_retrieval(items: list[dict], user_id: int, top_k: int) -> list[dict]:
    rows = []
    for item in items:
        question = item["question"]
        expected_sources = set(item.get("expected_sources") or [])
        expected_pages = set(item.get("expected_pages") or [])

        started = time.perf_counter()
        results = search_kb(question, user_id, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        rank = 0
        page_hit = None
        for index, result in enumerate(results, start=1):
            if result["source"] not in expected_sources:
                continue
            if rank == 0:
                rank = index
            if expected_pages and result["page"] in expected_pages:
                page_hit = True
        if expected_pages and page_hit is None:
            page_hit = False

        rows.append(
            {
                "question": question,
                "sources": [result["source"] for result in results],
                "pages": [result["page"] for result in results],
                "scores": [round(result["score"], 4) for result in results],
                "expected_sources": sorted(expected_sources),
                "rank": rank,
                "hit": rank > 0,
                "page_hit": page_hit,
                "latency_ms": round(elapsed_ms, 1),
            }
        )
    return rows


def evaluate_answers(rows: list[dict], items: list[dict], user_id: int) -> None:
    """调用 Agent 生成回答，检查引用与关键词覆盖(需要可用的 LLM Key)。"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    from web_app import build_graph

    graph = build_graph(checkpointer=MemorySaver(), store=InMemoryStore())
    for row, item in zip(rows, items):
        config = {
            "configurable": {"thread_id": f"eval-{abs(hash(row['question']))}", "user_id": user_id},
            "recursion_limit": get_settings().recursion_limit,
        }
        try:
            result = graph.invoke(
                {"messages": [{"role": "user", "content": row["question"]}]},
                config,
            )
            answer = str(result["messages"][-1].content)
        except Exception as exc:  # 单题失败不影响整体
            answer = f"[ERROR] {exc}"

        keywords = [str(keyword) for keyword in (item.get("keywords") or [])]
        covered = [keyword for keyword in keywords if keyword in answer]
        row["answer"] = answer
        row["citation_hit"] = ("来源" in answer) or ("[来源" in answer)
        row["keyword_coverage"] = (
            round(len(covered) / len(keywords), 3) if keywords else None
        )


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    hits = sum(1 for row in rows if row["hit"])
    reciprocal_ranks = [1 / row["rank"] if row["rank"] else 0.0 for row in rows]
    page_rows = [row for row in rows if row["page_hit"] is not None]
    summary = {
        "questions": total,
        "hit_at_k": round(hits / total, 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "page_hit_rate": (
            round(sum(1 for row in page_rows if row["page_hit"]) / len(page_rows), 4)
            if page_rows
            else None
        ),
        "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 1),
    }

    if any("citation_hit" in row for row in rows):
        summary["citation_rate"] = round(
            sum(1 for row in rows if row.get("citation_hit")) / total, 4
        )
        coverages = [
            row["keyword_coverage"]
            for row in rows
            if row.get("keyword_coverage") is not None
        ]
        if coverages:
            summary["keyword_coverage"] = round(statistics.mean(coverages), 4)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索 / 回答评测")
    parser.add_argument("--user", type=int, required=True, help="租户 ID")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=get_settings().retrieve_top_k)
    parser.add_argument("--answers", action="store_true", help="调用 LLM 并检查引用")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    items = load_golden(args.golden)
    rows = evaluate_retrieval(items, args.user, args.top_k)
    if args.answers:
        evaluate_answers(rows, items, args.user)

    summary = summarize(rows)
    report = {
        "user_id": args.user,
        "golden": str(args.golden),
        "settings": {
            "embedding_model": get_settings().embedding_model,
            "reranker_model": get_settings().reranker_model,
            "retrieve_top_k": args.top_k,
        },
        "summary": summary,
        "details": rows,
    }

    output = args.output or (ROOT / "eval" / f"report-{int(time.time())}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细已写入：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
