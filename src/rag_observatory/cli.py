"""
Command Line Interface (CLI) for RAG Reliability Observatory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from rag_observatory.schema import CorpusChunk, EvaluationQuery
from rag_observatory.benchmark_runner import BenchmarkRunner


def load_corpus(path: str) -> List[CorpusChunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(CorpusChunk.model_validate_json(line.strip()))
    return chunks


def load_queries(path: str) -> List[EvaluationQuery]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(EvaluationQuery.model_validate_json(line.strip()))
    return queries


def main():
    parser = argparse.ArgumentParser(description="RAG Reliability Observatory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-strategy RAG ablation study")
    bench_parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    bench_parser.add_argument("--queries", required=True, help="Path to benchmark queries JSONL")
    bench_parser.add_argument("--out", required=False, default="reports/rag_ablation_report.md", help="Output report path")

    # query diagnostic command
    diag_parser = subparsers.add_parser("diagnose", help="Diagnose single query execution")
    diag_parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    diag_parser.add_argument("--query", required=True, help="Query text")
    diag_parser.add_argument("--strategy", default="hybrid_reranked", choices=["bm25", "dense", "hybrid", "hybrid_reranked"])

    args = parser.parse_args()

    chunks = load_corpus(args.corpus)
    runner = BenchmarkRunner(chunks)

    if args.command == "benchmark":
        queries = load_queries(args.queries)
        results = runner.run_ablation_study(queries)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        md = "# 🔬 RAG Reliability Observatory: Architecture Ablation Report\n\n"
        md += "| Strategy | Success Rate (%) | Recall@3 | nDCG@3 | Faithfulness | Citation Precision |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"

        for strat, data in results.items():
            md += f"| **`{strat}`** | `{data['success_rate_pct']}%` | `{data['mean_recall_at_k']:.4f}` | `{data['mean_ndcg']:.4f}` | `{data['mean_faithfulness']:.4f}` | `{data['mean_citation_precision']:.4f}` |\n"

        md += "\n---\n\n## Failure Mode Attribution Breakdown\n\n"
        for strat, data in results.items():
            md += f"### Strategy: `{strat}`\n"
            for f_mode, count in data["failure_distribution"].items():
                md += f"- **`{f_mode}`**: {count} instances\n"
            md += "\n"

        out_path.write_text(md, encoding="utf-8")
        print(f"[RAG Observatory] Ablation study complete. Results exported to {out_path}")
        for strat, data in results.items():
            print(f"  [{strat}] Success: {data['success_rate_pct']}% | Recall@3: {data['mean_recall_at_k']:.3f} | nDCG: {data['mean_ndcg']:.3f}")

    elif args.command == "diagnose":
        q = EvaluationQuery(
            query_id="manual_test",
            question=args.query,
            reference_answer="Manual test query",
            ground_truth_chunk_ids=[],
            answerable=True,
        )
        diag = runner.run_query(q, strategy=args.strategy)
        print(json.dumps(diag.dict(), indent=2))


if __name__ == "__main__":
    main()
