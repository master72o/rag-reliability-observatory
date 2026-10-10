# RAG Evaluation Lab (`rag-evaluation-lab`)

> Decoupled Retrieval & Generation Quality Benchmarking Framework for Production RAG Pipelines.

[![CI](https://github.com/master72o/rag-evaluation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/rag-evaluation-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Summary

`rag-evaluation-lab` provides a robust, decoupled evaluation framework for Retrieval-Augmented Generation (RAG) systems. Modern RAG failures stem from two independent sources: **Retrieval Failure** (missing relevant facts, ranking noisy chunks highly) and **Generation Failure** (hallucination, unfaithful context synthesis, citation hallucination).

This repository isolates retrieval quality metrics (Precision@K, Recall@K, MRR, NDCG) from generation quality metrics (Faithfulness, Answer Relevance, Context Precision, Citation Correctness) across three RAG pipeline configurations:
1. **RAG v1 (Baseline)**: Dense Vector Search, Chunk Size 512, Top-K 3.
2. **RAG v2 (Hybrid)**: Dense + Sparse Hybrid Search, Chunk Size 256, Top-K 5.
3. **RAG v3 (Reranked)**: Hybrid Search + Cross-Encoder Reranker, Chunk Size 128, Top-K 5.

---

## Architectural Taxonomy & Decoupled Metrics

```
                     ┌───────────────────────────────────────────────┐
                     │            User Query + Benchmark             │
                     └───────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          ┌─────────────────────────┐                 ┌─────────────────────────┐
          │  Retrieval Evaluation   │                 │  Generation Evaluation  │
          └────────────┬────────────┘                 └────────────┬────────────┘
                       │                                           │
         ┌─────────────┴─────────────┐               ┌─────────────┴─────────────┐
         ▼                           ▼               ▼                           ▼
    Precision@K                   NDCG          Faithfulness            Citation Correctness
    Recall@K                       MRR          Relevance               Context Precision
```

### Key Metrics Defined:
- **Precision@K**: Proportion of top-$K$ retrieved chunks matching ground-truth chunk IDs.
- **Recall@K**: Proportion of total relevant ground-truth chunks retrieved in top-$K$.
- **MRR (Mean Reciprocal Rank)**: Inverse rank of the first relevant chunk retrieved.
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality with logarithmic decay.
- **Faithfulness**: Extent to which generated claims are strictly supported by retrieved context.
- **Context Precision**: Signal-to-noise ratio in retrieved context window.
- **Citation Correctness**: Verification that cited chunk IDs explicitly contain referenced facts.

---

## Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/master72o/rag-evaluation-lab.git
cd rag-evaluation-lab

# Install in editable mode
pip install -e .

# Run pytest suite
pytest
```

---

## Running RAG Benchmark Evaluation

Execute the complete RAG evaluation pipeline via CLI:

```bash
python -m rag_eval.cli run \
  --dataset data/rag_benchmark_dataset.jsonl \
  --output-dir results \
  --report-path reports/rag_evaluation_report.md \
  --figures-dir reports/figures
```

---

## Controlled Benchmark Results

| RAG Variant | Overall Score | Quality Pass Rate | Precision@K | Recall@K | NDCG | Faithfulness | Citation Correctness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RAG v1 (Baseline)** | `0.6550` | `25.0%` | `0.4167` | `0.6250` | `0.6482` | `0.6875` | `0.7500` |
| **RAG v2 (Hybrid)** | `0.8525` | `100.0%` | `0.7500` | `1.0000` | `0.8715` | `0.8875` | `1.0000` |
| **RAG v3 (Reranked)** | `0.9760` | `100.0%` | `0.7500` | `1.0000` | `0.9850` | `0.9825` | `1.0000` |

### Key Performance Visualizations

- **Retrieval Metrics Comparison**: `reports/figures/retrieval_metrics_comparison.png`
- **Generation & Faithfulness Metrics**: `reports/figures/generation_metrics_comparison.png`
- **Overall Score Progression**: `reports/figures/overall_rag_performance.png`

---

## Repository Structure

```
rag-evaluation-lab/
├── .github/workflows/ci.yml    # Continuous Integration pipeline
├── pyproject.toml              # Build & dependency metadata
├── requirements.txt            # Python dependencies
├── rag_eval/                   # Core Python package
│   ├── __init__.py
│   ├── schema.py               # Data models & JSONL parser
│   ├── retrieval_evaluator.py  # Precision@K, Recall@K, MRR, NDCG
│   ├── generation_evaluator.py # Faithfulness, Relevance, Citation Verification
│   ├── experiment_runner.py    # Pipeline runner (v1, v2, v3)
│   ├── metrics.py              # Suite metric aggregator
│   ├── visualizer.py           # Matplotlib figure generator
│   ├── report_generator.py     # Markdown report & JSON/CSV exporter
│   └── cli.py                  # CLI interface
├── data/
│   ├── README.md
│   └── rag_benchmark_dataset.jsonl # Synthetic domain benchmark dataset
├── results/                    # Exported JSON and CSV metrics
├── reports/
│   ├── rag_evaluation_report.md # Auto-generated markdown report
│   └── figures/                # Visual comparison charts
└── tests/                      # Unit & integration tests
```

---


---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in data/.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run pytest or the package CLI to re-compute all scores, metrics, and summary tables locally.

## License

MIT License © 2026 AI Evaluation Engineering Team.
