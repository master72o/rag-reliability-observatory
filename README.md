# 🔭 RAG Reliability Observatory (`rag-reliability-observatory`)

> **A Diagnostic Laboratory for Retrieval-Augmented Generation Systems — Decoupling Retrieval Failures, Ranking Degradation, Evidence Grounding & Failure Attribution.**

[![CI](https://github.com/master72o/rag-reliability-observatory/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/rag-reliability-observatory/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](app.py)

---

## 📌 Executive Overview

Most RAG implementations demonstrate a chatbot querying a vector database and returning an answer. When the assistant answers incorrectly, traditional evaluations produce a single aggregate score without answering the critical engineering question: **which component failed and why?**

When a RAG system provides an incorrect answer, there are five distinct root causes:
1. **The evidence exists, but the retriever missed it** (`RETRIEVAL_MISS`).
2. **The retriever found the evidence, but ranked it too low** (`EVIDENCE_RANKED_LOW`).
3. **The evidence was retrieved in the top rank, but the generator ignored it** (`CONTEXT_IGNORED`).
4. **The answer cited non-existent or fabricated passage IDs** (`CITATION_MISMATCH`).
5. **The prompt has no answer in the documents, but the model hallucinated instead of abstaining** (`UNGROUNDED_HALLUCINATION`).

**RAG Reliability Observatory** is not another RAG chatbot. It is a **controlled diagnostic and measurement platform** that decouples retrieval from generation, isolates failure mechanisms, and executes reproducible ablation experiments across lexical (BM25), dense vector, hybrid Reciprocal Rank Fusion (RRF), and contextual rerankers.

---

## 🏗️ Architecture & Diagnostic Pipeline

```
                               ┌────────────────────────────────┐
                               │  Question + Reference Corpus   │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
     Document Chunking Engine                                       Benchmark Query Suite
    (256 vs 512 Chunk Windows)                                    (Direct, Multi-Hop, Unanswerable)
               │                                                               │
               └───────────────────────────────┬───────────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Decoupled Retrieval Layers    │
                              └────────────────┬─────────────────┘
                                               │
             ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
             ▼                   ▼                           ▼                   ▼
        BM25 Lexical        Dense Semantic              Hybrid RRF           Contextual Reranker
      (Okapi Scoring)     (Cosine Vector Space)      (Rank Fusion k=60)      (Phrase Proximity)
             │                   │                           │                   │
             └───────────────────┴─────────────┬─────────────┴───────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Retrieved Context Passages    │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Evidence-Grounded Answer Engine │
                              │ (Structured Answer + Citations)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Multilayer Evaluation Engine  │
                              └────────────────┬─────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
     Retrieval Metrics                                                   Generation Metrics
 (Recall@K, Precision@K, MRR, nDCG)                              (Faithfulness, Correctness, Citations)
             │                                                                   │
             └─────────────────────────────────┬─────────────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Failure Attribution Engine    │
                              └────────────────┬─────────────────┘
                                               │
         ┌───────────────────┬─────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                 ▼                   ▼                   ▼
   Retrieval Miss      Ranked Low       Context Ignored     Citation Mismatch   Hallucinated Guess
```

---

## 🔬 Multi-Strategy Ablation Benchmark

Controlled ablation study evaluating 4 distinct retrieval architectures over the standardized knowledge corpus (`data/corpus/`) across 10 multi-category benchmark queries:

| Strategy | Architecture Configuration | Recall@3 | nDCG@3 | Faithfulness | Citation Precision | Primary Root Failure |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`bm25`** | Okapi BM25 ($k_1=1.5, b=0.75$) | `1.0000` | `1.0000` | `1.0000` | `1.0000` | Keyword lexical mismatch on synonyms |
| **`dense`** | TF-IDF Cosine Vector Space | `1.0000` | `1.0000` | `1.0000` | `1.0000` | Numerical parameter dilution |
| **`hybrid`** | Reciprocal Rank Fusion ($k=60$) | `1.0000` | `1.0000` | `0.2000` | `0.2000` | Threshold over-abstention |
| **`hybrid_reranked`** | Hybrid RRF + Contextual Reranker | `1.0000` | `1.0000` | `0.9000` | `0.9000` | Top-1 evidence alignment |

> *All metrics computed via `rag-observatory benchmark` and verifiable via automated CI.*

---

## 🏷️ Diagnostic Failure Attribution Taxonomy

AnnotatorIQ categorizes every query execution into an actionable diagnostic class:

| Failure Category | Diagnostic Definition | Engineering Remediation |
| :--- | :--- | :--- |
| **`RETRIEVAL_MISS`** | Target ground-truth chunk not present in top-$k$ results ($Recall@K = 0$). | Increase retrieval depth $k$, switch from lexical to hybrid, or reduce chunk size. |
| **`EVIDENCE_RANKED_LOW`** | Evidence retrieved, but pushed down ($MRR < 0.33$, rank $\ge 4$). | Add cross-encoder reranking or tune lexical/semantic weighting ratios. |
| **`CONTEXT_IGNORED`** | Evidence present in top-1 rank, but answer contradicts context ($Faithfulness < 0.60$). | Constrain generation temperature, enforce strict prompt grounding. |
| **`CITATION_MISMATCH`** | Answer references non-existent chunk IDs or ungrounded passage numbers. | Enforce structured JSON schema citation validation in generator. |
| **`INCORRECT_ABSTENTION`** | Ground-truth evidence was available, but model refused to answer. | Adjust confidence threshold downwards or improve instruction prompt. |
| **`UNGROUNDED_HALLUCINATION`** | Query unanswerable from corpus, but model produced unsupported guess. | Introduce few-shot abstention demonstrations; calibrate uncertainty. |
| **`SUCCESS_GROUNDED`** | Evidence retrieved in top ranks, answer correct, citations verified. | System operates within production SLA. |

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
# Clone repository
git clone https://github.com/master72o/rag-reliability-observatory.git
cd rag-reliability-observatory

# Install in editable mode
pip install -e .
```

### 2. Run Test Suite
```bash
pytest -v
```
*(Executes 19 unit tests covering BM25, Dense Cosine, Hybrid RRF, Reranking, Decoupled Metrics, and Failure Attribution).*

### 3. Run Benchmark Ablation Study via CLI
```bash
rag-observatory benchmark \
  --corpus data/corpus/knowledge_corpus.jsonl \
  --queries data/queries/benchmark_queries.jsonl \
  --out reports/rag_ablation_report.md
```

### 4. Diagnose a Specific Query
```bash
rag-observatory diagnose \
  --corpus data/corpus/knowledge_corpus.jsonl \
  --query "What is the expiration time (TTL) of OAuth 2.0 Bearer tokens?" \
  --strategy hybrid_reranked
```

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
rag-reliability-observatory/
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # MIT Open Source License
├── pyproject.toml                 # Packaging & CLI entrypoints
├── requirements.txt               # Pinned dependencies
├── app.py                         # Interactive Streamlit Observatory Dashboard
├── src/rag_observatory/
│   ├── __init__.py                # Package root & public exports
│   ├── schema.py                  # Pydantic v2 schemas: CorpusChunk, Query, FailureMode
│   ├── retriever.py               # BM25, Dense Vector, Hybrid RRF, and Contextual Reranker
│   ├── generator.py               # Evidence-grounded generation & citation validator
│   ├── evaluator.py               # Retrieval metrics (nDCG, MRR) & Generation metrics (Faithfulness)
│   ├── failure_attributor.py      # Diagnostic root cause classifier (6-tier taxonomy)
│   ├── benchmark_runner.py        # Multi-strategy ablation experiment orchestrator
│   └── cli.py                     # Command-line interface (`rag-observatory`)
├── data/
│   ├── corpus/
│   │   └── knowledge_corpus.jsonl # Technical knowledge passages (Auth, Rate Limits, Architecture)
│   └── queries/
│       └── benchmark_queries.jsonl # 10 benchmark queries (Direct, Multi-hop, Unanswerable)
├── scripts/
│   └── build_observatory_corpus.py # Benchmark corpus & query generator
├── reports/
│   └── rag_ablation_report.md     # Architecture ablation results report
└── tests/
    ├── test_retriever.py          # BM25, Dense, RRF, and Reranker unit tests
    ├── test_rag_evaluator.py      # Recall@k, MRR, nDCG, and Faithfulness metric tests
    ├── test_failure_attributor.py # Diagnostic taxonomy unit tests
    └── test_schema.py             # Schema parsing and validation tests
```

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**:
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Dataset Type**: Packaged benchmark corpus in `data/corpus/` and evaluation queries in `data/queries/`.
> - **Reproducibility**: Run `pytest -v` or `rag-observatory benchmark` to regenerate all ablation metrics and failure distributions locally.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
