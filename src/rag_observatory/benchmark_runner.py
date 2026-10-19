"""
Benchmark Runner executing comparative ablation studies across RAG architectures.
Compares BM25 Lexical, Dense Vector, Hybrid RRF, and Contextual Reranked pipelines.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np
from rag_observatory.schema import (
    CorpusChunk,
    EvaluationQuery,
    RetrievedPassage,
    GeneratedAnswer,
    DiagnosticResult,
    FailureMode,
)
from rag_observatory.retriever import (
    BM25Retriever,
    DenseVectorRetriever,
    HybridRRFRetriever,
    ContextualReranker,
)
from rag_observatory.evaluator import RAGEvaluator
from rag_observatory.generator import EvidenceGroundedGenerator
from rag_observatory.failure_attributor import FailureAttributor
from rag_observatory.cost_tracker import RAGCostEstimator


class BenchmarkRunner:
    """Orchestrates controlled ablations and produces failure attribution analytics."""

    def __init__(self, chunks: List[CorpusChunk]):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseVectorRetriever(chunks)
        self.hybrid = HybridRRFRetriever(self.bm25, self.dense)
        self.generator = EvidenceGroundedGenerator()

    def run_query(
        self, query: EvaluationQuery, strategy: str = "hybrid_reranked", top_k: int = 3
    ) -> DiagnosticResult:
        start_time = time.perf_counter()

        # 1. Retrieve
        if strategy == "bm25":
            retrieved = self.bm25.retrieve(query.question, top_k=top_k)
        elif strategy == "dense":
            retrieved = self.dense.retrieve(query.question, top_k=top_k)
        elif strategy == "hybrid":
            retrieved = self.hybrid.retrieve(query.question, top_k=top_k)
        elif strategy == "hybrid_reranked":
            candidates = self.hybrid.retrieve(query.question, top_k=top_k * 2)
            retrieved = ContextualReranker.rerank(query.question, candidates, top_k=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 2. Generate
        answer = self.generator.generate(query, retrieved)

        # 3. Evaluate Decoupled Layers
        ret_metrics = RAGEvaluator.evaluate_retrieval(retrieved, query.ground_truth_chunk_ids, k=top_k)
        gen_metrics = RAGEvaluator.evaluate_generation(answer, query, retrieved)

        # 4. Attribute Failure Mode
        diagnostic = FailureAttributor.diagnose(query, retrieved, answer, ret_metrics, gen_metrics)

        # 5. Latency & Token Cost Profiling
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        prompt_text = query.question + " " + " ".join(p.text for p in retrieved)
        prompt_tokens = max(10, len(prompt_text.split()))
        completion_tokens = max(5, len(answer.answer_text.split()))
        cost_eval = RAGCostEstimator.estimate_query_cost(prompt_tokens, completion_tokens)

        diagnostic.latency_ms = latency_ms
        diagnostic.estimated_cost_usd = cost_eval["total_cost_usd"]

        return diagnostic

    def run_ablation_study(
        self, queries: List[EvaluationQuery], top_k: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        strategies = ["bm25", "dense", "hybrid", "hybrid_reranked"]
        study_results = {}

        for strat in strategies:
            diagnostics: List[DiagnosticResult] = []
            for q in queries:
                diag = self.run_query(q, strategy=strat, top_k=top_k)
                diagnostics.append(diag)

            # Aggregate statistics
            recalls = [d.retrieval.recall_at_k for d in diagnostics]
            precisions = [d.retrieval.precision_at_k for d in diagnostics]
            mrrs = [d.retrieval.mrr for d in diagnostics]
            ndcgs = [d.retrieval.ndcg for d in diagnostics]
            faiths = [d.generation.faithfulness for d in diagnostics]
            cit_precs = [d.generation.citation_precision for d in diagnostics]
            successes = [1 if not d.is_failure else 0 for d in diagnostics]
            latencies = [d.latency_ms for d in diagnostics if d.latency_ms is not None]
            costs = [d.estimated_cost_usd for d in diagnostics if d.estimated_cost_usd is not None]

            mean_ndcg = round(float(np.mean(ndcgs)), 4)
            mean_latency = round(float(np.mean(latencies)), 2) if latencies else 1.0
            mean_cost = round(float(np.mean(costs)), 6) if costs else 0.0001
            efficiency_score = RAGCostEstimator.compute_efficiency_score(mean_ndcg, mean_latency, mean_cost)

            # Failure taxonomy distribution
            failure_counts = {}
            for d in diagnostics:
                mode_str = d.failure_mode.value
                failure_counts[mode_str] = failure_counts.get(mode_str, 0) + 1

            study_results[strat] = {
                "strategy": strat,
                "total_queries": len(queries),
                "success_count": sum(successes),
                "success_rate_pct": round((sum(successes) / len(queries)) * 100, 1),
                "mean_recall_at_k": round(float(np.mean(recalls)), 4),
                "mean_precision_at_k": round(float(np.mean(precisions)), 4),
                "mean_mrr": round(float(np.mean(mrrs)), 4),
                "mean_ndcg": mean_ndcg,
                "mean_faithfulness": round(float(np.mean(faiths)), 4),
                "mean_citation_precision": round(float(np.mean(cit_precs)), 4),
                "mean_latency_ms": mean_latency,
                "mean_cost_per_query_usd": mean_cost,
                "efficiency_score": efficiency_score,
                "failure_distribution": failure_counts,
                "detailed_diagnostics": [(d.model_dump() if hasattr(d, "model_dump") else d.dict()) for d in diagnostics],
            }

        return study_results

