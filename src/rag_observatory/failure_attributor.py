"""
Failure Attribution Engine for RAG Systems.
Separates retrieval mistakes, missing evidence, context ignoring, and hallucinated answers.
"""

from typing import List, Dict, Any
from rag_observatory.schema import (
    EvaluationQuery,
    RetrievedPassage,
    GeneratedAnswer,
    RetrievalMetrics,
    GenerationMetrics,
    FailureMode,
    DiagnosticResult,
)


class FailureAttributor:
    """Diagnoses root cause of failure across decoupled RAG components."""

    @staticmethod
    def diagnose(
        query: EvaluationQuery,
        retrieved: List[RetrievedPassage],
        answer: GeneratedAnswer,
        ret_metrics: RetrievalMetrics,
        gen_metrics: GenerationMetrics,
    ) -> DiagnosticResult:
        ret_cids = [p.chunk_id for p in retrieved]
        target_set = set(query.ground_truth_chunk_ids)

        # 1. Unanswerable prompt with hallucinated response
        if not query.answerable:
            if not answer.is_abstention:
                return DiagnosticResult(
                    query_id=query.query_id,
                    query_type=query.query_type,
                    failure_mode=FailureMode.UNGROUNDED_HALLUCINATION,
                    is_failure=True,
                    diagnostic_explanation="Prompt has no supporting evidence in corpus. Generator hallucinated an answer instead of abstaining.",
                    retrieval=ret_metrics,
                    generation=gen_metrics,
                    retrieved_chunk_ids=ret_cids,
                    target_chunk_ids=query.ground_truth_chunk_ids,
                    cited_chunk_ids=answer.cited_chunk_ids,
                )
            else:
                return DiagnosticResult(
                    query_id=query.query_id,
                    query_type=query.query_type,
                    failure_mode=FailureMode.SUCCESS_GROUNDED,
                    is_failure=False,
                    diagnostic_explanation="Successfully identified unanswerable prompt and abstained with insufficient evidence signal.",
                    retrieval=ret_metrics,
                    generation=gen_metrics,
                    retrieved_chunk_ids=ret_cids,
                    target_chunk_ids=query.ground_truth_chunk_ids,
                    cited_chunk_ids=answer.cited_chunk_ids,
                )

        # 2. Erroneous refusal when evidence was present
        if answer.is_abstention and query.answerable:
            return DiagnosticResult(
                query_id=query.query_id,
                query_type=query.query_type,
                failure_mode=FailureMode.INCORRECT_ABSTENTION,
                is_failure=True,
                diagnostic_explanation="Ground-truth evidence was available, but generator falsely abstained from answering.",
                retrieval=ret_metrics,
                generation=gen_metrics,
                retrieved_chunk_ids=ret_cids,
                target_chunk_ids=query.ground_truth_chunk_ids,
                cited_chunk_ids=answer.cited_chunk_ids,
            )

        # 3. Complete Retrieval Miss (Target chunk not in top-k)
        if ret_metrics.recall_at_k == 0.0:
            return DiagnosticResult(
                query_id=query.query_id,
                query_type=query.query_type,
                failure_mode=FailureMode.RETRIEVAL_MISS,
                is_failure=True,
                diagnostic_explanation=f"Retriever failed to find ground-truth chunks {query.ground_truth_chunk_ids} in top-{len(retrieved)} results.",
                retrieval=ret_metrics,
                generation=gen_metrics,
                retrieved_chunk_ids=ret_cids,
                target_chunk_ids=query.ground_truth_chunk_ids,
                cited_chunk_ids=answer.cited_chunk_ids,
            )

        # 4. Evidence Retrieved, but Ranked Too Low (Pushed down rank > 3)
        if ret_metrics.mrr < 0.334:  # Rank 3 is 0.333
            return DiagnosticResult(
                query_id=query.query_id,
                query_type=query.query_type,
                failure_mode=FailureMode.EVIDENCE_RANKED_LOW,
                is_failure=True,
                diagnostic_explanation=f"Ground-truth evidence was retrieved but ranked low (MRR: {ret_metrics.mrr:.3f}, rank >= 4).",
                retrieval=ret_metrics,
                generation=gen_metrics,
                retrieved_chunk_ids=ret_cids,
                target_chunk_ids=query.ground_truth_chunk_ids,
                cited_chunk_ids=answer.cited_chunk_ids,
            )

        # 5. Citation Mismatch (Cited non-existent chunk ID or unretrieved chunk)
        ret_set = set(ret_cids)
        invalid_citations = [c for c in answer.cited_chunk_ids if c not in ret_set]
        if invalid_citations or gen_metrics.citation_precision < 0.50:
            return DiagnosticResult(
                query_id=query.query_id,
                query_type=query.query_type,
                failure_mode=FailureMode.CITATION_MISMATCH,
                is_failure=True,
                diagnostic_explanation=f"Generator fabricated or cited invalid chunk IDs {invalid_citations} not in retrieved context.",
                retrieval=ret_metrics,
                generation=gen_metrics,
                retrieved_chunk_ids=ret_cids,
                target_chunk_ids=query.ground_truth_chunk_ids,
                cited_chunk_ids=answer.cited_chunk_ids,
            )

        # 6. Context Ignored (Evidence was at rank 1, but answer lacks faithfulness or correctness)
        if gen_metrics.faithfulness < 0.60 or gen_metrics.answer_correctness < 0.50:
            return DiagnosticResult(
                query_id=query.query_id,
                query_type=query.query_type,
                failure_mode=FailureMode.CONTEXT_IGNORED,
                is_failure=True,
                diagnostic_explanation="Evidence was retrieved in top ranks, but generator ignored context or produced unfaithful statements.",
                retrieval=ret_metrics,
                generation=gen_metrics,
                retrieved_chunk_ids=ret_cids,
                target_chunk_ids=query.ground_truth_chunk_ids,
                cited_chunk_ids=answer.cited_chunk_ids,
            )

        # 7. Grounded Success
        return DiagnosticResult(
            query_id=query.query_id,
            query_type=query.query_type,
            failure_mode=FailureMode.SUCCESS_GROUNDED,
            is_failure=False,
            diagnostic_explanation="Retrieval found evidence at top rank; generator produced faithful answer with verified citations.",
            retrieval=ret_metrics,
            generation=gen_metrics,
            retrieved_chunk_ids=ret_cids,
            target_chunk_ids=query.ground_truth_chunk_ids,
            cited_chunk_ids=answer.cited_chunk_ids,
        )

    @classmethod
    def aggregate_diagnostics(cls, diagnostics: List[DiagnosticResult]) -> Dict[str, Any]:
        """
        Aggregates query-level diagnostic outcomes across a dataset, calculating
        failure distributions, query-type cross-tabulations, and deterministic mitigations.
        """
        if not diagnostics:
            return {
                "total_queries": 0,
                "success_count": 0,
                "failure_count": 0,
                "grounded_success_rate": 0.0,
                "failure_distribution": {},
                "query_type_breakdown": {},
                "top_failure_mode": None,
                "recommended_mitigation": "No queries evaluated.",
            }

        total = len(diagnostics)
        successes = sum(1 for d in diagnostics if not d.is_failure)
        failures = total - successes

        # Distribution by FailureMode
        mode_counts: Dict[str, int] = {}
        for d in diagnostics:
            mode_key = d.failure_mode.value if hasattr(d.failure_mode, "value") else str(d.failure_mode)
            mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1

        mode_distribution = {
            k: {"count": v, "percentage": round((v / total) * 100, 2)}
            for k, v in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True)
        }

        # Breakdown by query type
        type_groups: Dict[str, Dict[str, int]] = {}
        for d in diagnostics:
            qtype = d.query_type.value if hasattr(d.query_type, "value") else str(d.query_type)
            if qtype not in type_groups:
                type_groups[qtype] = {"total": 0, "failures": 0}
            type_groups[qtype]["total"] += 1
            if d.is_failure:
                type_groups[qtype]["failures"] += 1

        query_type_breakdown = {}
        for qtype, stats in type_groups.items():
            query_type_breakdown[qtype] = {
                "total": stats["total"],
                "failures": stats["failures"],
                "failure_rate": round(stats["failures"] / max(stats["total"], 1), 4),
            }

        # Identify dominant failure mode (excluding success_grounded)
        failure_only_counts = {
            k: v for k, v in mode_counts.items() if k != FailureMode.SUCCESS_GROUNDED.value
        }
        top_failure = max(failure_only_counts, key=failure_only_counts.get) if failure_only_counts else None

        mitigations = {
            FailureMode.RETRIEVAL_MISS.value: "Tune hybrid alpha towards dense embeddings or increase top-k retrieval depth.",
            FailureMode.EVIDENCE_RANKED_LOW.value: "Incorporate a cross-encoder reranker (e.g., BGE-Reranker-Large) into the post-retrieval pipeline.",
            FailureMode.CONTEXT_IGNORED.value: "Reinforce system prompt with chain-of-thought grounding and strict context-conditioned instructions.",
            FailureMode.CITATION_MISMATCH.value: "Enforce JSON Schema or regex constrained decoding on chunk citation IDs.",
            FailureMode.INCORRECT_ABSTENTION.value: "Lower conservative abstention threshold and provide high-relevance exemplar demonstrations.",
            FailureMode.UNGROUNDED_HALLUCINATION.value: "Implement self-consistency verification and fine-tune classifier on out-of-domain unanswerable prompts.",
        }

        recommendation = mitigations.get(top_failure, "All evaluated queries met grounded reliability criteria.")

        return {
            "total_queries": total,
            "success_count": successes,
            "failure_count": failures,
            "grounded_success_rate": round(successes / total, 4),
            "failure_distribution": mode_distribution,
            "query_type_breakdown": query_type_breakdown,
            "top_failure_mode": top_failure,
            "recommended_mitigation": recommendation,
        }

    @staticmethod
    def compute_context_utilization_efficiency(
        retrieved: List[RetrievedPassage],
        answer: GeneratedAnswer,
    ) -> Dict[str, Any]:
        """
        Measures how efficiently retrieved context passages are cited in the generated answer,
        computing citation density, rank-weighted utilization (1 / rank), and unused context waste ratio.
        """
        if not retrieved:
            return {
                "retrieved_count": 0,
                "cited_retrieved_count": 0,
                "context_utilization_rate": 0.0,
                "context_waste_ratio": 0.0,
                "rank_weighted_utilization": 0.0,
            }

        cited_set = set(answer.cited_chunk_ids)
        cited_passages = [p for p in retrieved if p.chunk_id in cited_set]
        total_ret = len(retrieved)
        cited_count = len(cited_passages)

        utilization_rate = cited_count / total_ret
        waste_ratio = 1.0 - utilization_rate

        max_harmonic = sum(1.0 / max(p.rank, 1) for p in retrieved)
        cited_harmonic = sum(1.0 / max(p.rank, 1) for p in cited_passages)
        rank_weighted = cited_harmonic / max_harmonic if max_harmonic > 0 else 0.0

        return {
            "retrieved_count": total_ret,
            "cited_retrieved_count": cited_count,
            "context_utilization_rate": round(utilization_rate, 4),
            "context_waste_ratio": round(waste_ratio, 4),
            "rank_weighted_utilization": round(rank_weighted, 4),
        }

    @staticmethod
    def score_severity_weighted_reliability(
        diagnostics: List[DiagnosticResult],
    ) -> Dict[str, Any]:
        """
        Computes a severity-weighted reliability index across diagnostic results,
        penalizing dangerous failure modes (ungrounded hallucinations, fabricated citations)
        more heavily than conservative abstentions or low-ranked evidence.
        """
        if not diagnostics:
            return {
                "severity_weighted_index": 0.0,
                "critical_hallucination_rate": 0.0,
                "release_gate_status": "NO_DATA",
            }

        severity_weights = {
            FailureMode.SUCCESS_GROUNDED.value: 0.0,
            FailureMode.INCORRECT_ABSTENTION.value: 0.35,
            FailureMode.EVIDENCE_RANKED_LOW.value: 0.45,
            FailureMode.RETRIEVAL_MISS.value: 0.60,
            FailureMode.CONTEXT_IGNORED.value: 0.75,
            FailureMode.CITATION_MISMATCH.value: 0.90,
            FailureMode.UNGROUNDED_HALLUCINATION.value: 1.00,
        }

        total_penalty = 0.0
        critical_count = 0
        for d in diagnostics:
            mode_val = d.failure_mode.value if hasattr(d.failure_mode, "value") else str(d.failure_mode)
            penalty = severity_weights.get(mode_val, 0.50 if d.is_failure else 0.0)
            total_penalty += penalty
            if mode_val in (
                FailureMode.UNGROUNDED_HALLUCINATION.value,
                FailureMode.CITATION_MISMATCH.value,
            ):
                critical_count += 1

        n = len(diagnostics)
        reliability_index = max(0.0, 1.0 - (total_penalty / n))
        critical_rate = critical_count / n

        status = (
            "PASS"
            if reliability_index >= 0.80 and critical_rate <= 0.05
            else "WARN"
            if reliability_index >= 0.65
            else "BLOCK"
        )

        return {
            "severity_weighted_index": round(reliability_index, 4),
            "critical_hallucination_rate": round(critical_rate, 4),
            "release_gate_status": status,
        }


