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

