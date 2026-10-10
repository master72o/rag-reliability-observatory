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
