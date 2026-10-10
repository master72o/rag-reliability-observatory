"""
Unit tests for Failure Attribution and Diagnostic Classification in RAG Observatory.
"""

from rag_observatory.schema import (
    EvaluationQuery,
    RetrievedPassage,
    GeneratedAnswer,
    RetrievalMetrics,
    GenerationMetrics,
    FailureMode,
    QueryType,
)
from rag_observatory.failure_attributor import FailureAttributor


def test_ungrounded_hallucination_detected():
    query = EvaluationQuery(
        query_id="q1",
        question="What is the Bitcoin discount?",
        reference_answer="",
        ground_truth_chunk_ids=[],
        answerable=False,  # Unanswerable!
        query_type=QueryType.UNANSWERABLE,
    )
    answer = GeneratedAnswer(
        answer_text="You get a 25% discount for Bitcoin.",
        is_abstention=False,  # Should have abstained!
    )
    ret_m = RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, mrr=1.0, ndcg=1.0, retrieved_count=3, target_count=0)
    gen_m = GenerationMetrics(faithfulness=0.0, answer_correctness=0.0, citation_precision=0.0, citation_recall=0.0, abstention_accuracy=0.0)

    diag = FailureAttributor.diagnose(query, [], answer, ret_m, gen_m)
    assert diag.failure_mode == FailureMode.UNGROUNDED_HALLUCINATION
    assert diag.is_failure is True


def test_retrieval_miss_detected():
    query = EvaluationQuery(
        query_id="q2",
        question="What is the token TTL?",
        reference_answer="3600 seconds",
        ground_truth_chunk_ids=["chunk_auth_1"],
        answerable=True,
    )
    answer = GeneratedAnswer(answer_text="Not found", is_abstention=False)
    ret_m = RetrievalMetrics(precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, ndcg=0.0, retrieved_count=3, target_count=1)
    gen_m = GenerationMetrics(faithfulness=0.0, answer_correctness=0.0, citation_precision=0.0, citation_recall=0.0, abstention_accuracy=1.0)

    diag = FailureAttributor.diagnose(query, [RetrievedPassage(chunk_id="wrong_chunk", score=0.5, rank=1, text="")], answer, ret_m, gen_m)
    assert diag.failure_mode == FailureMode.RETRIEVAL_MISS
    assert diag.is_failure is True


def test_citation_mismatch_detected():
    query = EvaluationQuery(
        query_id="q3",
        question="What is the token TTL?",
        reference_answer="3600 seconds",
        ground_truth_chunk_ids=["chunk_auth_1"],
        answerable=True,
    )
    answer = GeneratedAnswer(
        answer_text="TTL is 3600s.",
        cited_chunk_ids=["fabricated_chunk_999"],  # Invalid citation!
        is_abstention=False,
    )
    ret_m = RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, mrr=1.0, ndcg=1.0, retrieved_count=1, target_count=1)
    gen_m = GenerationMetrics(faithfulness=1.0, answer_correctness=1.0, citation_precision=0.0, citation_recall=0.0, abstention_accuracy=1.0)

    passages = [RetrievedPassage(chunk_id="chunk_auth_1", score=0.9, rank=1, text="TTL is 3600s.")]
    diag = FailureAttributor.diagnose(query, passages, answer, ret_m, gen_m)
    assert diag.failure_mode == FailureMode.CITATION_MISMATCH
    assert diag.is_failure is True


def test_grounded_success_detected():
    query = EvaluationQuery(
        query_id="q4",
        question="What is the token TTL?",
        reference_answer="3600 seconds",
        ground_truth_chunk_ids=["chunk_auth_1"],
        answerable=True,
    )
    answer = GeneratedAnswer(
        answer_text="TTL is 3600 seconds.",
        cited_chunk_ids=["chunk_auth_1"],
        is_abstention=False,
    )
    ret_m = RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, mrr=1.0, ndcg=1.0, retrieved_count=1, target_count=1)
    gen_m = GenerationMetrics(faithfulness=1.0, answer_correctness=1.0, citation_precision=1.0, citation_recall=1.0, abstention_accuracy=1.0)

    passages = [RetrievedPassage(chunk_id="chunk_auth_1", score=0.9, rank=1, text="TTL is 3600 seconds.")]
    diag = FailureAttributor.diagnose(query, passages, answer, ret_m, gen_m)
    assert diag.failure_mode == FailureMode.SUCCESS_GROUNDED
    assert diag.is_failure is False
