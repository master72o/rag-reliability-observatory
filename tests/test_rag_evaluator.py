"""
Unit tests for RAGEvaluator metrics: Recall@k, Precision@k, MRR, nDCG, and Faithfulness.
"""

from rag_observatory.schema import (
    EvaluationQuery,
    RetrievedPassage,
    GeneratedAnswer,
)
from rag_observatory.evaluator import RAGEvaluator


def test_retrieval_metrics_calculation():
    retrieved = [
        RetrievedPassage(chunk_id="c_other", score=0.9, rank=1, text="Irrelevant"),
        RetrievedPassage(chunk_id="c_target_1", score=0.8, rank=2, text="Target 1"),
        RetrievedPassage(chunk_id="c_target_2", score=0.7, rank=3, text="Target 2"),
    ]
    targets = ["c_target_1", "c_target_2"]
    metrics = RAGEvaluator.evaluate_retrieval(retrieved, targets, k=3)

    assert metrics.precision_at_k == round(2 / 3, 4)
    assert metrics.recall_at_k == 1.0  # Found both targets
    assert metrics.mrr == 0.50  # First target at rank 2 -> 1/2


def test_generation_faithfulness_and_correctness():
    query = EvaluationQuery(
        query_id="q1",
        question="What is the token TTL?",
        reference_answer="Bearer tokens have a TTL of 3600 seconds.",
        ground_truth_chunk_ids=["c1"],
        answerable=True,
    )
    answer = GeneratedAnswer(
        answer_text="Bearer tokens have a strict TTL of 3600 seconds.",
        cited_chunk_ids=["c1"],
        is_abstention=False,
    )
    passages = [
        RetrievedPassage(chunk_id="c1", score=0.95, rank=1, text="Bearer tokens have a strict TTL of 3600 seconds."),
    ]
    gen_metrics = RAGEvaluator.evaluate_generation(answer, query, passages)

    assert gen_metrics.faithfulness >= 0.85
    assert gen_metrics.answer_correctness >= 0.85
    assert gen_metrics.citation_precision == 1.0
