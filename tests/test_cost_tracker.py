import pytest
from rag_observatory.cost_tracker import RAGCostEstimator


def test_estimate_query_cost():
    res = RAGCostEstimator.estimate_query_cost(prompt_tokens=1000, completion_tokens=200)
    assert res["prompt_tokens"] == 1000
    assert res["completion_tokens"] == 200
    assert res["total_tokens"] == 1200
    assert res["total_cost_usd"] > 0
    assert res["prompt_cost_usd"] == round(1000 * (0.15 / 1_000_000), 6)


def test_compute_efficiency_score():
    score_fast_accurate = RAGCostEstimator.compute_efficiency_score(ndcg=0.95, latency_ms=10.0, cost_usd=0.0001)
    score_slow_inaccurate = RAGCostEstimator.compute_efficiency_score(ndcg=0.50, latency_ms=100.0, cost_usd=0.001)
    assert score_fast_accurate > score_slow_inaccurate


def test_rank_retrieval_strategies():
    strategies = {
        "BM25": {"ndcg": 0.85, "latency_ms": 5.0, "prompt_tokens": 400},
        "Dense": {"ndcg": 0.90, "latency_ms": 25.0, "prompt_tokens": 500},
        "Hybrid": {"ndcg": 0.94, "latency_ms": 30.0, "prompt_tokens": 600},
    }
    ranked = RAGCostEstimator.rank_retrieval_strategies(strategies)
    assert len(ranked) == 3
    assert all("strategy" in item for item in ranked)
    assert all("efficiency_score" in item for item in ranked)
    # Check descending order of efficiency score
    assert ranked[0]["efficiency_score"] >= ranked[1]["efficiency_score"] >= ranked[2]["efficiency_score"]
