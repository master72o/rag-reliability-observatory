"""
RAG System Efficiency, Latency, and Token Cost Estimation Engine.
Analyzes the Pareto tradeoff between retrieval quality (MRR, nDCG) and inference overhead.
"""

from typing import Dict, Any, List, Optional


DEFAULT_PRICING_PER_MILLION = {
    "prompt_tokens": 0.15,      # $0.15 per 1M prompt tokens (e.g., Gemini Flash)
    "completion_tokens": 0.60,  # $0.60 per 1M completion tokens
}


class RAGCostEstimator:
    """Estimates financial cost, latency, and Pareto efficiency index for RAG strategies."""

    @staticmethod
    def estimate_query_cost(
        prompt_tokens: int,
        completion_tokens: int,
        pricing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculates USD cost for a RAG inference query based on prompt and completion token counts."""
        rates = pricing or DEFAULT_PRICING_PER_MILLION
        p_rate = rates.get("prompt_tokens", 0.15) / 1_000_000.0
        c_rate = rates.get("completion_tokens", 0.60) / 1_000_000.0

        prompt_cost = prompt_tokens * p_rate
        completion_cost = completion_tokens * c_rate
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_cost_usd": round(prompt_cost, 6),
            "completion_cost_usd": round(completion_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }

    @staticmethod
    def compute_efficiency_score(ndcg: float, latency_ms: float, cost_usd: float) -> float:
        """
        Computes composite efficiency score:
        Higher is better. Rewards high nDCG while penalizing excessive latency and token cost.
        """
        if latency_ms <= 0:
            latency_ms = 1.0
        # Normalize: nDCG / (log(latency) * normalized_cost)
        cost_factor = max(0.0001, cost_usd * 1000)
        score = (ndcg * 100.0) / ((latency_ms ** 0.5) * (1.0 + cost_factor))
        return round(float(score), 4)

    @classmethod
    def rank_retrieval_strategies(
        cls, strategies_data: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks multiple retrieval strategies (e.g. BM25, Dense, Hybrid, Rerank)
        by their composite efficiency score.
        """
        rankings = []
        for name, metrics in strategies_data.items():
            ndcg = metrics.get("ndcg", 0.0)
            latency = metrics.get("latency_ms", 10.0)
            tokens = int(metrics.get("prompt_tokens", 500))
            cost_info = cls.estimate_query_cost(tokens, completion_tokens=150)
            cost = cost_info["total_cost_usd"]

            eff_score = cls.compute_efficiency_score(ndcg, latency, cost)
            rankings.append({
                "strategy": name,
                "ndcg": ndcg,
                "latency_ms": latency,
                "cost_usd": cost,
                "efficiency_score": eff_score,
            })

        rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
        return rankings
