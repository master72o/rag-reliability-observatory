"""
Multi-layer Evaluation Engine computing decoupled Retrieval and Generation metrics.
"""

import math
from typing import List, Set
from rag_observatory.schema import (
    RetrievedPassage,
    GeneratedAnswer,
    EvaluationQuery,
    RetrievalMetrics,
    GenerationMetrics,
)
from rag_observatory.retriever import TextTokenizer


class RAGEvaluator:
    """Evaluates decoupled retrieval performance and generation quality."""

    @staticmethod
    def evaluate_retrieval(
        retrieved: List[RetrievedPassage], target_chunk_ids: List[str], k: int = 5
    ) -> RetrievalMetrics:
        retrieved_k = retrieved[:k]
        target_set = set(target_chunk_ids)

        if not target_set:
            # Query is unanswerable / has no target chunks
            return RetrievalMetrics(
                precision_at_k=1.0,
                recall_at_k=1.0,
                mrr=1.0,
                ndcg=1.0,
                retrieved_count=len(retrieved_k),
                target_count=0,
            )

        ret_ids = [p.chunk_id for p in retrieved_k]
        hits = [1 if cid in target_set else 0 for cid in ret_ids]

        # Precision@k & Recall@k
        prec = sum(hits) / max(1, len(ret_ids))
        rec = sum(hits) / max(1, len(target_set))

        # MRR
        mrr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h == 1:
                mrr = 1.0 / rank
                break

        # nDCG@k
        dcg = 0.0
        for rank, h in enumerate(hits, start=1):
            if h == 1:
                dcg += 1.0 / math.log2(rank + 1)

        idcg = 0.0
        for rank in range(1, min(len(target_set), k) + 1):
            idcg += 1.0 / math.log2(rank + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return RetrievalMetrics(
            precision_at_k=round(prec, 4),
            recall_at_k=round(rec, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            retrieved_count=len(retrieved_k),
            target_count=len(target_set),
        )

    @staticmethod
    def evaluate_generation(
        answer: GeneratedAnswer,
        query: EvaluationQuery,
        retrieved: List[RetrievedPassage],
    ) -> GenerationMetrics:
        # Case 1: Unanswerable prompt evaluation (Abstention check)
        if not query.answerable:
            abstention_acc = 1.0 if answer.is_abstention else 0.0
            return GenerationMetrics(
                faithfulness=1.0 if answer.is_abstention else 0.0,
                answer_correctness=1.0 if answer.is_abstention else 0.0,
                citation_precision=1.0 if answer.is_abstention else 0.0,
                citation_recall=1.0 if answer.is_abstention else 0.0,
                abstention_accuracy=abstention_acc,
            )

        # Case 2: Erroneous refusal on answerable query
        if answer.is_abstention and query.answerable:
            return GenerationMetrics(
                faithfulness=0.0,
                answer_correctness=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                abstention_accuracy=0.0,
            )

        # Case 3: Grounded Answer Evaluation
        ans_toks = set(TextTokenizer.tokenize(answer.answer_text))
        ref_toks = set(TextTokenizer.tokenize(query.reference_answer))

        # Answer correctness via token overlap
        overlap_ref = len(ans_toks.intersection(ref_toks))
        ans_corr = (overlap_ref / len(ref_toks)) if ref_toks else 1.0

        # Context faithfulness: check answer token coverage in retrieved passages
        ret_text = " ".join(p.text for p in retrieved)
        ret_toks = set(TextTokenizer.tokenize(ret_text))
        overlap_ctx = len(ans_toks.intersection(ret_toks))
        faithfulness = (overlap_ctx / len(ans_toks)) if ans_toks else 1.0

        # Citation validation: do cited IDs exist in retrieved passages?
        ret_cids = {p.chunk_id for p in retrieved}
        valid_citations = [c for c in answer.cited_chunk_ids if c in ret_cids]
        cit_prec = (len(valid_citations) / len(answer.cited_chunk_ids)) if answer.cited_chunk_ids else 0.0

        # Citation recall against ground truth target chunks
        target_cids = set(query.ground_truth_chunk_ids)
        cit_rec = (len(set(answer.cited_chunk_ids).intersection(target_cids)) / len(target_cids)) if target_cids else 1.0

        return GenerationMetrics(
            faithfulness=round(min(1.0, faithfulness), 4),
            answer_correctness=round(min(1.0, ans_corr), 4),
            citation_precision=round(min(1.0, cit_prec), 4),
            citation_recall=round(min(1.0, cit_rec), 4),
            abstention_accuracy=1.0,
        )
