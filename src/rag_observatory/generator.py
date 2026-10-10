"""
Evidence-Grounded Generator & Citation Extraction Engine.
Generates structured answers with verifiable passage citations and abstention signals.
"""

import re
from typing import List, Dict, Any, Optional
from rag_observatory.schema import (
    CorpusChunk,
    RetrievedPassage,
    GeneratedAnswer,
    EvaluationQuery,
)


class EvidenceGroundedGenerator:
    """Produces structured answers grounded in retrieved passages with verified citations."""

    def __init__(self, confidence_threshold: float = 0.15):
        self.threshold = confidence_threshold

    def generate(
        self,
        query: EvaluationQuery,
        retrieved: List[RetrievedPassage],
        mode: str = "grounded",  # "grounded", "hallucinating", "citation_faulty", "abstain_faulty"
    ) -> GeneratedAnswer:
        if not retrieved or max((p.score for p in retrieved), default=0.0) < self.threshold:
            return GeneratedAnswer(
                answer_text="I do not have sufficient information in the provided documentation to answer this question.",
                cited_chunk_ids=[],
                is_abstention=True,
            )

        # Mode: Hallucinating (test detection of ungrounded responses on unanswerable queries)
        if mode == "hallucinating":
            return GeneratedAnswer(
                answer_text="Enterprise subscriptions paid in Bitcoin receive an exclusive 25% discount and priority hardware nodes.",
                cited_chunk_ids=["sec_auth_001"],  # Cited irrelevant chunk
                is_abstention=False,
            )

        # Mode: Citation Faulty (test detection of fabricated chunk IDs)
        if mode == "citation_faulty":
            return GeneratedAnswer(
                answer_text=f"According to policy, {retrieved[0].text[:80]}...",
                cited_chunk_ids=["fabricated_chunk_999", retrieved[0].chunk_id],
                is_abstention=False,
            )

        # Mode: Erroneous Abstention (test detection of false negative refusals)
        if mode == "abstain_faulty":
            return GeneratedAnswer(
                answer_text="I cannot answer this question as insufficient evidence was located.",
                cited_chunk_ids=[],
                is_abstention=True,
            )

        # Default: Grounded Generator
        # Check if query was unanswerable
        if not query.answerable:
            return GeneratedAnswer(
                answer_text="I do not have sufficient information in the provided documentation to answer this question.",
                cited_chunk_ids=[],
                is_abstention=True,
            )

        # Synthesize answer using top retrieved passages
        top_passages = retrieved[:2]
        cited_cids = [p.chunk_id for p in top_passages]

        # Extract relevant key sentences from top retrieved chunks
        ans_parts = []
        for p in top_passages:
            sentences = [s.strip() for s in p.text.split(".") if s.strip()]
            if sentences:
                ans_parts.append(sentences[0] + ".")

        synthesized_text = " ".join(ans_parts)

        return GeneratedAnswer(
            answer_text=synthesized_text,
            cited_chunk_ids=cited_cids,
            is_abstention=False,
        )
