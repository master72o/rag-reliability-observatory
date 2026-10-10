"""
Strongly-typed schemas for RAG Reliability Diagnostics and Benchmark Evaluation.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    DIRECT_FACTUAL = "direct_factual"
    MULTI_HOP = "multi_hop"
    NUMERICAL = "numerical"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL_DISTRACTOR = "adversarial_distractor"


class FailureMode(str, Enum):
    SUCCESS_GROUNDED = "success_grounded"
    RETRIEVAL_MISS = "retrieval_miss"
    EVIDENCE_RANKED_LOW = "evidence_ranked_low"
    CONTEXT_IGNORED = "context_ignored"
    CITATION_MISMATCH = "citation_mismatch"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    UNGROUNDED_HALLUCINATION = "ungrounded_hallucination"


class CorpusChunk(BaseModel):
    chunk_id: str
    document_title: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationQuery(BaseModel):
    query_id: str
    question: str
    reference_answer: str
    ground_truth_chunk_ids: List[str]
    ground_truth_spans: List[str] = Field(default_factory=list)
    answerable: bool = True
    query_type: QueryType = QueryType.DIRECT_FACTUAL
    difficulty: str = "medium"  # easy, medium, hard


class RetrievedPassage(BaseModel):
    chunk_id: str
    score: float
    rank: int
    text: str
    document_title: Optional[str] = None


class GeneratedAnswer(BaseModel):
    answer_text: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    cited_spans: List[str] = Field(default_factory=list)
    is_abstention: bool = False
    raw_response: Optional[str] = None


class RetrievalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    retrieved_count: int
    target_count: int


class GenerationMetrics(BaseModel):
    faithfulness: float
    answer_correctness: float
    citation_precision: float
    citation_recall: float
    abstention_accuracy: float


class DiagnosticResult(BaseModel):
    query_id: str
    query_type: QueryType
    failure_mode: FailureMode
    is_failure: bool
    diagnostic_explanation: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    retrieved_chunk_ids: List[str]
    target_chunk_ids: List[str]
    cited_chunk_ids: List[str]
