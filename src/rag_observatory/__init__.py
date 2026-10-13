"""
RAG Reliability Observatory: Diagnostic Platform for Retrieval-Augmented Generation Systems.
"""

__version__ = "2.0.0"
__author__ = "Munir (master72o)"

from rag_observatory.schema import (
    CorpusChunk,
    EvaluationQuery,
    RetrievedPassage,
    GeneratedAnswer,
    FailureMode,
    DiagnosticResult,
    RetrievalMetrics,
    GenerationMetrics,
)
from rag_observatory.evaluator import RAGEvaluator
from rag_observatory.retriever import (
    TextTokenizer,
    BM25Retriever,
    DenseVectorRetriever,
    HybridRRFRetriever,
    ContextualReranker,
)
from rag_observatory.generator import EvidenceGroundedGenerator
from rag_observatory.failure_attributor import FailureAttributor

__all__ = [
    "CorpusChunk",
    "EvaluationQuery",
    "RetrievedPassage",
    "GeneratedAnswer",
    "FailureMode",
    "DiagnosticResult",
    "RetrievalMetrics",
    "GenerationMetrics",
    "RAGEvaluator",
    "TextTokenizer",
    "BM25Retriever",
    "DenseVectorRetriever",
    "HybridRRFRetriever",
    "ContextualReranker",
    "EvidenceGroundedGenerator",
    "FailureAttributor",
]
