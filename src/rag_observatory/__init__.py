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

__all__ = [
    "CorpusChunk",
    "EvaluationQuery",
    "RetrievedPassage",
    "GeneratedAnswer",
    "FailureMode",
    "DiagnosticResult",
    "RetrievalMetrics",
    "GenerationMetrics",
]
