"""
Unit tests for rag_observatory top-level package exports.
"""

import rag_observatory


def test_top_level_exports():
    expected_exports = [
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
    for export_name in expected_exports:
        assert hasattr(rag_observatory, export_name), f"Missing export: {export_name}"
        assert export_name in rag_observatory.__all__
