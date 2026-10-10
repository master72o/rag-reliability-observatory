"""
Unit tests for RAG Observatory retrievers: BM25, Dense, Hybrid RRF, and Reranker.
"""

import pytest
from rag_observatory.schema import CorpusChunk
from rag_observatory.retriever import (
    BM25Retriever,
    DenseVectorRetriever,
    HybridRRFRetriever,
    ContextualReranker,
    TextTokenizer,
)


@pytest.fixture
def sample_chunks():
    return [
        CorpusChunk(chunk_id="c1", document_title="Auth", text="OAuth 2.0 Bearer tokens have a strict TTL of 3600 seconds."),
        CorpusChunk(chunk_id="c2", document_title="Rate", text="Rate limits on production accounts are 10,000 requests per minute."),
        CorpusChunk(chunk_id="c3", document_title="Refund", text="Merchandise returns are permitted within 30 days of delivery."),
    ]


def test_tokenizer_filters_stopwords():
    tokens = TextTokenizer.tokenize("The OAuth 2.0 Bearer tokens have a strict TTL.")
    assert "the" not in tokens
    assert "oauth" in tokens
    assert "bearer" in tokens


def test_bm25_retrieval_exact_match(sample_chunks):
    retriever = BM25Retriever(sample_chunks)
    res = retriever.retrieve("What is the TTL for OAuth tokens?", top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "c1"


def test_dense_retrieval(sample_chunks):
    retriever = DenseVectorRetriever(sample_chunks)
    res = retriever.retrieve("customer refund and merchandise return policy", top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "c3"


def test_hybrid_rrf_combines(sample_chunks):
    bm25 = BM25Retriever(sample_chunks)
    dense = DenseVectorRetriever(sample_chunks)
    hybrid = HybridRRFRetriever(bm25, dense)
    res = hybrid.retrieve("rate limit quota", top_k=2)
    assert len(res) == 2
    assert res[0].chunk_id == "c2"


def test_contextual_reranker(sample_chunks):
    bm25 = BM25Retriever(sample_chunks)
    candidates = bm25.retrieve("OAuth 2.0 Bearer tokens", top_k=3)
    reranked = ContextualReranker.rerank("OAuth 2.0 Bearer tokens", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "c1"
