"""
Decoupled Multi-Strategy Retrieval Engine: BM25 Lexical, Dense Vector, Hybrid RRF, and Reranking.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from rag_observatory.schema import CorpusChunk, RetrievedPassage


class TextTokenizer:
    """Standardized tokenizer for BM25 and vocabulary indexing."""

    STOPWORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "of",
        "is", "are", "was", "were", "be", "been", "by", "this", "that", "it", "from",
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = clean.split()
        return [t for t in tokens if len(t) > 1 and t not in cls.STOPWORDS]


class BM25Retriever:
    """Okapi BM25 Lexical Retrieval Implementation."""

    def __init__(self, chunks: List[CorpusChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)

        self.doc_tokens = [TextTokenizer.tokenize(c.text) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(1, self.N)

        # Document frequencies
        self.df: Dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for t in unique_terms:
                self.df[t] += 1

        # IDF lookup
        self.idf: Dict[str, float] = {}
        for t, freq in self.df.items():
            self.idf[t] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        q_tokens = TextTokenizer.tokenize(query)
        if not q_tokens:
            return []

        scores = []
        for idx, (chunk, doc_toks) in enumerate(zip(self.chunks, self.doc_tokens)):
            dl = self.doc_lens[idx]
            tf_counts = Counter(doc_toks)
            score = 0.0

            for q_term in q_tokens:
                if q_term in tf_counts:
                    tf = tf_counts[q_term]
                    idf_val = self.idf.get(q_term, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self.avg_dl)))
                    score += idf_val * (numerator / max(1e-6, denominator))

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, sc) in enumerate(scores[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    score=round(sc, 4),
                    rank=rank,
                    text=chunk.text,
                    document_title=chunk.document_title,
                )
            )
        return results


class DenseVectorRetriever:
    """Dense Semantic Vector Retrieval (TF-IDF weighted Cosine Vector Space)."""

    def __init__(self, chunks: List[CorpusChunk]):
        self.chunks = chunks
        self.N = len(chunks)
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        all_tokens = [TextTokenizer.tokenize(c.text) for c in chunks]
        df = Counter()
        for doc in all_tokens:
            for term in set(doc):
                df[term] += 1

        for term, freq in df.items():
            idx = len(self.vocab)
            self.vocab[term] = idx
            self.idf[term] = math.log(1.0 + (self.N / (1.0 + freq)))

        self.V = len(self.vocab)
        # Build L2-normalized doc vectors
        self.doc_vectors = []
        for doc in all_tokens:
            vec = self._vectorize(doc)
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens: List[str]) -> Dict[int, float]:
        tf = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for t, count in tf.items():
            if t in self.vocab:
                dim = self.vocab[t]
                val = (1.0 + math.log(count)) * self.idf[t]
                vec[dim] = val
                norm_sq += val * val

        norm = math.sqrt(norm_sq)
        if norm > 0:
            for dim in vec:
                vec[dim] /= norm
        return vec

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        q_tokens = TextTokenizer.tokenize(query)
        q_vec = self._vectorize(q_tokens)
        if not q_vec:
            return []

        scores = []
        for idx, d_vec in enumerate(self.doc_vectors):
            dot = sum(q_vec[dim] * d_vec.get(dim, 0.0) for dim in q_vec)
            scores.append((idx, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, sc) in enumerate(scores[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    score=round(sc, 4),
                    rank=rank,
                    text=chunk.text,
                    document_title=chunk.document_title,
                )
            )
        return results


class HybridRRFRetriever:
    """Hybrid Retrieval fusing Lexical (BM25) and Dense using Reciprocal Rank Fusion."""

    def __init__(self, bm25: BM25Retriever, dense: DenseVectorRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.chunk_map = {c.chunk_id: c for c in bm25.chunks}

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        lex_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = Counter()
        for p in lex_results:
            rrf_scores[p.chunk_id] += 1.0 / (self.rrf_k + p.rank)
        for p in dense_results:
            rrf_scores[p.chunk_id] += 1.0 / (self.rrf_k + p.rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (cid, score) in enumerate(sorted_items[:top_k], start=1):
            chunk = self.chunk_map[cid]
            results.append(
                RetrievedPassage(
                    chunk_id=cid,
                    score=round(score, 5),
                    rank=rank,
                    text=chunk.text,
                    document_title=chunk.document_title,
                )
            )
        return results


class ContextualReranker:
    """Reranker assessing query phrase proximity and coverage over top candidates."""

    @staticmethod
    def rerank(query: str, passages: List[RetrievedPassage], top_k: int = 5) -> List[RetrievedPassage]:
        q_tokens = TextTokenizer.tokenize(query)
        if not q_tokens:
            return passages[:top_k]

        reranked = []
        for p in passages:
            doc_toks = TextTokenizer.tokenize(p.text)
            coverage = len(set(q_tokens).intersection(set(doc_toks))) / max(1, len(q_tokens))
            
            # Phrase bonus if exact query substring occurs
            exact_bonus = 0.35 if query.lower().strip() in p.text.lower() else 0.0
            
            combined_score = (p.score * 0.5) + (coverage * 0.35) + exact_bonus
            reranked.append((p, combined_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (p, sc) in enumerate(reranked[:top_k], start=1):
            results.append(
                RetrievedPassage(
                    chunk_id=p.chunk_id,
                    score=round(sc, 4),
                    rank=rank,
                    text=p.text,
                    document_title=p.document_title,
                )
            )
        return results
