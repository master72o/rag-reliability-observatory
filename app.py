"""
RAG Reliability Observatory: Interactive Streamlit Diagnostics Dashboard.
Compatible with Streamlit Community Cloud and Hugging Face Spaces.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from rag_observatory.schema import CorpusChunk, EvaluationQuery, FailureMode
from rag_observatory.benchmark_runner import BenchmarkRunner
from rag_observatory.retriever import TextTokenizer

st.set_page_config(
    page_title="RAG Reliability Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #2d3748;
    }
    .badge-success {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-error {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> tuple[List[CorpusChunk], List[EvaluationQuery]]:
    c_path = Path("data/corpus/knowledge_corpus.jsonl")
    q_path = Path("data/queries/benchmark_queries.jsonl")

    chunks = []
    if c_path.exists():
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(CorpusChunk.model_validate_json(line.strip()))

    queries = []
    if q_path.exists():
        with open(q_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(EvaluationQuery.model_validate_json(line.strip()))

    return chunks, queries


chunks, queries = load_data()

# Initialize Runner
runner = BenchmarkRunner(chunks)

# Sidebar
st.sidebar.title("🔭 RAG Observatory")
st.sidebar.caption("Diagnostic & Failure Attribution Lab v2.0")
st.sidebar.markdown("---")
top_k_select = st.sidebar.slider("Retrieval Depth (Top-K)", 1, 5, 3)
selected_strategy = st.sidebar.selectbox(
    "Evaluation Architecture:",
    ["hybrid_reranked", "hybrid", "dense", "bm25"],
    format_func=lambda x: {
        "hybrid_reranked": "Hybrid RRF + Contextual Reranker",
        "hybrid": "Hybrid RRF (Lexical + Dense)",
        "dense": "Dense Semantic Cosine",
        "bm25": "BM25 Lexical Keyword",
    }[x],
)
st.sidebar.markdown("---")
st.sidebar.info("""
**Diagnostic Capabilities**:
- Decoupled Retrieval vs Generation
- Reciprocal Rank Fusion (RRF)
- Hallucination vs Abstention Detection
- Ground-Truth Citation Verification
""")

# Run Full Ablation Study
@st.cache_data
def get_ablation_results(top_k: int):
    return runner.run_ablation_study(queries, top_k=top_k)

ablation = get_ablation_results(top_k_select)
current_stats = ablation[selected_strategy]

# Header
st.title("🔭 RAG Reliability Observatory")
st.markdown("A diagnostic platform that uncovers **why** retrieval-augmented generation systems fail, separating retrieval misses, ranking degradation, context ignoring, and citation fabrications.")

# Top KPIs
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Corpus Passages", len(chunks))
with k2:
    st.metric("Benchmark Queries", len(queries))
with k3:
    st.metric("Success Rate", f"{current_stats['success_rate_pct']}%")
with k4:
    st.metric("Mean Recall@K", f"{current_stats['mean_recall_at_k']:.3f}")
with k5:
    st.metric("Mean nDCG@K", f"{current_stats['mean_ndcg']:.3f}")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Multi-Strategy Ablation Study",
    "⚠️ Failure Attribution Explorer",
    "🔍 Query Deep-Dive & Citation Inspector",
    "🧪 Live Diagnostic Sandbox",
])

# TAB 1: ABLATION STUDY
with tab1:
    st.subheader(f"Architecture Comparison across {len(queries)} Benchmark Queries (Top-K={top_k_select})")
    st.write("Controlled ablation testing BM25 Keyword, Dense Semantic, Hybrid RRF, and Reranking across identical queries.")

    ab_df = pd.DataFrame([
        {
            "Strategy": {
                "bm25": "BM25 Lexical",
                "dense": "Dense Semantic",
                "hybrid": "Hybrid RRF",
                "hybrid_reranked": "Hybrid RRF + Reranker",
            }[k],
            "Success Rate (%)": v["success_rate_pct"],
            "Recall@K": v["mean_recall_at_k"],
            "Precision@K": v["mean_precision_at_k"],
            "MRR": v["mean_mrr"],
            "nDCG@K": v["mean_ndcg"],
            "Faithfulness": v["mean_faithfulness"],
            "Citation Precision": v["mean_citation_precision"],
        }
        for k, v in ablation.items()
    ])
    st.dataframe(ab_df, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Retrieval Quality Comparison (nDCG vs Recall)")
        st.bar_chart(ab_df.set_index("Strategy")[["Recall@K", "nDCG@K"]])
    with colB:
        st.markdown("#### End-to-End Success Rate (%)")
        st.bar_chart(ab_df.set_index("Strategy")["Success Rate (%)"])

# TAB 2: FAILURE ATTRIBUTION EXPLORER
with tab2:
    st.subheader(f"Failure Attribution Breakdown — `{selected_strategy}`")
    st.write("Identifies which subsystem caused each observed error.")

    fail_dist = current_stats["failure_distribution"]
    col_f1, col_f2 = st.columns([1, 2])

    with col_f1:
        st.markdown("#### Failure Distribution")
        f_df = pd.DataFrame(list(fail_dist.items()), columns=["Failure Mode", "Count"]).set_index("Failure Mode")
        st.dataframe(f_df, use_container_width=True)

    with col_f2:
        st.markdown("#### Taxonomy Breakdown Chart")
        st.bar_chart(f_df)

    st.markdown("---")
    st.subheader("Diagnostic Query Log")
    diag_records = current_stats["detailed_diagnostics"]
    selected_failure_filter = st.selectbox(
        "Filter by Failure Mode:",
        ["ALL"] + sorted(list(fail_dist.keys())),
    )

    filtered_diags = [
        d for d in diag_records
        if selected_failure_filter == "ALL" or d["failure_mode"] == selected_failure_filter
    ]

    log_df = pd.DataFrame([
        {
            "Query ID": d["query_id"],
            "Query Type": d["query_type"],
            "Failure Mode": d["failure_mode"],
            "Status": "❌ FAILED" if d["is_failure"] else "✅ SUCCESS",
            "Recall@K": d["retrieval"]["recall_at_k"],
            "nDCG": d["retrieval"]["ndcg"],
            "Faithfulness": d["generation"]["faithfulness"],
            "Root Cause Explanation": d["diagnostic_explanation"],
        }
        for d in filtered_diags
    ])
    st.dataframe(log_df, use_container_width=True)

# TAB 3: QUERY DEEP-DIVE
with tab3:
    st.subheader("Query Deep-Dive & Citation Inspector")
    st.write("Examine the exact evidence flow for any individual benchmark query.")

    q_map = {q.query_id: q for q in queries}
    selected_qid = st.selectbox(
        "Select Query:",
        list(q_map.keys()),
        format_func=lambda x: f"{x}: {q_map[x].question[:90]}...",
    )

    if selected_qid:
        target_q = q_map[selected_qid]
        res = runner.run_query(target_q, strategy=selected_strategy, top_k=top_k_select)

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown(f"**Question:** *\"{target_q.question}\"*")
            st.markdown(f"**Query Type:** `{target_q.query_type.value}` | **Answerable:** `{target_q.answerable}`")
            st.markdown(f"**Reference Answer:**\n> {target_q.reference_answer}")
            st.markdown(f"**Ground Truth Evidence Chunks:** `{target_q.ground_truth_chunk_ids}`")

        with col_q2:
            st.markdown(f"**Diagnostic Verdict:** `{res.failure_mode.value.upper()}`")
            st.markdown(f"**Root Cause Analysis:**\n> {res.diagnostic_explanation}")
            status_box = "badge-success" if not res.is_failure else "badge-error"
            st.markdown(f"<span class='{status_box}'>Status: {'PASSED' if not res.is_failure else 'FAILED'}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Retrieved Evidence Passages")
        retrieved_passages = runner.hybrid.retrieve(target_q.question, top_k=top_k_select)
        for p in retrieved_passages:
            is_target = p.chunk_id in target_q.ground_truth_chunk_ids
            prefix = "⭐ [GROUND TRUTH EVIDENCE]" if is_target else "📄 [RETRIEVED CONTEXT]"
            with st.expander(f"{prefix} Rank {p.rank} — `{p.chunk_id}`: {p.document_title} (Score: {p.score:.4f})", expanded=is_target):
                st.write(p.text)

# TAB 4: LIVE DIAGNOSTIC SANDBOX
with tab4:
    st.subheader("Live Diagnostic Sandbox")
    st.write("Enter a custom prompt to test retrieval, evidence ranking, and grounded answer synthesis against the corpus.")

    user_query = st.text_input("Enter Question:", "What is the token expiration TTL for OAuth 2.0?")
    custom_strat = st.selectbox("Strategy for Sandbox:", ["hybrid_reranked", "hybrid", "dense", "bm25"], key="sandbox_strat")

    if st.button("Run Diagnostic Retrieval"):
        sim_q = EvaluationQuery(
            query_id="sandbox_query",
            question=user_query,
            reference_answer="",
            ground_truth_chunk_ids=[],
            answerable=True,
        )
        res = runner.run_query(sim_q, strategy=custom_strat, top_k=top_k_select)

        st.success("Retrieval & Synthesis Complete!")
        st.markdown("#### Retrieved Evidence:")
        raw_retrieved = runner.bm25.retrieve(user_query, top_k=top_k_select) if custom_strat == "bm25" else runner.hybrid.retrieve(user_query, top_k=top_k_select)
        for p in raw_retrieved:
            st.write(f"- **Rank {p.rank}** (`{p.chunk_id}` - Score: {p.score:.4f}): {p.text}")
