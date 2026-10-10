"""
Builds realistic knowledge corpus and evaluation queries for RAG Reliability Observatory.
"""

import json
from pathlib import Path

data_dir = Path("data")
corpus_dir = data_dir / "corpus"
queries_dir = data_dir / "queries"
corpus_dir.mkdir(parents=True, exist_ok=True)
queries_dir.mkdir(parents=True, exist_ok=True)

# 1. Corpus Chunks
corpus_chunks = [
    {
        "chunk_id": "sec_auth_001",
        "document_title": "Enterprise API Authentication & Token Lifecycle",
        "text": "All API requests must supply an OAuth 2.0 Bearer token in the Authorization header. Bearer tokens have a strict TTL of 3600 seconds (1 hour). Refresh tokens remain valid for 30 days and must be rotated upon each refresh invocation. Any token with an expired timestamp immediately returns HTTP 401 Unauthorized with error code TOKEN_EXPIRED."
    },
    {
        "chunk_id": "sec_rate_002",
        "document_title": "API Gateway Rate Limiting & Quotas",
        "text": "Production tier accounts are allocated a maximum of 10,000 requests per minute with a burst concurrency limit of 250 requests. If the rate limit is exceeded, the gateway issues HTTP 429 Too Many Requests alongside a Retry-After header indicating backoff in seconds. Sandbox tiers are throttled to 60 requests per minute."
    },
    {
        "chunk_id": "pol_refund_003",
        "document_title": "Customer Refund & Order Cancellation Policy",
        "text": "Customers may request a full refund within 30 days of delivery if the merchandise is unopened in original packaging. Software licenses and downloadable media are non-refundable once the activation key has been viewed or generated. Expedited shipping fees are non-refundable under all circumstances."
    },
    {
        "chunk_id": "pol_return_004",
        "document_title": "Defective Hardware Return & RMA Workflow",
        "text": "Hardware reported defective within 90 days of purchase qualifies for advance RMA replacement. The merchant covers return shipping costs via a prepaid label. Replacements ship via 2-day priority freight within 24 hours of receiving the RMA tracking confirmation."
    },
    {
        "chunk_id": "arch_cache_005",
        "document_title": "Distributed Cache Consistency & Invalidation",
        "text": "The platform utilizes a Redis cluster deployed in a multi-AZ read-replica setup. Cache invalidation follows a Cache-Aside pattern with event-driven change data capture via Debezium. When a database record updates, invalidation events purge corresponding cache keys within a p99 SLA of 85 milliseconds."
    },
    {
        "chunk_id": "arch_failover_006",
        "document_title": "Database High Availability & Automatic Failover",
        "text": "PostgreSQL databases operate with synchronous streaming replication. In the event of primary node failure, AWS Aurora performs automatic failover to the replica within 30 to 60 seconds. The application pool automatically reconnects via DNS endpoint routing without requiring container restart."
    },
    {
        "chunk_id": "ops_retention_007",
        "document_title": "Data Governance, PII Scrubbing & Retention Schedule",
        "text": "Customer transactional logs are retained for exactly 7 years to satisfy financial audit compliance. PII fields including credit card numbers and social security numbers are tokenized at ingestion and permanently deleted from analytical data warehouses after 90 days."
    },
    {
        "chunk_id": "ops_encryption_008",
        "document_title": "Cryptographic Key Management & Storage Security",
        "text": "All persistent disk storage is encrypted at rest using AES-256 via AWS KMS customer-managed keys (CMK). Cryptographic keys are rotated automatically every 365 days. Data in transit requires TLS 1.3 encryption across all public and inter-service VPC endpoints."
    },
    {
        "chunk_id": "ml_governance_009",
        "document_title": "AI Model Safety & Hallucination Quality Gates",
        "text": "Production LLM applications must pass automated evaluation before deployment. The release quality gate mandates a minimum factual precision of 0.85 and a maximum hallucination rate of 0.05 on held-out gold test sets. Models exceeding a p95 inference latency of 800ms are barred from production."
    },
    {
        "chunk_id": "ml_rlhf_010",
        "document_title": "Human Preference Calibration & RLHF Guidelines",
        "text": "Human preference annotations require a minimum of 3 independent annotators per prompt. Pairwise preference data is filtered using consensus voting; pairs with less than 66% annotator agreement are routed to SME adjudication before ingestion into DPO alignment pipelines."
    }
]

with open(corpus_dir / "knowledge_corpus.jsonl", "w", encoding="utf-8") as f:
    for c in corpus_chunks:
        f.write(json.dumps(c) + "\n")

# 2. Benchmark Queries
queries = [
    # A. Direct Factual Lookup (Single Hop)
    {
        "query_id": "q_direct_001",
        "question": "What is the expiration time (TTL) of OAuth 2.0 Bearer tokens?",
        "reference_answer": "Bearer tokens have a strict TTL of 3600 seconds (1 hour).",
        "ground_truth_chunk_ids": ["sec_auth_001"],
        "answerable": True,
        "query_type": "direct_factual",
        "difficulty": "easy"
    },
    {
        "query_id": "q_direct_002",
        "question": "What happens if a client exceeds the API rate limit on production accounts?",
        "reference_answer": "The API gateway issues an HTTP 429 Too Many Requests response with a Retry-After header indicating backoff in seconds.",
        "ground_truth_chunk_ids": ["sec_rate_002"],
        "answerable": True,
        "query_type": "direct_factual",
        "difficulty": "easy"
    },
    {
        "query_id": "q_direct_003",
        "question": "Are software licenses eligible for a refund after the activation key is viewed?",
        "reference_answer": "No, software licenses and downloadable media are non-refundable once the activation key has been viewed or generated.",
        "ground_truth_chunk_ids": ["pol_refund_003"],
        "answerable": True,
        "query_type": "direct_factual",
        "difficulty": "easy"
    },
    {
        "query_id": "q_direct_004",
        "question": "What is the p99 SLA for purging cache keys when a database record updates?",
        "reference_answer": "The p99 SLA for invalidating cache keys is 85 milliseconds.",
        "ground_truth_chunk_ids": ["arch_cache_005"],
        "answerable": True,
        "query_type": "direct_factual",
        "difficulty": "easy"
    },

    # B. Multi-Hop Synthesis Queries (Requires combining 2 distinct chunks)
    {
        "query_id": "q_multihop_005",
        "question": "What is the return window for unopened merchandise versus defective hardware?",
        "reference_answer": "Unopened merchandise can be returned within 30 days of delivery, whereas defective hardware qualifies for return within 90 days of purchase under the advance RMA policy.",
        "ground_truth_chunk_ids": ["pol_refund_003", "pol_return_004"],
        "answerable": True,
        "query_type": "multi_hop",
        "difficulty": "medium"
    },
    {
        "query_id": "q_multihop_006",
        "question": "What cryptographic standards protect data at rest and data in transit?",
        "reference_answer": "Data at rest is protected by AES-256 via AWS KMS keys rotated every 365 days, while data in transit requires TLS 1.3 encryption.",
        "ground_truth_chunk_ids": ["ops_encryption_008"],
        "answerable": True,
        "query_type": "multi_hop",
        "difficulty": "medium"
    },

    # C. Unanswerable Queries (Crucial for testing Abstention & Hallucination)
    {
        "query_id": "q_unanswerable_007",
        "question": "What is the discount rate for enterprise annual subscriptions paid in Bitcoin?",
        "reference_answer": "I do not have sufficient information in the provided documentation to answer this question.",
        "ground_truth_chunk_ids": [],
        "answerable": False,
        "query_type": "unanswerable",
        "difficulty": "hard"
    },
    {
        "query_id": "q_unanswerable_008",
        "question": "What are the GPU requirements for hosting the LLaMA-3 70B model locally?",
        "reference_answer": "I do not have sufficient information in the provided documentation to answer this question.",
        "ground_truth_chunk_ids": [],
        "answerable": False,
        "query_type": "unanswerable",
        "difficulty": "hard"
    },

    # D. Adversarial Lexical Distractor Queries (Tricks BM25 or Dense matching)
    {
        "query_id": "q_adversarial_009",
        "question": "How long are customer transaction financial audit logs stored before deletion?",
        "reference_answer": "Transactional logs are retained for exactly 7 years to satisfy financial audit compliance.",
        "ground_truth_chunk_ids": ["ops_retention_007"],
        "answerable": True,
        "query_type": "adversarial_distractor",
        "difficulty": "hard"
    },
    {
        "query_id": "q_adversarial_010",
        "question": "What agreement percentage is required before pairwise preferences proceed to DPO alignment without adjudication?",
        "reference_answer": "A minimum of 66% annotator agreement is required; pairs with less than 66% agreement are routed to SME adjudication.",
        "ground_truth_chunk_ids": ["ml_rlhf_010"],
        "answerable": True,
        "query_type": "adversarial_distractor",
        "difficulty": "hard"
    }
]

with open(queries_dir / "benchmark_queries.jsonl", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")

print(f"Generated {len(corpus_chunks)} corpus chunks and {len(queries)} benchmark queries.")
