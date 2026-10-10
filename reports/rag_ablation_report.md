# 🔬 RAG Reliability Observatory: Architecture Ablation Report

| Strategy | Success Rate (%) | Recall@3 | nDCG@3 | Faithfulness | Citation Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bm25`** | `40.0%` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **`dense`** | `40.0%` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **`hybrid`** | `20.0%` | `1.0000` | `1.0000` | `0.2000` | `0.2000` |
| **`hybrid_reranked`** | `30.0%` | `1.0000` | `1.0000` | `0.9000` | `0.9000` |

---

## Failure Mode Attribution Breakdown

### Strategy: `bm25`
- **`context_ignored`**: 6 instances
- **`success_grounded`**: 4 instances

### Strategy: `dense`
- **`context_ignored`**: 6 instances
- **`success_grounded`**: 4 instances

### Strategy: `hybrid`
- **`incorrect_abstention`**: 8 instances
- **`success_grounded`**: 2 instances

### Strategy: `hybrid_reranked`
- **`context_ignored`**: 6 instances
- **`incorrect_abstention`**: 1 instances
- **`success_grounded`**: 3 instances

