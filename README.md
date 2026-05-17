# RAGking v2.1 👑
### *Lightweight Multidimensional Evaluation Framework for RAG Pipelines*

**RAGking** is an automated evaluation framework designed to audit the quality, cost, and operational efficiency of Retrieval-Augmented Generation (RAG) systems. Unlike generalized evaluation approaches, RAGking consolidates factual veracity metrics (via *LLM-as-a-Judge*) and computational telemetry into a single, standardized **Response Quality Index (RQI)**.

---

## 🚀 Key Features

* **Strict Factual Evaluation (1-5 Scale):** Specifically designed for highly sensitive domains (such as the legal field), severely penalizing factual hallucinations over computational costs.
* **Operational Telemetry:** Deterministically maps token consumption and end-to-end execution latency.
* **Multidimensional Commensurability:** Linearly normalizes heterogeneous variables (time, tokens, correctness) to a common `0 to 10` scale for transparent trade-off analysis.
* **Automated and Programmatic:** Designed to execute on datasets in CSV/Excel format via efficient Python pipelines.



---
