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

## 📐 Evaluation and Normalization Metrics

The framework consolidates three critical dimensions in the calculation of the **Final RQI**:

1. **Factual Correctness Score (S_C):** Based on an automated evaluation rubric from 1 to 5 and linearly normalized:
   S_C = ((Score_LLM - 1) / (5 - 1)) * 10

2. **Token Efficiency Score (S_T):** Inverted Min-Max scaling to reward prompt size optimization and API cost efficiency:
   S_T = 10 * ((T_max - T_i) / (T_max - T_min))

3. **Latency Efficiency Score (S_L):** Inverted Min-Max scaling to evaluate the system's response speed in milliseconds:
   S_L = 10 * ((L_max - L_i) / (L_max - L_min))

### Response Quality Index (RQI)
The definitive system score is the arithmetic mean of the three scaled scores:
RQI_i = (S_C,i + S_T,i + S_L,i) / 3

---
