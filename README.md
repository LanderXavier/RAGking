# RAGking v2.1 👑
### *Lightweight Multidimensional Evaluation Framework for RAG Pipelines*

**RAGking** is an automated evaluation framework designed to audit the quality, cost, and operational efficiency of Retrieval-Augmented Generation (RAG) systems. Unlike generalized evaluation approaches, RAGking consolidates factual veracity metrics (via *LLM-as-a-Judge*), token efficiency, latency, maintenance complexity, and failure rates into a single, standardized **RAGking Score**.

---

## 🚀 Key Features

* **Multidimensional Scoring:** Evaluates pipelines across 5 normalized dimensions: Correctness ($\hat{C}$), Economic Efficiency ($\hat{E}$), Latency ($\hat{L}$), Maintenance ($\hat{M}$), and Failure Rate ($\hat{F}$).
* **Dynamic Weighting:** Allows researchers to assign specific mathematical weights ($w$) to each variable based on the domain's strictness (e.g., prioritizing factual correctness in legal contexts).
* **Operational Telemetry:** Deterministically maps token consumption, API costs, and end-to-end execution latency.
* **Automated and Programmatic:** Designed to execute on datasets in CSV/Excel format via efficient Python pipelines.

---

## 📐 RAGking Evaluation Formulas

The framework relies on a robust set of equations to calculate the operational and economic viability of the generative system.

### 1. Cost and Efficiency Metrics
To determine the cost-effectiveness and token usage of each query, the following formulas are applied:

**Cost per Query (USD):**
$$Cost_{query} = \frac{T_{in} \cdot P_{in} + T_{out} \cdot P_{out}}{10^6}$$
*(Where $T$ represents tokens and $P$ represents the price per million tokens for input and output).*

**Economic Efficiency:**
$$Economic\ Efficiency = \frac{Correctness}{Cost_{query}}$$

**Token Efficiency:**
$$Token\ Efficiency = \frac{T_{in} + T_{out}}{\max(Correctness, 1)}$$

**Failure Rate:**
$$Failure\ Rate = \frac{N_{fail}}{N_{total}} \times 100$$

### 2. The RAGking Score
The definitive multidimensional system score (0-100) is calculated using the weighted sum of the normalized dimensions ($\hat{C}, \hat{E}, \hat{L}, \hat{M}, \hat{F}$):

$$RAGking\ Score = (w_C \cdot \hat{C} + w_E \cdot \hat{E} + w_L \cdot \hat{L} + w_M \cdot \hat{M} + w_F \cdot \hat{F}) \times 100$$

---
