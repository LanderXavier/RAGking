# RAGking v2.1 👑
### *Lightweight Multidimensional Evaluation Framework for RAG Pipelines*

**RAGking** es un framework de evaluación automatizada diseñado para auditar la calidad, el costo y la eficiencia operativa de sistemas de Generación Aumentada por Recuperación (RAG). A diferencia de los enfoques de evaluación generalizados, RAGking permite consolidar métricas de veracidad factual (vía *LLM-as-a-Judge*) y telemetría computacional en un único **Response Quality Index (RQI)** estandarizado.

---

## 🚀 Características Clave

* **Evaluación Factual Estricta (1-5 Escala):** Diseñado específicamente para dominios de alta sensibilidad (como el ámbito legal), penalizando severamente las alucinaciones factuales sobre los costos computacionales.
* **Telemetría Operativa:** Mapea el consumo de tokens y la latencia de ejecución de extremo a extremo de forma determinista.
* **Conmensurabilidad Multidimensional:** Normaliza linealmente variables heterogéneas (tiempo, tokens, correctitud) a una escala común de `0 a 10` para un análisis de *trade-offs* transparente.
* **Automatizado y Programático:** Diseñado para ejecutarse sobre conjuntos de datos en formato CSV/Excel mediante pipelines de Python eficientes.

---

## 📐 Métricas de Evaluación y Normalización

El framework consolida tres dimensiones críticas en el cálculo del **RQI Final**:

1. **Factual Correctness Score (S_C):** Basado en una rúbrica de evaluación automatizada de 1 a 5 y normalizado linealmente:
   S_C = ((Score_LLM - 1) / (5 - 1)) * 10

2. **Token Efficiency Score (S_T):** Escala Min-Max invertida para premiar la optimización del tamaño del prompt y el costo de la API:
   S_T = 10 * ((T_max - T_i) / (T_max - T_min))

3. **Latency Efficiency Score (S_L):** Escala Min-Max invertida para evaluar la velocidad de respuesta del sistema en milisegundos:
   S_L = 10 * ((L_max - L_i) / (L_max - L_min))

### Response Quality Index (RQI)
El puntaje definitivo del sistema es la media aritmética de los tres scores escalados:
RQI_i = (S_C,i + S_T,i + S_L,i) / 3

---

## 🛠️ Estructura del Repositorio

```bash
├── data/
│   ├── RAG.csv                # Respuestas del sistema y logs brutos de ejecución
│   └── noRAG.csv              # Respuestas del baseline (Zero-shot)
├── src/
│   ├── llm_evaluator.py       # Pipeline de LLM-as-a-Judge (Rúbrica 1-5)
│   └── normalization.py       # Motor matemático de escalado a 10 y cálculo de RQI
├── notebooks/
│   └── analytics_suite.ipynb  # Jupyter Notebook para análisis en vivo y gráficos (.pdf)
├── requirements.txt           # Dependencias del sistema
└── README.md
