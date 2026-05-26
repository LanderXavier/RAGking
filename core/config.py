"""
core/config.py — Global state, constants, and i18n helpers.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
state = {
    "df": None,
    "csv_path": None,
    "weights": {
        "correctness": float(os.getenv("W_CORRECTNESS",  0.30)),
        # Replaced token efficiency weight with economic efficiency weight
        "economic":    float(os.getenv("W_ECONOMIC",     0.30)),
        "latency":     float(os.getenv("W_LATENCY",      0.20)),
        "maintenance": float(os.getenv("W_MAINTENANCE",  0.10)),
        "failure":     float(os.getenv("W_FAILURE",      0.10)),
    },
    "llm": {
        "response_provider":    os.getenv("RESPONSE_PROVIDER",    "deepseek"),
        "response_model":       os.getenv("RESPONSE_MODEL",       "deepseek-chat"),
        "response_cost_per_1k": float(os.getenv("RESPONSE_COST_PER_1K", 0.0014)),
        # Token pricing (per 1M tokens) — configurable via menu option
        "input_cost_per_1m":    float(os.getenv("INPUT_COST_PER_1M", 0.23)),
        "output_cost_per_1m":   float(os.getenv("OUTPUT_COST_PER_1M", 0.69)),
        "embedding_provider":   os.getenv("EMBEDDING_PROVIDER",   "openai"),
        "embedding_model":      os.getenv("EMBEDDING_MODEL",      "text-embedding-3-small"),
        "judge_provider":       os.getenv("JUDGE_PROVIDER",       "openai"),
        "judge_model":          os.getenv("JUDGE_MODEL",          "gpt-4o-mini"),
        "judge_temperature":    float(os.getenv("JUDGE_TEMPERATURE", 0.0)),
        "judge_cost_per_1k":    float(os.getenv("JUDGE_COST_PER_1K", 0.00015)),
        "judge_embedding":      os.getenv("JUDGE_EMBEDDING",      "text-embedding-3-small"),
        "local_base_url":       os.getenv("LOCAL_BASE_URL",       "http://localhost:11434/v1"),
        # Optional per-model pricing (USD per 1M tokens) for input/output.
        # Keys should match the `Model` column values when available.
        "price_table": {
            "deepseek-chat": {"input_per_1m": float(os.getenv("DS_INPUT_1M", 0.14)),
                               "output_per_1m": float(os.getenv("DS_OUTPUT_1M", 0.69))},
            "deepseek-v3":   {"input_per_1m": float(os.getenv("DS_V3_INPUT_1M", 0.14)),
                               "output_per_1m": float(os.getenv("DS_V3_OUTPUT_1M", 0.69))},
            "gpt-4-turbo":   {"input_per_1m": float(os.getenv("GPT4T_INPUT_1M", 0.30)),
                               "output_per_1m": float(os.getenv("GPT4T_OUTPUT_1M", 1.00))},
            "gpt-4.1-mini":  {"input_per_1m": float(os.getenv("G41M_INPUT_1M", 0.20)),
                               "output_per_1m": float(os.getenv("G41M_OUTPUT_1M", 0.80))},
            "gemini-2.5-flash": {"input_per_1m": float(os.getenv("GEMINI_INPUT_1M", 0.25)),
                                  "output_per_1m": float(os.getenv("GEMINI_OUTPUT_1M", 0.90))},
        },
    },
    "results_df": None,
    "operational_factors": None,
    "ui_lang": "en",
    "report_lang": "en",
    "frameworks": {},      # {name: results_df} for multi-framework comparison
    "_warned": set(),
}

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLUMNS = [
    "Question", "Correct Answer", "Generated Answer",
    "Prompt Tokens", "Completion Tokens", "Total Tokens",
    "Execution Time (ms)"
]

KNOWN_PROVIDERS = {
    "openai":   {"models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], "key_env": "OPENAI_API_KEY"},
    "deepseek": {"models": ["deepseek-chat", "deepseek-reasoner"],      "key_env": "DEEPSEEK_API_KEY"},
    "local":    {"models": ["llama3", "mistral", "gemma2", "phi3"],     "key_env": None},
}

KNOWN_EMBEDDINGS = {
    "openai": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    "local":  ["nomic-embed-text", "mxbai-embed-large"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  I18N
# ══════════════════════════════════════════════════════════════════════════════
UI_TEXT = {
    "en": {
        "main_menu":          "Main Menu",
        "load_dataset":       "Load Dataset",
        "adjust_weights":     "Adjust Weights",
        "llm_settings":       "LLM & Embeddings Settings",
        "generate_raw_metrics":"Generate Raw Metrics & Bounds",
        "generate_report":    "Generate Report from Bounds",
        "compare_frameworks": "Compare RAG Frameworks",
        "language_settings":  "Framework/Report Language",
        "exit":               "Exit",
        "select_option":      "Select an option",
        "invalid_option":     "Invalid option.",
        "status_loaded":      "CSV loaded",
        "status_not_loaded":  "No CSV loaded",
        "status_results_ready":  "Results ready",
        "status_not_evaluated":  "Not evaluated",
        "lang_section":       "7 · Framework and Report Language",
        "framework_lang":     "Framework language [es/en]",
        "report_lang":        "PDF report language [es/en]",
        "lang_saved":         "Languages saved.",
        "export_options":     "Export Options",
        "select_export":      "Select export option",
        "excel_output":       "Excel filename",
        "pdf_output":         "PDF filename",
        "ops_input_title":    "Operational Inputs for Report and Scoring",
        "ops_global":         "Operational Factors · Global",
        "ops_model":          "Operational Factors · {model}",
        "ops_intro":          "These values are used in the Maintenance dimension (M).",
        "ops_deploy":         "Deploy time in hours",
        "ops_difficulty":     "Implementation difficulty [1-5]",
        "ops_maint":          "Maintenance effort in hours/week",
        "fw_compare_title":   "6 · Compare RAG Orchestration Frameworks",
        "fw_add_prompt":      "Framework name (e.g. LangChain, LlamaIndex) — leave blank to finish",
        "fw_csv_prompt":      "Path to CSV for '{name}'",
        "fw_loaded":          "Framework '{name}' loaded ({rows} rows)",
        "fw_min_warning":     "Add at least 2 frameworks to compare.",
        "fw_generating":      "Generating comparison charts and report...",
        "fw_export_dir":      "Directory for comparison export",
    },
    "es": {
        "main_menu":          "Menú Principal",
        "load_dataset":       "Cargar Dataset",
        "adjust_weights":     "Ajustar Pesos",
        "llm_settings":       "Configuración LLM y Embeddings",
        "generate_raw_metrics":"Generar Métricas Crudas y Bounds",
        "generate_report":    "Generar Reporte desde Bounds",
        "compare_frameworks": "Comparar Frameworks RAG",
        "language_settings":  "Idioma del Framework/Reporte",
        "exit":               "Salir",
        "select_option":      "Seleccione una opción",
        "invalid_option":     "Opción inválida.",
        "status_loaded":      "CSV cargado",
        "status_not_loaded":  "Sin CSV cargado",
        "status_results_ready":  "Resultados listos",
        "status_not_evaluated":  "Sin evaluar",
        "lang_section":       "7 · Idioma del Framework y Reporte",
        "framework_lang":     "Idioma del framework [es/en]",
        "report_lang":        "Idioma del reporte PDF [es/en]",
        "lang_saved":         "Idiomas guardados.",
        "export_options":     "Opciones de Exportación",
        "select_export":      "Seleccione opción de exportación",
        "excel_output":       "Nombre de archivo Excel",
        "pdf_output":         "Nombre de archivo PDF",
        "ops_input_title":    "Parámetros Operacionales para Reporte y Scoring",
        "ops_global":         "Factores Operacionales · Global",
        "ops_model":          "Factores Operacionales · {model}",
        "ops_intro":          "Estos valores se usan en la dimensión Mantenimiento (M).",
        "ops_deploy":         "Tiempo de despliegue en horas",
        "ops_difficulty":     "Dificultad de implementación [1-5]",
        "ops_maint":          "Esfuerzo de mantenimiento en horas/semana",
        "fw_compare_title":   "6 · Comparar Frameworks de Orquestación RAG",
        "fw_add_prompt":      "Nombre del framework (ej. LangChain) — vacío para terminar",
        "fw_csv_prompt":      "Ruta al CSV para '{name}'",
        "fw_loaded":          "Framework '{name}' cargado ({rows} filas)",
        "fw_min_warning":     "Agrega al menos 2 frameworks para comparar.",
        "fw_generating":      "Generando gráficos de comparación y reporte...",
        "fw_export_dir":      "Directorio para exportar comparación",
    },
}


def tr(key, **kwargs):
    lang = state.get("ui_lang", "en")
    txt = UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)
    return txt.format(**kwargs) if kwargs else txt
