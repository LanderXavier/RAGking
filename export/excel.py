"""
export/excel.py — Excel export with multiple sheets.
"""

from datetime import datetime
import pandas as pd

from core.config import state
from ui.helpers import ok, err


def export_excel(results_df: pd.DataFrame, output_path: str, frameworks: dict = None) -> bool:
    """
    Export results to Excel.
    If frameworks dict is provided, adds one sheet per framework + comparison.
    """
    c = state["llm"]
    w = state["weights"]

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

            # ── Single dataset sheets ─────────────────────────────────────────
            if results_df is not None:
                results_df.to_excel(writer, sheet_name="Raw Results", index=False)

                if "Model" in results_df.columns:
                    summary = results_df.groupby("Model").agg(
                        Rows        =("Question",            "count"),
                        Correctness =("Correctness",         "mean"),
                        Latency_ms  =("Execution Time (ms)", "mean"),
                        Token_Eff   =("token_eff_raw",       "mean"),
                        Total_Tokens=("Total Tokens",        "sum"),
                        Failures    =("is_failure",          "sum"),
                        Sim         =("String Similarity",   "mean"),
                        RAGking     =("RAGking_score",       "mean"),
                    ).round(2).sort_values("RAGking", ascending=False)
                    summary.to_excel(writer, sheet_name="Summary by Model")

            # ── Multi-framework sheets ─────────────────────────────────────────
            if frameworks:
                comparison_rows = []
                for name, df in frameworks.items():
                    deploy_mean = round(df["deploy_hours"].mean(), 2) if "deploy_hours" in df.columns else None
                    difficulty_mean = round(df["implementation_difficulty"].mean(), 2) if "implementation_difficulty" in df.columns else None
                    maint_mean = round(df["maintenance_hours_week"].mean(), 2) if "maintenance_hours_week" in df.columns else None
                    safe_name = name[:28]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=f"FW_{safe_name}", index=False)
                    comparison_rows.append({
                        "Framework":       name,
                        "N":               len(df),
                        "RAGking Score":   round(df["RAGking_score"].mean(), 2),
                        "Correctness":     round(df["Correctness"].mean(), 2),
                        "Latency (ms)":    round(df["Execution Time (ms)"].mean(), 1),
                        "String Similarity": round(df["String Similarity"].mean(), 3),
                        "Failure Rate (%)": round(df["failure_rate"].mean() * 100, 1) if "failure_rate" in df.columns else None,
                        "Total Tokens":    int(df["Total Tokens"].sum()),
                        "Cost per Query (USD)": round(df["cost_per_query"].mean(), 6) if "cost_per_query" in df.columns else None,
                        "Economic Efficiency (Correctness/USD)": round(df["economic_eff_raw"].mean(), 6) if "economic_eff_raw" in df.columns else None,
                        "Token Efficiency (tok/pt)": round(df["token_eff_raw"].mean(), 4) if "token_eff_raw" in df.columns else None,
                        "Deploy Time (hours)": deploy_mean,
                        "Difficulty (1-5)": difficulty_mean,
                        "Maintenance (h/wk)": maint_mean,
                        "Response Model":  (df["Model"].unique()[0] if "Model" in df.columns and len(df["Model"].unique())>0 else c.get("response_model")),
                        "Input cost per 1M":  c.get("input_cost_per_1m", None),
                        "Output cost per 1M": c.get("output_cost_per_1m", None),
                    })

                cmp_df = pd.DataFrame(comparison_rows).set_index("Framework")
                cmp_df.to_excel(writer, sheet_name="Framework Comparison")

            # ── Config sheet ──────────────────────────────────────────────────
            config_df = pd.DataFrame([{
                "wC Correctness":       w["correctness"],
                "wE Economic Eff.":     w.get("economic", w.get("token_eff", None)),
                "wL Latency":           w["latency"],
                "wM Maintenance":       w["maintenance"],
                "wF Failure Rate":      w["failure"],
                "Response Model":       c["response_model"],
                "Judge Model":          c["judge_model"],
                "Embedding Model":      c["embedding_model"],
                "Response cost per 1K": c.get("response_cost_per_1k", None),
                "Input cost per 1M":    c.get("input_cost_per_1m", None),
                "Output cost per 1M":   c.get("output_cost_per_1m", None),
                "Per-model price table": c.get("price_table", {}),
                "Generated on":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            }])
            config_df.to_excel(writer, sheet_name="Config & Weights", index=False)

        ok(f"Excel exported  →  {output_path}")
        return True

    except Exception as e:
        err(f"Excel export failed: {e}")
        return False


def export_csv_results(results_df: pd.DataFrame, output_path: str) -> bool:
    """Export the scored results DataFrame to a plain CSV."""
    try:
        results_df.to_csv(output_path, index=False)
        ok(f"CSV exported  →  {output_path}")
        return True
    except Exception as e:
        err(f"CSV export failed: {e}")
        return False


def export_framework_comparison_csv(frameworks: dict, output_path: str) -> bool:
    """Export a single CSV comparing all frameworks by key metrics."""
    rows = []
    for name, df in frameworks.items():
        rows.append({
            "Framework":          name,
            "N":                  len(df),
            "RAGking_score_mean": round(df["RAGking_score"].mean(), 2),
            "RAGking_score_std":  round(df["RAGking_score"].std(), 2),
            "Correctness_mean":   round(df["Correctness"].mean(), 2),
            "Correctness_std":    round(df["Correctness"].std(), 2),
            "Latency_ms_mean":    round(df["Execution Time (ms)"].mean(), 1),
            "Latency_ms_std":     round(df["Execution Time (ms)"].std(), 1),
            "String_Sim_mean":    round(df["String Similarity"].mean(), 3),
            "Failure_rate_pct":   round(df["failure_rate"].mean() * 100, 1) if "failure_rate" in df.columns else None,
            "Total_Tokens_sum":   int(df["Total Tokens"].sum()),
            "Prompt_Tokens_mean": round(df["Prompt Tokens"].mean(), 1),
            "Token_Efficiency_mean": round(df["token_eff_raw"].mean(), 4) if "token_eff_raw" in df.columns else None,
            "Deploy_Time_hours_mean": round(df["deploy_hours"].mean(), 2) if "deploy_hours" in df.columns else None,
            "Difficulty_mean": round(df["implementation_difficulty"].mean(), 2) if "implementation_difficulty" in df.columns else None,
            "Maintenance_hours_week_mean": round(df["maintenance_hours_week"].mean(), 2) if "maintenance_hours_week" in df.columns else None,
            "Response_Model": (df["Model"].unique()[0] if "Model" in df.columns and len(df["Model"].unique())>0 else state["llm"].get("response_model")),
            "Input_cost_per_1M": state["llm"].get("input_cost_per_1m", None),
            "Output_cost_per_1M": state["llm"].get("output_cost_per_1m", None),
        })

    cmp_df = pd.DataFrame(rows)
    try:
        cmp_df.to_csv(output_path, index=False)
        ok(f"Comparison CSV  →  {output_path}")
        return True
    except Exception as e:
        err(f"CSV export failed: {e}")
        return False
