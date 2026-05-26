"""
ui/menu.py — Main CLI menu and all sub-menus (load, weights, LLM, metrics, compare, language).
"""

import os
import sys
import math
import time

import numpy as np
import pandas as pd

from core.config import state, tr, REQUIRED_COLUMNS, KNOWN_PROVIDERS, KNOWN_EMBEDDINGS
from ui.helpers import (
    banner, section, ok, err, warn, info, ask, ask_float, ask_float_range,
    ask_int_range, pause, status_badge, check_api_key, C,
)


# ══════════════════════════════════════════════════════════════════════════════
#  1 — LOAD CSV
# ══════════════════════════════════════════════════════════════════════════════
def _try_load_csv(path: str) -> pd.DataFrame | None:
    """Attempt multiple separators and return a valid DataFrame or None."""
    parse_attempts = [
        {"sep": "\t"},
        {"sep": ";"},
        {"sep": ","},
        {"sep": None, "engine": "python"},
        {"sep": r"\s+", "engine": "python"},
    ]
    for opts in parse_attempts:
        try:
            df = pd.read_csv(path, **opts)
            if len(df.columns) >= 5:
                return df
        except Exception:
            continue
    return None


def menu_load_csv():
    banner()
    section("1 · Load Dataset  (CSV / TSV)")
    print(f"""
  {C.WH}Required columns (case-sensitive):{C.R}
  {C.GY}  Question · Correct Answer · Generated Answer{C.R}
  {C.GY}  Prompt Tokens · Completion Tokens · Total Tokens · Execution Time (ms){C.R}

  {C.YL}⚠  Correctness and String Similarity will be computed automatically.{C.R}
""")

    print(f"\n  Note: you may type a filename only (e.g. 'mydata') and '.csv' will be assumed.\n")
    path_input = ask("Path to CSV file or filename ('.csv' assumed if no extension)", state["csv_path"] or "results.csv")
    # If user provided no extension, assume .csv
    if path_input and not os.path.splitext(path_input)[1]:
        path = path_input + ".csv"
    else:
        path = path_input
    if not os.path.exists(path):
        err(f"File not found: {path}"); pause(); return

    df = _try_load_csv(path)
    if df is None:
        err("Could not parse the file. Check the separator/format."); pause(); return

    section("Column Validation")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        err("Missing required columns:  " + ", ".join(missing))
        warn("Column names are case-sensitive."); pause(); return

    ok(f"All {len(REQUIRED_COLUMNS)} required columns found.")

    numeric_cols = ["Prompt Tokens", "Completion Tokens", "Total Tokens", "Execution Time (ms)"]
    type_errors = []
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            type_errors.append(col)
    if type_errors:
        err(f"Non-numeric values in: {type_errors}"); pause(); return

    section("Preview (first 3 rows)")
    for i, row in df.head(3).iterrows():
        q = str(row["Question"])[:60] + ("…" if len(str(row["Question"])) > 60 else "")
        print(f"\n  {C.GY}Row {i+1}{C.R}  {C.WH}{q}{C.R}")
        print(f"         Tokens: {C.YL}{row['Total Tokens']}{C.R}  Latency: {C.YL}{row['Execution Time (ms)']}ms{C.R}")

    print()
    ok(f"Dataset ready  →  {C.YL}{len(df)} rows{C.R}, {C.YL}{len(df.columns)} columns{C.R}")
    state["df"] = df
    state["csv_path"] = path
    state["results_df"] = None
    pause()


# ══════════════════════════════════════════════════════════════════════════════
#  2 — ADJUST WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def menu_weights():
    banner()
    section("2 · Adjust RAGking Weights")
    w = state["weights"]
    total = sum(w.values())
    t_color = C.GR if math.isclose(total, 1.0, abs_tol=0.01) else C.RD

    print(f"""
  {C.WH}ℛ = ( wC·Ĉ + wT·T̂ + wL·L̂ + wM·M̂ + wF·F̂ ) × 100{C.R}

  Current weights (must sum to 1.0):
  {C.CY}  wC  Correctness       {C.YL}{w['correctness']:.2f}
    {C.CY}  wE  Economic Eff.     {C.YL}{w.get('economic', w.get('token_eff', 0.0)):.2f}
  {C.CY}  wL  Latency           {C.YL}{w['latency']:.2f}
  {C.CY}  wM  Maintenance       {C.YL}{w['maintenance']:.2f}
  {C.CY}  wF  Failure Rate      {C.YL}{w['failure']:.2f}{C.R}
  {'─'*40}
  Σ = {t_color}{C.B}{total:.2f}{C.R}

  {C.GY}Press Enter to keep current values.{C.R}
""")

    nw = {
        "correctness": ask_float("wC  Correctness       ", w["correctness"]),
        "economic":    ask_float("wE  Economic Eff.     ", w.get("economic", w.get("token_eff", 0.20))),
        "latency":     ask_float("wL  Latency           ", w["latency"]),
        "maintenance": ask_float("wM  Maintenance       ", w["maintenance"]),
        "failure":     ask_float("wF  Failure Rate      ", w["failure"]),
    }
    total_new = sum(nw.values())
    if not math.isclose(total_new, 1.0, abs_tol=0.01):
        err(f"Weights sum to {total_new:.3f} — must equal 1.0. NOT saved."); pause(); return

    state["weights"] = nw
    ok(f"Weights saved.  Σ = {total_new:.2f}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
#  3 — LLM & EMBEDDING SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def _pick_provider(label: str, current: str) -> str:
    providers = list(KNOWN_PROVIDERS.keys())
    print(f"\n  {C.WH}{label}:{C.R}")
    for i, p in enumerate(providers, 1):
        marker = f"{C.GR}●{C.R}" if p == current else f"{C.GY}○{C.R}"
        kstatus = (f"{C.GR}key ✔{C.R}" if check_api_key(p)
                   else f"{C.RD}key missing{C.R}")
        print(f"    {marker} {C.YL}[{i}]{C.R}  {p:<12}  {C.GY}({kstatus}){C.R}")
    choice = ask(f"Choose [1–{len(providers)}]", "1")
    try:
        selected = providers[int(choice) - 1]
    except (ValueError, IndexError):
        selected = current
    if not check_api_key(selected):
        key_env = KNOWN_PROVIDERS[selected].get("key_env", "N/A")
        warn(f"Set {key_env} in your .env file before running evaluations.")
    return selected


def _pick_model(provider: str, current: str, kind: str = "model") -> str:
    models = (KNOWN_EMBEDDINGS.get(provider, KNOWN_EMBEDDINGS["openai"])
              if kind == "embedding"
              else KNOWN_PROVIDERS[provider]["models"])
    print(f"\n  {C.WH}Available {kind}s for {provider}:{C.R}")
    for i, m in enumerate(models, 1):
        marker = f"{C.GR}●{C.R}" if m == current else f"{C.GY}○{C.R}"
        print(f"    {marker} {C.YL}[{i}]{C.R}  {m}")
    choice = ask(f"Choose [1–{len(models)}] or type a custom name", "1")
    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        return choice if choice else current


def menu_llm_settings():
    banner()
    section("3 · LLM & Embedding Settings")
    c = state["llm"]

    section("A · Response LLM")
    c["response_provider"]    = _pick_provider("Provider", c["response_provider"])
    c["response_model"]       = _pick_model(c["response_provider"], c["response_model"])
    c["response_cost_per_1k"] = ask_float(
        "Cost per 1K tokens USD (0 if local/free)", c["response_cost_per_1k"])
    # Input/Output token pricing (per 1M tokens)
    c["input_cost_per_1m"]  = ask_float(
        "Input cost per 1M tokens USD (e.g. 0.23)", c.get("input_cost_per_1m", 0.23))
    c["output_cost_per_1m"] = ask_float(
        "Output cost per 1M tokens USD (e.g. 0.69)", c.get("output_cost_per_1m", 0.69))

    section("B · Retrieval Embedding Model")
    c["embedding_provider"] = _pick_provider("Provider", c["embedding_provider"])
    c["embedding_model"]    = _pick_model(c["embedding_provider"], c["embedding_model"], kind="embedding")

    section("C · Judge LLM")
    c["judge_provider"]    = _pick_provider("Provider", c["judge_provider"])
    c["judge_model"]       = _pick_model(c["judge_provider"], c["judge_model"])
    c["judge_temperature"] = ask_float_range(
        "Judge temperature (0 = deterministic)", c.get("judge_temperature", 0.0),
        min_val=0.0, max_val=2.0)

    section("D · Judge Embedding Model")
    c["judge_embedding"] = _pick_model(c["judge_provider"], c["judge_embedding"], kind="embedding")

    state["llm"] = c
    ok("LLM settings saved.")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
#  ASCII HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _ascii_hist(series, label, color, bins=10, bar_width=32):
    vals = series.dropna()
    if len(vals) == 0:
        return
    counts, edges = np.histogram(vals, bins=bins)
    mx = max(counts) if max(counts) > 0 else 1
    print(f"\n  {C.B}{color}{label}{C.R}")
    print(f"  {C.GY}{'─'*55}{C.R}")
    for count, edge in zip(counts, edges):
        bar = "█" * int(bar_width * count / mx)
        print(f"  {C.GY}{edge:8.1f}{C.R}  {color}{bar:<{bar_width}}{C.R}  {C.WH}{count}{C.R}")
    print(f"  {C.GY}  n={len(vals)}  min={vals.min():.2f}  max={vals.max():.2f}  mean={vals.mean():.2f}{C.R}")


def _print_summary(df: pd.DataFrame):
    group_col = "Model" if "Model" in df.columns else None
    if group_col:
        summary = df.groupby(group_col).agg(
            Rows        =("Question",            "count"),
            Correctness =("Correctness",         "mean"),
            Latency_ms  =("Execution Time (ms)", "mean"),
            TokEff      =("token_eff_raw",       "mean"),
            Tokens      =("Total Tokens",        "sum"),
            Failures    =("is_failure",          "sum"),
            Sim         =("String Similarity",   "mean"),
            RAGking     =("RAGking_score",       "mean"),
        ).round(2).sort_values("RAGking", ascending=False)
        rows = [(str(n), row) for n, row in summary.iterrows()]
    else:
        rows = [("All data", {
            "Rows":       len(df),
            "Correctness":df["Correctness"].mean(),
            "Latency_ms": df["Execution Time (ms)"].mean(),
            "TokEff":    df["token_eff_raw"].mean(),
            "Tokens":     df["Total Tokens"].sum(),
            "Failures":   int(df["is_failure"].sum()),
            "Sim":        df["String Similarity"].mean(),
            "RAGking":    df["RAGking_score"].mean(),
        })]

    print(f"\n  {C.B}{C.WH}{'Config':<22} {'Rows':>5} {'Corr':>6} {'Lat(ms)':>9} {'TokEff':>8} {'Tokens':>8} {'Fail':>5} {'Sim':>6} {'ℛ':>9}{C.R}")
    print(f"  {C.GY}{'─'*84}{C.R}")
    for name, row in rows:
        sc = float(row.get("RAGking", row.get("RAGking_score", 0)))
        sc_c = C.GR if sc >= 70 else (C.YL if sc >= 40 else C.RD)
        print(
            f"  {C.CY}{name:<22}{C.R}"
            f"  {int(row['Rows']):>5}"
            f"  {float(row['Correctness']):>6.2f}"
            f"  {float(row['Latency_ms']):>9.1f}"
            f"  {float(row['TokEff']):>8.2f}"
            f"  {int(row['Tokens']):>8}"
            f"  {int(row['Failures']):>5}"
            f"  {float(row['Sim']):>6.3f}"
            f"  {sc_c}{C.B}{sc:>9.2f}{C.R}"
        )


def _print_raw_summary(df: pd.DataFrame):
    group_col = "Model" if "Model" in df.columns else None
    if group_col:
        summary = df.groupby(group_col).agg(
            Rows        =("Question",            "count"),
            Correctness =("Correctness",         "mean"),
            Latency_ms  =("Execution Time (ms)", "mean"),
            TokEff      =("token_eff_raw",       "mean"),
            Cost        =("cost_per_query",      "mean"),
            EconEff     =("economic_eff_raw",    "mean"),
            Failures    =("is_failure",          "sum"),
            Sim         =("String Similarity",   "mean"),
        ).round(4)
        rows = [(str(n), row) for n, row in summary.iterrows()]
    else:
        rows = [("All data", {
            "Rows":       len(df),
            "Correctness":df["Correctness"].mean(),
            "Latency_ms": df["Execution Time (ms)"].mean(),
            "TokEff":    df["token_eff_raw"].mean(),
            "Cost":      df["cost_per_query"].mean(),
            "EconEff":   df["economic_eff_raw"].mean(),
            "Failures":  int(df["is_failure"].sum()),
            "Sim":       df["String Similarity"].mean(),
        })]

    print(f"\n  {C.B}{C.WH}{'Config':<22} {'Rows':>5} {'Corr':>6} {'Lat(ms)':>9} {'TokEff':>8} {'Cost':>12} {'EconEff':>10} {'Fail':>5} {'Sim':>6}{C.R}")
    print(f"  {C.GY}{'─'*96}{C.R}")
    for name, row in rows:
        print(
            f"  {C.CY}{name:<22}{C.R}"
            f"  {int(row['Rows']):>5}"
            f"  {float(row['Correctness']):>6.2f}"
            f"  {float(row['Latency_ms']):>9.1f}"
            f"  {float(row['TokEff']):>8.2f}"
            f"  {float(row['Cost']):>12.6f}"
            f"  {float(row['EconEff']):>10.4f}"
            f"  {int(row['Failures']):>5}"
            f"  {float(row['Sim']):>6.3f}"
        )


def _pick_metric_bounds(raw_df: pd.DataFrame) -> tuple[dict, str]:
    from core.metrics import build_global_bounds, extract_metric_bounds, get_bounds_profile, list_bounds_profiles

    section("Normalization Bounds")
    print(f"""
  {C.WH}Choose which bounds the individual report should use.{C.R}

  {C.YL}[1]{C.R}  Local bounds from the current dataset (existing behavior)
  {C.YL}[2]{C.R}  Global bounds merged from all saved profiles in bounds.json
  {C.YL}[3]{C.R}  One saved bounds profile from bounds.json
""")

    choice = ask("Select bounds source", "1")

    if choice == "1":
        return extract_metric_bounds(raw_df), "local bounds from current dataset"

    if choice == "2":
        global_bounds = build_global_bounds()
        if global_bounds:
            return global_bounds, "global bounds from bounds.json"
        warn("No saved bounds profiles found in bounds.json. Falling back to local bounds.")
        return extract_metric_bounds(raw_df), "local bounds from current dataset"

    if choice == "3":
        profiles = list_bounds_profiles()
        if not profiles:
            warn("No saved bounds profiles found in bounds.json. Falling back to local bounds.")
            return extract_metric_bounds(raw_df), "local bounds from current dataset"

        print(f"\n  {C.WH}Available bounds profiles:{C.R}")
        for i, profile_name in enumerate(profiles, 1):
            print(f"    {C.YL}[{i}]{C.R}  {profile_name}")
        selected_raw = ask(f"Choose [1–{len(profiles)}]", "1")
        try:
            selected_profile = profiles[int(selected_raw) - 1]
        except (ValueError, IndexError):
            selected_profile = profiles[0]

        profile_bounds = get_bounds_profile(selected_profile)
        if profile_bounds:
            return profile_bounds, f"bounds profile: {selected_profile}"

        warn(f"Bounds profile '{selected_profile}' could not be loaded. Falling back to local bounds.")
        return extract_metric_bounds(raw_df), "local bounds from current dataset"

    return extract_metric_bounds(raw_df), "local bounds from current dataset"


# ══════════════════════════════════════════════════════════════════════════════
#  4 — GENERATE RAW METRICS & BOUNDS
# ══════════════════════════════════════════════════════════════════════════════
def menu_raw_metrics():
    banner()
    section(tr("generate_raw_metrics"))

    if state["df"] is None:
        err("No dataset loaded. Go to option 1 first."); pause(); return

    from core.llm import compute_correctness_and_similarity
    from core.metrics import collect_operational_factors, compute_raw_metrics, extract_metric_bounds, save_bounds_profile, make_bounds_snapshot_name
    from export.excel import export_excel, export_csv_results

    df = state["df"].copy()

    if "Correctness" not in df.columns or "String Similarity" not in df.columns:
        info("Computing Correctness (LLM judge) and String Similarity (embeddings)...")
        df = compute_correctness_and_similarity(df, state["llm"], progress=True)
    else:
        info("Using existing Correctness / String Similarity columns.")

    operational_factors = collect_operational_factors(df)
    state["operational_factors"] = operational_factors

    default_fail = int((
        df["Correctness"].isna()
        | (pd.to_numeric(df["Correctness"], errors="coerce").fillna(0) == 0)
        | df["Generated Answer"].isna()
        | (df["Generated Answer"].astype(str).str.strip() == "")
    ).sum())
    failure_count = ask_int_range(
        f"Failure count out of {len(df)} questions",
        default_fail, min_val=0, max_val=len(df),
    )

    raw_results = compute_raw_metrics(
        df,
        operational_factors=operational_factors,
        failure_count_override=failure_count,
    )
    state["raw_metrics_df"] = raw_results
    state["results_df"] = raw_results

    # Derive base name from input CSV to use as default output prefix
    from pathlib import Path
    csv_path = state.get("csv_path") or "results.csv"
    base_name = Path(csv_path).stem
    bounds_profile_name = make_bounds_snapshot_name(base_name)
    bounds_snapshot = extract_metric_bounds(raw_results)
    save_bounds_profile(bounds_profile_name, bounds_snapshot, source_name=csv_path, row_count=len(raw_results))

    section("Raw Metrics Summary")
    _print_raw_summary(raw_results)

    info(f"Bounds snapshot saved as '{bounds_profile_name}' in bounds.json.")
    ok("Raw metrics computed. No RAGking score was calculated in this step.")

    section(tr("export_options"))
    print(f"""
  {C.YL}[1]{C.R}  Export raw CSV only
  {C.YL}[2]{C.R}  Export raw Excel only
  {C.YL}[3]{C.R}  Export raw CSV + Excel
  {C.YL}[0]{C.R}  Skip export
""")
    choice = ask("Select export option", "3")

    if choice in ["1", "3"]:
        out = ask("CSV filename", f"{base_name}_raw_metrics.csv")
        if out and not os.path.splitext(out)[1]:
            out = out + ".csv"
        export_csv_results(raw_results, out)
        state["raw_metrics_csv_path"] = out

    if choice in ["2", "3"]:
        out = ask("Excel filename", f"{base_name}_raw_metrics.xlsx")
        if out and not os.path.splitext(out)[1]:
            out = out + ".xlsx"
        export_excel(raw_results, out)

    pause()


# ══════════════════════════════════════════════════════════════════════════════
#  5 — GENERATE REPORTS FROM BOUNDS
# ══════════════════════════════════════════════════════════════════════════════
def menu_reports():
    banner()
    section(tr("generate_report"))

    if state["df"] is None and state.get("raw_metrics_df") is None:
        err("No dataset loaded. Go to option 1 first."); pause(); return

    from pathlib import Path
    from core.metrics import collect_operational_factors, compute_raw_metrics, compute_metrics
    from charts.plots import (
        plot_distribution, plot_dashboard, plot_model_comparison,
    )
    from export.excel import export_excel, export_csv_results
    from export.pdf_report import generate_pdf_report

    load_choice = ask("Load raw metrics CSV from option 4? [y/n]", "y").strip().lower()
    raw_results = None

    if load_choice in {"y", "yes", "s", "si", "sí"}:
        default_raw = state.get("raw_metrics_csv_path") or f"{Path(state.get('csv_path') or 'results.csv').stem}_raw_metrics.csv"
        raw_path = ask("Path to raw metrics CSV", default_raw)
        if raw_path and not os.path.splitext(raw_path)[1]:
            raw_path = raw_path + ".csv"
        if not os.path.exists(raw_path):
            err(f"File not found: {raw_path}"); pause(); return
        raw_results = pd.read_csv(raw_path)
        state["raw_metrics_df"] = raw_results
        state["raw_metrics_csv_path"] = raw_path
        info(f"Loaded raw metrics CSV: {raw_path}")
    elif state.get("raw_metrics_df") is not None:
        raw_results = state["raw_metrics_df"].copy()
        info("Using raw metrics already present in memory.")
    else:
        df = state["df"].copy()
        from core.llm import compute_correctness_and_similarity

        if "Correctness" not in df.columns or "String Similarity" not in df.columns:
            info("Computing Correctness (LLM judge) and String Similarity (embeddings)...")
            df = compute_correctness_and_similarity(df, state["llm"], progress=True)
        else:
            info("Using existing Correctness / String Similarity columns.")

        operational_factors = collect_operational_factors(df)
        state["operational_factors"] = operational_factors

        default_fail = int((
            df["Correctness"].isna()
            | (pd.to_numeric(df["Correctness"], errors="coerce").fillna(0) == 0)
            | df["Generated Answer"].isna()
            | (df["Generated Answer"].astype(str).str.strip() == "")
        ).sum())
        failure_count = ask_int_range(
            f"Failure count out of {len(df)} questions",
            default_fail, min_val=0, max_val=len(df),
        )

        raw_results = compute_raw_metrics(
            df,
            operational_factors=operational_factors,
            failure_count_override=failure_count,
        )
        state["raw_metrics_df"] = raw_results
        state["raw_metrics_csv_path"] = state.get("csv_path") or "results.csv"

    if "Correctness" not in raw_results.columns or "String Similarity" not in raw_results.columns:
        err("Raw metrics CSV must include Correctness and String Similarity columns."); pause(); return

    if "Generated Answer" not in raw_results.columns:
        raw_results["Generated Answer"] = ""

    if state.get("operational_factors") is None:
        state["operational_factors"] = collect_operational_factors(raw_results)

    operational_factors = state["operational_factors"]

    if "failure_rate" in raw_results.columns:
        failure_count = None
    else:
        default_fail = int((
            raw_results["Correctness"].isna()
            | (pd.to_numeric(raw_results["Correctness"], errors="coerce").fillna(0) == 0)
            | raw_results.get("Generated Answer", pd.Series([""] * len(raw_results))).isna()
        ).sum())
        failure_count = ask_int_range(
            f"Failure count out of {len(raw_results)} questions",
            default_fail, min_val=0, max_val=len(raw_results),
        )

    metric_bounds, bounds_source = _pick_metric_bounds(raw_results)
    state["metric_bounds_source"] = bounds_source

    results = compute_metrics(
        raw_results,
        operational_factors=operational_factors,
        failure_count_override=failure_count,
        metric_bounds=metric_bounds,
    )
    state["results_df"] = results

    csv_path = state.get("raw_metrics_csv_path") or state.get("csv_path") or "results.csv"
    base_name = Path(csv_path).stem

    section("Summary Table")
    _print_summary(results)

    section("ASCII Charts")
    _ascii_hist(results["Correctness"],         "Correctness  [1–5]",        C.GR)
    _ascii_hist(results["Execution Time (ms)"], "Latency  [ms]",             C.YL)
    _ascii_hist(results["String Similarity"],   "String Similarity  [0–1]",  C.MG)
    _ascii_hist(results["RAGking_score"],       "ℛ RAGking Score  [0–100]",  C.CY)

    section("Estimated API Cost")
    c = state["llm"]
    total_prompt = results["Prompt Tokens"].sum()
    total_comp = results["Completion Tokens"].sum()
    input_cost = c.get("input_cost_per_1m", 0.0)
    output_cost = c.get("output_cost_per_1m", 0.0)
    est = (total_prompt / 1_000_000.0) * input_cost + (total_comp / 1_000_000.0) * output_cost
    print(f"""
    Total prompt tokens   {C.YL}{total_prompt:>12,}{C.R}
    Total completion toks {C.YL}{total_comp:>12,}{C.R}
    Estimated cost         {C.YL}${est:>11.4f} USD{C.R}  {C.GY}@ Input ${input_cost}/1M | Output ${output_cost}/1M{C.R}
    Avg cost per query     {C.YL}${results['cost_per_query'].mean():.6f}{C.R}
    Avg econ efficiency    {C.YL}{results['economic_eff_raw'].mean():.6f} Correctness/USD{C.R}
    Avg token efficiency   {C.YL}{results['token_eff_raw'].mean():.2f} tok/pt{C.R}
""")

    section(tr("export_options"))
    print(f"""
  {C.YL}[1]{C.R}  Export to Excel  (.xlsx)
  {C.YL}[2]{C.R}  Export to CSV
  {C.YL}[3]{C.R}  Generate PDF Report
  {C.YL}[4]{C.R}  Export Individual Charts (large PNG)
  {C.YL}[5]{C.R}  All of the above
  {C.YL}[0]{C.R}  Skip export
""")
    choice = ask(tr("select_export"), "5")

    if choice in ["1"]:
        out = ask(tr("excel_output"), f"{base_name}.xlsx")
        if out and not os.path.splitext(out)[1]:
            out = out + ".xlsx"
        export_excel(results, out)

    if choice in ["2"]:
        out = ask("CSV filename", f"{base_name}_results.csv")
        if out and not os.path.splitext(out)[1]:
            out = out + ".csv"
        export_csv_results(results, out)

    if choice in ["3"]:
        pdf_path = ask(tr("pdf_output"), f"{base_name}_report.pdf")
        if pdf_path and not os.path.splitext(pdf_path)[1]:
            pdf_path = pdf_path + ".pdf"
        generate_pdf_report(results, pdf_path,
                            operational_factors=operational_factors,
                            report_lang=state.get("report_lang", "en"),
                            failure_count_override=failure_count)

    if choice in ["4"]:
        _export_individual_charts(results, base_name)

    if choice == "5":
        export_root = os.path.join(".", base_name)
        os.makedirs(export_root, exist_ok=True)

        excel_out = os.path.join(export_root, f"{base_name}.xlsx")
        csv_out = os.path.join(export_root, f"{base_name}_results.csv")
        pdf_out = os.path.join(export_root, f"{base_name}_report.pdf")
        charts_dir = os.path.join(export_root, "charts")

        export_excel(results, excel_out)
        export_csv_results(results, csv_out)
        generate_pdf_report(results, pdf_out,
                            operational_factors=operational_factors,
                            report_lang=state.get("report_lang", "en"),
                            failure_count_override=failure_count)
        _export_individual_charts(results, base_name, charts_dir=charts_dir)

        ok(f"All exports saved to folder: {C.CY}{export_root}{C.R}")
        pause()
        return

    pause()


def _export_individual_charts(results: pd.DataFrame, base_name: str = None, charts_dir: str = None):
    from charts.plots import plot_distribution, plot_dashboard, plot_model_comparison

    default_dir = charts_dir or (f"./charts_{base_name}" if base_name else "./charts_export")
    charts_dir = ask("Directory for charts", default_dir) if charts_dir is None else charts_dir
    os.makedirs(charts_dir, exist_ok=True)

    prefix = base_name + "_" if base_name else ""
    tasks = [
        ("Correctness",        "Correctness Score [1–5]",  "#2ecc71",  f"{prefix}01_Correctness_Distribution.png"),
        ("Execution Time (ms)","Latency (ms)",             "#f39c12",  f"{prefix}02_Latency_Distribution.png"),
        ("RAGking_score",      "RAGking Score [0–100]",    "#3498db",  f"{prefix}03_RAGking_Distribution.png"),
    ]
    for col, xlabel, color, fname in tasks:
        try:
            path = os.path.join(charts_dir, fname)
            plot_distribution(results[col], path, xlabel=xlabel, color=color, scale=1.6)
            ok(f"{fname}  →  {C.CY}{path}{C.R}")
        except Exception as e:
            err(f"{fname}: {e}")

    try:
        dash_fname = f"{prefix}04_Dashboard.png"
        dash_path = os.path.join(charts_dir, dash_fname)
        plot_dashboard(results, dash_path, scale=1.6)
        ok(f"{dash_fname}  →  {C.CY}{dash_path}{C.R}")
    except Exception as e:
        err(f"Dashboard: {e}")

    if "Model" in results.columns and len(results["Model"].unique()) > 1:
        try:
            cmp_fname = f"{prefix}05_Model_Comparison.png"
            cmp_path = os.path.join(charts_dir, cmp_fname)
            plot_model_comparison(results, cmp_path, scale=1.6)
            ok(f"{cmp_fname}  →  {C.CY}{cmp_path}{C.R}")
        except Exception as e:
            err(f"Model comparison: {e}")

    ok(f"All charts exported to: {C.CY}{charts_dir}{C.R}")


# ══════════════════════════════════════════════════════════════════════════════
#  6 — COMPARE RAG FRAMEWORKS
# ══════════════════════════════════════════════════════════════════════════════
def menu_compare_frameworks():
    banner()
    section(tr("fw_compare_title"))

    print(f"""
  {C.WH}Load 2 or more RAG orchestration framework CSVs to compare them.{C.R}
    {C.GY}Each CSV must be a saved report CSV that already contains RAGking_score.{C.R}
    {C.GY}The comparison will reuse the CSV values directly and will not recompute metrics.{C.R}
  {C.GY}Leave framework name blank to finish adding frameworks.{C.R}
""")

    from charts.plots import (
        plot_framework_comparison_bars, plot_framework_ragking_bars,
        plot_framework_spider, plot_framework_box,
    )
    from export.excel import export_excel, export_framework_comparison_csv
    from export.pdf_report import generate_framework_pdf

    frameworks: dict = {}  # {name: scored_df}

    # ── Collect frameworks ────────────────────────────────────────────────────
    while True:
        name = ask(tr("fw_add_prompt"), "").strip()
        if not name:
            if len(frameworks) >= 2:
                break
            elif len(frameworks) == 0:
                warn("Enter at least one framework name."); continue
            else:
                warn(tr("fw_min_warning")); continue

        csv_path = ask(tr("fw_csv_prompt", name=name), f"{name.lower().replace(' ', '_')}.csv")
        if csv_path and not os.path.splitext(csv_path)[1]:
            csv_path = csv_path + ".csv"
        if not os.path.exists(csv_path):
            err(f"File not found: {csv_path}"); continue

        df = _try_load_csv(csv_path)
        if df is None:
            err("Could not parse file."); continue

        required_compare_cols = [
            "Correctness", "String Similarity", "RAGking_score",
            "cost_per_query", "token_eff_raw", "economic_eff_raw",
            "C_norm", "E_norm", "L_norm", "M_norm", "F_norm",
        ]
        missing = [col for col in required_compare_cols if col not in df.columns]
        if missing:
            err(
                f"'{name}' is missing comparison columns: {missing}. Use the exported report CSV from the individual report first."
            )
            continue

        frameworks[name] = df
        state["frameworks"][name] = df
        ok(tr("fw_loaded", name=name, rows=len(df)))

    if len(frameworks) < 2:
        warn(tr("fw_min_warning")); pause(); return

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Framework Comparison Summary")
    print(f"\n  {C.B}{C.WH}{'Framework':<22} {'N':>5} {'ℛ Avg':>8} {'Correctness':>12} {'Latency ms':>12} {'TokEff':>8} {'Sim':>6}{C.R}")
    print(f"  {C.GY}{'─'*81}{C.R}")
    for name, df in frameworks.items():
        sc = df["RAGking_score"].mean()
        sc_c = C.GR if sc >= 70 else (C.YL if sc >= 40 else C.RD)
        print(
            f"  {C.CY}{name:<22}{C.R}"
            f"  {len(df):>5}"
            f"  {sc_c}{C.B}{sc:>8.2f}{C.R}"
            f"  {df['Correctness'].mean():>12.2f}"
            f"  {df['Execution Time (ms)'].mean():>12.1f}"
            f"  {df['token_eff_raw'].mean():>8.2f}"
            f"  {df['String Similarity'].mean():>6.3f}"
        )

    # ── Export menu ───────────────────────────────────────────────────────────
    section("Export Comparison")
    print(f"""
  {C.YL}[1]{C.R}  Export Comparison CSV
  {C.YL}[2]{C.R}  Export per-framework Excel (all frameworks in one workbook)
  {C.YL}[3]{C.R}  Export individual charts (PNG)
  {C.YL}[4]{C.R}  Generate full PDF comparison report
  {C.YL}[5]{C.R}  All of the above
  {C.YL}[0]{C.R}  Skip
""")
    choice = ask("Select export option", "5")

    default_dir = "./framework_comparison"
    out_dir = ask("Output directory", default_dir)
    os.makedirs(out_dir, exist_ok=True)

    if choice in ["1", "5"]:
        csv_out = os.path.join(out_dir, "framework_comparison.csv")
        export_framework_comparison_csv(frameworks, csv_out)

    if choice in ["2", "5"]:
        xl_out = os.path.join(out_dir, "framework_comparison.xlsx")
        export_excel(None, xl_out, frameworks=frameworks)

    if choice in ["3", "5"]:
        info("Exporting comparison charts...")
        report_lang = state.get("report_lang", "en")
        _export_framework_charts(frameworks, out_dir)

    if choice in ["4", "5"]:
        pdf_out = os.path.join(out_dir, "framework_comparison_report.pdf")
        generate_framework_pdf(frameworks, pdf_out,
                               report_lang=state.get("report_lang", "en"))

    ok(f"All comparison exports saved to:  {C.CY}{out_dir}{C.R}")
    pause()


def _export_framework_charts(frameworks: dict, out_dir: str):
    from charts.plots import (
        plot_framework_ragking_bars, plot_framework_comparison_bars,
        plot_framework_spider, plot_framework_individual_spiders, plot_framework_box,
    )

    tasks = [
        ("01_RAGking_Scores.png",      plot_framework_ragking_bars,     {}),
        ("02_Grouped_Metrics.png",     plot_framework_comparison_bars,  {}),
        ("03_Spider_Chart.png",        plot_framework_spider,           {}),
    ]
    for fname, fn, kwargs in tasks:
        try:
            path = os.path.join(out_dir, fname)
            # Export comparison charts with larger text for readability
            fn(frameworks, path, scale=1.6, **kwargs)
            ok(f"{fname}  →  {C.CY}{path}{C.R}")
        except Exception as e:
            err(f"{fname}: {e}")

    # Individual spider charts per framework
    try:
        individual_paths = plot_framework_individual_spiders(frameworks, out_dir, scale=1.6)
        for path in individual_paths:
            fname = os.path.basename(path)
            ok(f"{fname}  →  {C.CY}{path}{C.R}")
    except Exception as e:
        err(f"Individual spider charts: {e}")

    # Box plots for key metrics
    for col, lbl, fname in [
        ("Correctness",        "Correctness [1–5]",    "04_BoxPlot_Correctness.png"),
        ("Execution Time (ms)","Latency (ms)",         "05_BoxPlot_Latency.png"),
        ("RAGking_score",      "RAGking Score [0–100]","06_BoxPlot_RAGking.png"),
    ]:
        try:
            path = os.path.join(out_dir, fname)
            plot_framework_box(frameworks, col, lbl, path, scale=1.6)
            ok(f"{fname}  →  {C.CY}{path}{C.R}")
        except Exception as e:
            err(f"{fname}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  7 — LANGUAGE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def menu_language():
    banner()
    section(tr("lang_section"))
    ui_lang = ask(tr("framework_lang"), state.get("ui_lang", "en")).lower()
    if ui_lang not in ["es", "en"]:
        ui_lang = state.get("ui_lang", "en")
    rep_lang = ask(tr("report_lang"), state.get("report_lang", ui_lang)).lower()
    if rep_lang not in ["es", "en"]:
        rep_lang = state.get("report_lang", ui_lang)
    state["ui_lang"]    = ui_lang
    state["report_lang"] = rep_lang
    ok(tr("lang_saved"))
    pause()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    while True:
        banner()
        status_badge()
        w = state["weights"]
        c = state["llm"]

        print(f"""  {C.B}{C.WH}{tr('main_menu')}{C.R}
  {'─'*58}

  {C.YL}[1]{C.R}  {C.WH}{tr('load_dataset')}{C.R}
       {C.GY}Import a CSV with RAG pipeline results{C.R}

  {C.YL}[2]{C.R}  {C.WH}{tr('adjust_weights')}{C.R}
    {C.GY}wC={w['correctness']}  wE={w.get('economic', w.get('token_eff', 0.0))}  wL={w['latency']}  wM={w['maintenance']}  wF={w['failure']}{C.R}

  {C.YL}[3]{C.R}  {C.WH}{tr('llm_settings')}{C.R}
       {C.GY}Judge: {c['judge_model']}   Embed: {c['embedding_model']}{C.R}

  {C.YL}[4]{C.R}  {C.WH}{tr('generate_raw_metrics')}{C.R}
      {C.GY}Generate raw per-question metrics and save bounds{C.R}

  {C.YL}[5]{C.R}  {C.WH}{tr('generate_report')}{C.R}
      {C.GY}Generate report from bounds · ASCII charts · Excel + CSV + PDF{C.R}

  {C.YL}[6]{C.R}  {C.WH}{tr('compare_frameworks')}{C.R}
      {C.GY}Load 2+ frameworks · bar charts · spider · PDF{C.R}

  {C.YL}[7]{C.R}  {C.WH}{tr('language_settings')}{C.R}
      {C.GY}UI: {state.get('ui_lang','en')}   PDF: {state.get('report_lang','en')}{C.R}

  {C.YL}[0]{C.R}  {C.WH}{tr('exit')}{C.R}

  {'─'*58}""")

        choice = ask(tr("select_option"), "")
        if   choice == "1": menu_load_csv()
        elif choice == "2": menu_weights()
        elif choice == "3": menu_llm_settings()
        elif choice == "4": menu_raw_metrics()
        elif choice == "5": menu_reports()
        elif choice == "6": menu_compare_frameworks()
        elif choice == "7": menu_language()
        elif choice == "0":
            banner()
            print(f"\n  {C.CY}{C.B}Thanks for using RAGking.{C.R}\n")
            sys.exit(0)
        else:
            warn(tr("invalid_option"))
            time.sleep(0.7)
