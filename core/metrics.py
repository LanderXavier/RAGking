"""
core/metrics.py — RAGking score computation and normalization.
"""

import json
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import numpy as np

from core.config import state, tr
from ui.helpers import ask_float_range, section, info


BOUNDS_FILE = Path(__file__).resolve().parents[1] / "bounds.json"


def _to_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def normalize_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([1.0] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if invert else norm


def normalize_with_bounds(series: pd.Series, minimum: float, maximum: float, invert: bool = False) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = _to_float(minimum, 0.0)
    maximum = _to_float(maximum, 0.0)
    if maximum == minimum:
        return pd.Series([1.0] * len(series), index=series.index)
    norm = ((series - minimum) / (maximum - minimum)).clip(0.0, 1.0)
    return 1 - norm if invert else norm


def load_bounds_store() -> dict:
    if not BOUNDS_FILE.exists():
        return {}
    try:
        data = json.loads(BOUNDS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_bounds_profile(profile_name: str, bounds: dict, source_name: str = None, row_count: int = None) -> None:
    profile_name = str(profile_name or "").strip()
    if not profile_name:
        raise ValueError("Profile name is required.")

    bounds = dict(bounds or {})
    bounds["correctness"] = {"min": 1.0, "max": 5.0}

    store = load_bounds_store()
    store[profile_name] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_name,
        "rows": row_count,
        "bounds": bounds,
    }
    BOUNDS_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def make_bounds_snapshot_name(base_name: str) -> str:
    base_name = str(base_name or "bounds").strip().lower()
    base_name = re.sub(r"[^a-z0-9._-]+", "_", base_name)
    base_name = re.sub(r"_+", "_", base_name).strip("_") or "bounds"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{stamp}"


def list_bounds_profiles() -> list[str]:
    return sorted(load_bounds_store().keys())


def get_bounds_profile(profile_name: str) -> dict | None:
    profile_name = str(profile_name or "").strip()
    if not profile_name:
        return None
    store = load_bounds_store()
    profile = store.get(profile_name)
    if not isinstance(profile, dict):
        return None
    bounds = profile.get("bounds", {})
    return bounds if isinstance(bounds, dict) else None


def build_global_bounds() -> dict | None:
    store = load_bounds_store()
    if not store:
        return None

    metric_names = ["correctness", "economic", "latency", "maintenance"]
    global_bounds = {}

    for metric_name in metric_names:
        mins = []
        maxs = []
        for profile in store.values():
            bounds = profile.get("bounds", {}) if isinstance(profile, dict) else {}
            metric_bounds = bounds.get(metric_name, {}) if isinstance(bounds, dict) else {}
            if not isinstance(metric_bounds, dict):
                continue
            min_value = metric_bounds.get("min")
            max_value = metric_bounds.get("max")
            if min_value is not None and max_value is not None:
                mins.append(_to_float(min_value))
                maxs.append(_to_float(max_value))
        if mins and maxs:
            global_bounds[metric_name] = {"min": min(mins), "max": max(maxs)}

    return global_bounds if global_bounds else None


def extract_metric_bounds(df: pd.DataFrame) -> dict:
    def _series_or_empty(column_name: str) -> pd.Series:
        if column_name not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)

    metrics = {
        "correctness": _series_or_empty("Correctness"),
        "economic": _series_or_empty("economic_eff_raw"),
        "latency": _series_or_empty("Execution Time (ms)"),
        "maintenance": _series_or_empty("maintenance_score"),
    }
    bounds = {}
    for metric_name, series in metrics.items():
        valid = series.dropna()
        if len(valid) == 0:
            bounds[metric_name] = {"min": 0.0, "max": 0.0}
        else:
            bounds[metric_name] = {"min": float(valid.min()), "max": float(valid.max())}
    bounds["correctness"] = {"min": 1.0, "max": 5.0}
    return bounds


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE LABELS / COLORS
# ══════════════════════════════════════════════════════════════════════════════
def get_score_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Acceptable"
    else:
        return "Needs Improvement"


# ══════════════════════════════════════════════════════════════════════════════
#  OPERATIONAL FACTORS COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def collect_operational_factors(df: pd.DataFrame, interactive: bool = True) -> dict:
    """
    Extract operational factors from CSV columns if available;
    fall back to user input if columns are missing.
    """
    has_deploy_cols = all(col in df.columns for col in ["deploy_hours", "implementation_difficulty", "maintenance_hours_week"])

    if has_deploy_cols:
        # LS Extract values directly from CSV columns
        section("📋 Operational Factors (from CSV)")
        factors = {}
        if "Model" in df.columns:
            models = sorted(df["Model"].dropna().astype(str).unique().tolist())
            for model in models:
                section(f"Extracting factors for {model}...")
                model_df = df[df["Model"] == model]
                factors[model] = {
                    "deploy_hours":           float(model_df["deploy_hours"].median()),
                    "difficulty":             float(model_df["implementation_difficulty"].median()),
                    "maintenance_hours_week": float(model_df["maintenance_hours_week"].median()),
                }
                info(f"  Deploy: {factors[model]['deploy_hours']:.1f}h | Difficulty: {factors[model]['difficulty']:.1f}/5 | Maint: {factors[model]['maintenance_hours_week']:.1f}h/wk")
        else:
            section("Extracting global operational factors...")
            factors["__global__"] = {
                "deploy_hours":           float(df["deploy_hours"].median()),
                "difficulty":             float(df["implementation_difficulty"].median()),
                "maintenance_hours_week": float(df["maintenance_hours_week"].median()),
            }
            info(f"  Deploy: {factors['__global__']['deploy_hours']:.1f}h | Difficulty: {factors['__global__']['difficulty']:.1f}/5 | Maint: {factors['__global__']['maintenance_hours_week']:.1f}h/wk")
        return factors

        # Fallback: ask user for manual input if columns not in CSV
        if not interactive:
                section("Operational Factors · Defaults")
                default_block = {
                        "deploy_hours": 24,
                        "difficulty": 3,
                        "maintenance_hours_week": 8,
                }
                if "Model" in df.columns:
                        models = sorted(df["Model"].dropna().astype(str).unique().tolist())
                        return {model: default_block.copy() for model in models}
                return {"__global__": default_block.copy()}

        section(tr("ops_input_title"))
        print(f"""
    These values are used in the Maintenance dimension (M).
    Factors: deploy time · implementation difficulty · weekly maintenance.
""")

    factors = {}
    if "Model" in df.columns:
        models = sorted(df["Model"].dropna().astype(str).unique().tolist())
        for model in models:
            section(tr("ops_model", model=model))
            factors[model] = {
                "deploy_hours":           ask_float_range(tr("ops_deploy"),     24, min_val=0),
                "difficulty":             ask_float_range(tr("ops_difficulty"),  3, min_val=1, max_val=5),
                "maintenance_hours_week": ask_float_range(tr("ops_maint"),       8, min_val=0),
            }
    else:
        section(tr("ops_global"))
        factors["__global__"] = {
            "deploy_hours":           ask_float_range(tr("ops_deploy"),     24, min_val=0),
            "difficulty":             ask_float_range(tr("ops_difficulty"),  3, min_val=1, max_val=5),
            "maintenance_hours_week": ask_float_range(tr("ops_maint"),       8, min_val=0),
        }
    return factors


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY OPERATIONAL FACTORS → M_norm
# ══════════════════════════════════════════════════════════════════════════════
def apply_operational_factors(df: pd.DataFrame, operational_factors: dict) -> pd.DataFrame:
    df = df.copy()

    if not operational_factors:
        if "maintenance_score" not in df.columns:
            df["maintenance_score"] = 0.5
        df["M_norm"] = normalize_minmax(
            pd.to_numeric(df["maintenance_score"], errors="coerce").fillna(0.5), invert=True
        )
        return df

    def get_factor_row(model_name: str) -> dict:
        if model_name in operational_factors:
            return operational_factors[model_name]
        return operational_factors.get("__global__", {
            "deploy_hours": 24, "difficulty": 3, "maintenance_hours_week": 8,
        })

    if "Model" in df.columns:
        mapped = df["Model"].astype(str).apply(get_factor_row)
    else:
        default = get_factor_row("__global__")
        mapped = pd.Series([default] * len(df), index=df.index)

    df["deploy_hours"]              = mapped.apply(lambda x: float(x.get("deploy_hours", 24)))
    df["implementation_difficulty"] = mapped.apply(lambda x: float(x.get("difficulty", 3)))
    df["maintenance_hours_week"]    = mapped.apply(lambda x: float(x.get("maintenance_hours_week", 8)))

    deploy_burden      = (df["deploy_hours"] / 72.0).clip(0, 1)
    difficulty_burden  = ((df["implementation_difficulty"] - 1.0) / 4.0).clip(0, 1)
    maintenance_burden = (df["maintenance_hours_week"] / 20.0).clip(0, 1)

    burden = (
        0.35 * deploy_burden +
        0.35 * difficulty_burden +
        0.30 * maintenance_burden
    ).clip(0, 1)

    df["maintenance_score"] = burden
    df["M_norm"]            = 1 - burden
    return df


def compute_raw_metrics(
    df: pd.DataFrame,
    operational_factors: dict = None,
    failure_count_override: int = None,
) -> pd.DataFrame:
    df = df.copy()

    # Failure flag derived from core quality signals.
    derived_failure = (
        df["Correctness"].isna()
        | (pd.to_numeric(df["Correctness"], errors="coerce").fillna(0) == 0)
        | df["Generated Answer"].isna()
        | (df["Generated Answer"].astype(str).str.strip() == "")
    )

    # If CSV already provides `is_failure`, keep it and OR it with derived signal.
    # This preserves explicit failure annotations from source datasets.
    if "is_failure" in df.columns:
        raw_failure = df["is_failure"]
        text_map = {
            "1": True, "true": True, "t": True, "yes": True, "y": True,
            "si": True, "sí": True, "fail": True, "failed": True, "failure": True,
            "0": False, "false": False, "f": False, "no": False, "n": False,
            "ok": False, "success": False, "passed": False, "pass": False, "": False,
        }
        parsed_text = raw_failure.astype(str).str.strip().str.lower().map(text_map)
        parsed_num = pd.to_numeric(raw_failure, errors="coerce").map(
            lambda x: np.nan if pd.isna(x) else bool(int(x))
        )
        csv_failure = parsed_text.where(parsed_text.notna(), parsed_num)
        csv_failure = csv_failure.fillna(False).astype(bool)
        df["is_failure"] = csv_failure | derived_failure
    else:
        df["is_failure"] = derived_failure

    # Cost per query and efficiency metrics.
    llm_conf = state.get("llm", {})
    price_table = llm_conf.get("price_table", {})

    def _get_prices_for_model(model_name: str) -> tuple[float, float]:
        if not model_name:
            return (llm_conf.get("input_cost_per_1m", 0.0), llm_conf.get("output_cost_per_1m", 0.0))
        model_key = str(model_name).strip()
        p = price_table.get(model_key, None)
        if p:
            return (p.get("input_per_1m", llm_conf.get("input_cost_per_1m", 0.0)),
                    p.get("output_per_1m", llm_conf.get("output_cost_per_1m", 0.0)))
        return (llm_conf.get("input_cost_per_1m", 0.0), llm_conf.get("output_cost_per_1m", 0.0))

    def _row_cost(r):
        in_p1m, out_p1m = _get_prices_for_model(r.get("Model") if "Model" in r else None)
        in_p = float(in_p1m) / 1_000_000.0
        out_p = float(out_p1m) / 1_000_000.0
        pt = float(r.get("Prompt Tokens") or 0)
        ct = float(r.get("Completion Tokens") or 0)
        return pt * in_p + ct * out_p

    df["cost_per_query"] = df.apply(_row_cost, axis=1)

    prompt_tokens = pd.to_numeric(df["Prompt Tokens"], errors="coerce").fillna(0.0)
    completion_tokens = pd.to_numeric(df["Completion Tokens"], errors="coerce").fillna(0.0)
    correctness_for_token_eff = pd.to_numeric(df["Correctness"], errors="coerce").fillna(0.0)
    correctness_safe = correctness_for_token_eff.where(correctness_for_token_eff >= 1.0, 1.0)
    df["token_eff_raw"] = (prompt_tokens + completion_tokens) / correctness_safe
    df["token_eff_raw"] = df["token_eff_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["_cost_safe"] = df["cost_per_query"].replace(0, np.nan)
    df["economic_eff_raw"] = df.apply(
        lambda r: float(r.get("Correctness") or 0.0) / (r["_cost_safe"] if pd.notna(r["_cost_safe"]) else np.nan),
        axis=1,
    )
    df["economic_eff_raw"] = df["economic_eff_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = apply_operational_factors(df, operational_factors)

    if failure_count_override is not None:
        fr = min(max(float(failure_count_override) / max(len(df), 1), 0.0), 1.0)
        df["failure_rate"] = fr
    elif "failure_rate" in df.columns:
        fr_csv = pd.to_numeric(df["failure_rate"], errors="coerce")
        if pd.notna(fr_csv).any() and float(fr_csv.max(skipna=True)) > 1.0:
            fr_csv = fr_csv / 100.0

        fr_fallback = pd.Series(df["is_failure"].mean(), index=df.index)
        df["failure_rate"] = fr_csv.where(fr_csv.notna(), fr_fallback).clip(0.0, 1.0)
    elif "Model" in df.columns:
        fr_map = df.groupby("Model")["is_failure"].mean().rename("failure_rate")
        df = df.merge(fr_map, on="Model", how="left")
    else:
        df["failure_rate"] = df["is_failure"].mean()

    df.drop(columns=["_cost_safe"], inplace=True, errors="ignore")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FULL METRICS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(
    df: pd.DataFrame,
    operational_factors: dict = None,
    failure_count_override: int = None,
    metric_bounds: dict = None,
) -> pd.DataFrame:
    df = compute_raw_metrics(
        df,
        operational_factors=operational_factors,
        failure_count_override=failure_count_override,
    )

    # Normalized dimensions
    correctness_series = pd.to_numeric(df["Correctness"], errors="coerce").fillna(0)
    economic_series = df["economic_eff_raw"].astype(float).fillna(0.0)
    latency_series = pd.to_numeric(df["Execution Time (ms)"], errors="coerce").fillna(0)

    if metric_bounds:
        c_bounds = metric_bounds.get("correctness") or {}
        e_bounds = metric_bounds.get("economic") or {}
        l_bounds = metric_bounds.get("latency") or {}
        m_bounds = metric_bounds.get("maintenance") or {}

        df["C_norm"] = (
            normalize_with_bounds(correctness_series, c_bounds.get("min"), c_bounds.get("max"))
            if c_bounds else normalize_minmax(correctness_series)
        )
        df["E_norm"] = (
            normalize_with_bounds(economic_series, e_bounds.get("min"), e_bounds.get("max"))
            if e_bounds else normalize_minmax(economic_series)
        )
        df["L_norm"] = (
            normalize_with_bounds(latency_series, l_bounds.get("min"), l_bounds.get("max"), invert=True)
            if l_bounds else normalize_minmax(latency_series, invert=True)
        )
    else:
        df["C_norm"] = normalize_minmax(correctness_series)
        # Economic efficiency normalization (higher is better)
        df["E_norm"] = normalize_minmax(economic_series)
        df["L_norm"] = normalize_minmax(latency_series, invert=True)
    if metric_bounds:
        if m_bounds:
            df["M_norm"] = normalize_with_bounds(
                pd.to_numeric(df["maintenance_score"], errors="coerce").fillna(0.0),
                m_bounds.get("min"),
                m_bounds.get("max"),
                invert=True,
            )

    df["F_norm"] = 1 - df["failure_rate"]

    # RAGking composite score using Economic Efficiency instead of token-efficiency
    w = state["weights"]
    # Ensure weights sum to 1 (fallback normalization)
    total_w = sum(w.values()) if isinstance(w, dict) else 1.0
    if not np.isclose(total_w, 1.0):
        w = {k: (v / total_w) for k, v in w.items()}

    df["RAGking_score"] = (
        w.get("correctness", 0.0) * df["C_norm"]  +
        w.get("economic", 0.0)    * df["E_norm"]  +
        w.get("latency", 0.0)    * df["L_norm"]  +
        w.get("maintenance", 0.0)* df["M_norm"]  +
        w.get("failure", 0.0)    * df["F_norm"]
    ) * 100
    df["RAGking_score"] = df["RAGking_score"].round(2)
    return df