"""
charts/plots.py — All matplotlib chart generators.

Charts exported:
  - Distribution (correctness / latency / RAGking score)
  - Dashboard 4-panel
  - Model comparison bar chart
  - Framework comparison (multi-bar)
  - Spider / radar chart (only for framework comparison, based on min/max)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from reportlab.lib import colors as rl_colors


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE → COLOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def mpl_score_color(score: float) -> str:
    if score >= 80:   return "#2ecc71"
    elif score >= 60: return "#f39c12"
    else:             return "#e74c3c"


def rl_score_color(score: float):
    if score >= 80:   return rl_colors.HexColor("#2ecc71")
    elif score >= 60: return rl_colors.HexColor("#f39c12")
    else:             return rl_colors.HexColor("#e74c3c")


# Consistent palette for multi-framework charts
FRAMEWORK_PALETTE = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-METRIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
def plot_distribution(data, filename: str, xlabel: str = "Value",
                      color: str = "#3498db", dpi: int = 180, scale: float = 1.0) -> str:
    """Histogram of a single metric series. Returns filename."""
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    series = data.dropna()
    # If this looks like a Correctness score (1..5), use integer bins and integer x-ticks
    if len(series) > 0 and series.min() >= 1 and series.max() <= 5:
        # Use explicit bin edges so bar centers align with integer ticks 1..5
        bins_edges = np.arange(0.5, 5.6, 1.0)
        ax.hist(series, bins=bins_edges, color=color, edgecolor="black", alpha=0.75, linewidth=1.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)
    else:
        ax.hist(series, bins=15, color=color, edgecolor="black", alpha=0.75, linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=int(16 * scale), fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=int(16 * scale), fontweight="bold")
    ax.tick_params(labelsize=int(14 * scale))
    ax.grid(axis="y", alpha=0.3)

    stats = f"Mean = {data.mean():.2f}\nStd = {data.std():.2f}\nn = {len(data)}"
    ax.text(0.98, 0.97, stats, transform=ax.transAxes,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=int(13 * scale), fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  4-PANEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def plot_dashboard(df, filename: str, dpi: int = 150, scale: float = 1.0) -> str:
    """4-panel metrics overview dashboard."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # 1 — Correctness
    ax1 = fig.add_subplot(gs[0, 0])
    # Histogram for correctness: enforce integer bins and integer ticks 1..5
    corr = df["Correctness"].dropna()
    if len(corr) > 0 and corr.min() >= 1 and corr.max() <= 5:
        bins_edges = np.arange(0.5, 5.6, 1.0)
        ax1.hist(corr, bins=bins_edges, color="#2ecc71", edgecolor="black", alpha=0.7, linewidth=1.5)
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels(["1", "2", "3", "4", "5"], ha="center")
        ax1.set_xlim(0.5, 5.5)
    else:
        ax1.hist(corr, bins=12, color="#2ecc71", edgecolor="black", alpha=0.7, linewidth=1.5)
    ax1.set_title("Correctness Distribution", fontsize=int(13 * scale), fontweight="bold", loc="center")
    ax1.set_xlabel("Score [1–5]",   fontsize=int(12 * scale)); ax1.set_ylabel("Frequency", fontsize=int(12 * scale))
    ax1.tick_params(labelsize=int(11 * scale)); ax1.grid(axis="y", alpha=0.3)

    # 2 — Latency
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["Execution Time (ms)"].dropna(), bins=15, color="#f39c12", edgecolor="black", alpha=0.7, linewidth=1.5)
    ax2.set_title("Latency Distribution", fontsize=int(13 * scale), fontweight="bold", loc="center")
    ax2.set_xlabel("Time (ms)", fontsize=int(12 * scale)); ax2.set_ylabel("Frequency", fontsize=int(12 * scale))
    ax2.tick_params(labelsize=int(11 * scale)); ax2.grid(axis="y", alpha=0.3)

    # 3 — RAGking score
    ax3 = fig.add_subplot(gs[1, 0])
    vals = df["RAGking_score"].dropna()
    ax3.hist(vals, bins=15, color="#3498db", edgecolor="black", alpha=0.7, linewidth=1.5)
    ax3.axvline(vals.mean(), color="red", linestyle="--", linewidth=2.5,
                label=f"Mean: {vals.mean():.1f}")
    ax3.set_title("RAGking Score Distribution", fontsize=int(13 * scale), fontweight="bold", loc="center")
    ax3.set_xlabel("Score [0–100]", fontsize=int(12 * scale)); ax3.set_ylabel("Frequency", fontsize=int(12 * scale))
    ax3.tick_params(labelsize=int(11 * scale)); ax3.legend(fontsize=int(11 * scale)); ax3.grid(axis="y", alpha=0.3)

    # 4 — Scatter correctness vs latency
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(df["Execution Time (ms)"], df["Correctness"],
                     c=df["RAGking_score"], cmap="RdYlGn", s=120, alpha=0.6, edgecolor="black", linewidth=1)
    ax4.set_title("Correctness vs Latency", fontsize=int(13 * scale), fontweight="bold", loc="center")
    ax4.set_xlabel("Latency (ms)", fontsize=int(12 * scale)); ax4.set_ylabel("Correctness [1–5]", fontsize=int(12 * scale))
    ax4.tick_params(labelsize=int(11 * scale)); ax4.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label("RAGking Score", fontsize=int(11 * scale), fontweight="bold")
    cbar.ax.tick_params(labelsize=int(10 * scale))

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-DATASET MODEL COMPARISON (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(df, filename: str, dpi: int = 180, scale: float = 1.0) -> str | None:
    """Horizontal bar chart comparing models within one dataset."""
    if "Model" not in df.columns:
        return None
    models = df["Model"].unique()
    if len(models) <= 1:
        return None

    model_scores = df.groupby("Model")["RAGking_score"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(13 * scale, 8 * scale))
    fig.patch.set_facecolor("white")

    bar_colors = [mpl_score_color(s) for s in model_scores.values]
    ax.barh(model_scores.index, model_scores.values, color=bar_colors, edgecolor="black", linewidth=2)
    ax.set_xlabel("RAGking Score", fontsize=int(17 * scale), fontweight="bold")
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=int(15 * scale))

    for i, (_, val) in enumerate(model_scores.items()):
        ax.text(val + 2, i, f"{val:.1f}", va="center", fontweight="bold", fontsize=int(15 * scale))

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-FRAMEWORK COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════════
COMPARE_METRICS = {
    "RAGking Score":       ("RAGking_score",        "higher"),
    "Correctness":         ("Correctness",           "higher"),
    "Latency (ms)":        ("Execution Time (ms)",   "lower"),
    "String Similarity":   ("String Similarity",     "higher"),
    "Economic Efficiency": ("economic_eff_raw",      "higher"),
    "Cost per Query (USD)":("cost_per_query",        "lower"),
    "Failure Rate (%)":    ("failure_rate",          "lower"),
}


def plot_framework_comparison_bars(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Grouped bar chart comparing multiple frameworks across KEY metrics.
    Escala logarítmica para manejar diferencias masivas de magnitud.
    ARREGLADO: Los números se posicionan correctamente arriba de cada barra en escala log.
    """
    names   = list(frameworks.keys())
    metrics = list(COMPARE_METRICS.keys())
    x       = np.arange(len(metrics))
    n       = len(names)
    width   = 0.7 / n

    fig, ax = plt.subplots(figsize=(16 * scale, 9 * scale))
    fig.patch.set_facecolor("white")

    for i, (name, df) in enumerate(frameworks.items()):
        values = []
        for m_label, (col, _) in COMPARE_METRICS.items():
            if col in df.columns:
                val = df[col].mean() * 100 if col == "failure_rate" else df[col].mean()
            else:
                val = 0.0
            
            # Clip mínimo para escala logarítmica
            values.append(max(val, 0.001))

        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=FRAMEWORK_PALETTE[i % len(FRAMEWORK_PALETTE)],
                      edgecolor="black", linewidth=1.2, alpha=0.85)

        # Posicionar texto con manejo correcto de escala logarítmica
        for bar, val in zip(bars, values):
            display_val = 0.0 if val <= 0.001 else val
            # En escala log, posicionar el texto arriba sumando un factor en el espacio log
            bar_center_x = bar.get_x() + bar.get_width() / 2
            bar_height = bar.get_height()
            
            # Multiplicar la altura en lugar de sumar, para que funcione en log scale
            y_position = bar_height * 1.25
            
            ax.text(bar_center_x, y_position,
                    f"{display_val:.1f}", 
                    ha="center", va="bottom", 
                    fontsize=int(10 * scale), fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=int(11 * scale), fontweight="bold", rotation=0, ha="center")
    ax.tick_params(axis="y", labelsize=int(12 * scale))
    ax.set_ylabel("Value (Log Scale)", fontsize=int(14 * scale), fontweight="bold")
    
    # ESCALA LOGARÍTMICA
    ax.set_yscale("log") 
    
    ax.legend(fontsize=int(12 * scale), loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.4, which="both", linewidth=0.8)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_ragking_bars(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """Simple horizontal bar chart of final RAGking scores per framework."""
    scores = {
        name: df["RAGking_score"].mean()
        for name, df in frameworks.items()
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(12 * scale, max(5 * scale, len(scores) * 1.3 * scale)))
    fig.patch.set_facecolor("white")

    for i, (name, score) in enumerate(sorted_scores.items()):
        ax.barh(name, score, color=mpl_score_color(score), edgecolor="black", linewidth=2)
        ax.text(score + 1, i, f"{score:.2f}", va="center", fontsize=int(14 * scale), fontweight="bold")

    ax.set_xlabel("RAGking Score [0–100]", fontsize=int(16 * scale), fontweight="bold")
    ax.set_xlim(0, 105)
    ax.tick_params(labelsize=int(14 * scale))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  SPIDER / RADAR — for framework comparison only (min/max normalized)
# ══════════════════════════════════════════════════════════════════════════════
# Añadimos Min y Max absolutos (None si queremos que se calcule dinámicamente)
SPIDER_METRICS = [
    ("Correctness",       "Correctness",         "higher", 1.0, 5.0),
    ("Latency",           "Execution Time (ms)", "lower",  0.0, None),
    ("Economic Eff.",     "economic_eff_raw",    "higher", 0.0, None),
    ("Similarity",        "String Similarity",   "higher", 0.0, 1.0),
    ("Deploy Time",       "deploy_hours",        "lower",  0.0, None),
    ("Difficulty",        "implementation_difficulty", "lower", 1.0, 5.0),
    ("Maint. Effort",     "maintenance_hours_week", "lower", 0.0, None),
]

def _get_spider_values(frameworks: dict) -> tuple[list, dict]:
    """
    Compute normalized spider values.
    Usa límites absolutos para evitar distorsiones extremas al comparar solo 2 opciones.
    """
    categories = [m[0] for m in SPIDER_METRICS]

    # Recopilar medias brutas
    raw: dict[str, list] = {}
    for name, df in frameworks.items():
        row = []
        for _, col, _, _, _ in SPIDER_METRICS:
            if col in df.columns:
                vals = df[col].dropna()
                val = float(vals.median() if len(vals) > 1 else vals.mean()) if len(vals) > 0 else 0.0
            else:
                val = 0.0
            row.append(val)
        raw[name] = row

    # Obtener los máximos reales en caso de que alguna métrica no tenga tope (None)
    raw_matrix = np.array(list(raw.values()))
    data_maxs = raw_matrix.max(axis=0) if len(raw_matrix) > 0 else np.zeros(len(SPIDER_METRICS))

    normalized: dict[str, list] = {}
    for name, values in raw.items():
        norm = []
        for i, (_, _, direction, abs_min, abs_max) in enumerate(SPIDER_METRICS):
            metric_min = abs_min
            # Si no hay límite max absoluto, usa el mayor de los datos + 15% para dar respiro
            metric_max = abs_max if abs_max is not None else (data_maxs[i] * 1.15)
            
            rng = metric_max - metric_min
            if rng < 1e-9:
                n_val = 0.5
            else:
                n_val = (values[i] - metric_min) / rng
                n_val = max(0.0, min(1.0, n_val)) # Clampear para que no se salga del radar

                if direction == "lower":
                    n_val = 1.0 - n_val
            norm.append(n_val)
        normalized[name] = norm

    return categories, normalized


def plot_framework_spider(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Spider / radar chart for framework comparison.
    Values are min-max normalized across ALL frameworks.
    Generates: comparison chart (all on one) with enhanced aesthetics.
    """
    categories, normalized = _get_spider_values(frameworks)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(14 * scale, 14 * scale), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("white")

    # Plot each framework
    for i, (name, values) in enumerate(normalized.items()):
        vals = values + values[:1]
        color = FRAMEWORK_PALETTE[i % len(FRAMEWORK_PALETTE)]
        
        # Thicker lines, larger markers
        ax.plot(angles, vals, "o-", linewidth=4 * scale, markersize=12 * scale, 
            color=color, zorder=3)
        ax.fill(angles, vals, alpha=0.15, color=color)
        
        # Display values on each vertex
        for angle, value in zip(angles[:-1], values):
                ax.text(angle, value + 0.08, f"{value:.2f}",
                     ha="center", va="center",
                         fontsize=int(14 * scale), fontweight="bold", color=color, zorder=4)

    # Large category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=int(20 * scale), fontweight="bold", ha="center")
    
    # Y-axis tweaks
    ax.set_ylim(0, 1.15)  # Extra space for text
    ax.tick_params(labelsize=int(15 * scale))
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=int(13 * scale), color="grey")
    ax.grid(True, linewidth=1.5, alpha=0.7)
    
    # Legend with more space
    # Build legend showing framework names with mean RAGking score (colored)
    try:
        rag_scores = {name: (frameworks[name]["RAGking_score"].mean() if "RAGking_score" in frameworks[name].columns else 0.0)
                      for name in normalized.keys()}
        patches = [mpatches.Patch(color=FRAMEWORK_PALETTE[i % len(FRAMEWORK_PALETTE)],
                      label=f"{name} ({rag_scores.get(name,0.0):.1f})")
               for i, name in enumerate(normalized.keys())]
        legend = ax.legend(
            handles=patches,
            loc="upper right",
            bbox_to_anchor=(1.34, 1.22),
            fontsize=int(18 * scale),
            title="RAGking Score",
            title_fontsize=int(20 * scale),
            frameon=True,
            labelspacing=0.9,
            handlelength=1.6,
            borderpad=1.0,
            handletextpad=0.9,
            borderaxespad=0.8,
        )
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("bold")
    except Exception:
        fallback = ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.34, 1.22),
            fontsize=18,
            title="RAGking Score",
            title_fontsize=20,
            frameon=True,
        )
        if fallback.get_title() is not None:
            fallback.get_title().set_fontweight("bold")

    note = "Values normalized to [0–1] · Best performer: outer ring"
    fig.text(0.5, 0.02, note, ha="center", fontsize=int(14 * scale), color="grey", style="italic")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    return filename


def plot_framework_individual_spiders(frameworks: dict, out_dir: str, dpi: int = 180, scale: float = 1.0) -> list:
    """
    Generate individual spider charts for each framework.
    Returns list of generated filenames.
    """
    import os
    categories, all_normalized = _get_spider_values(frameworks)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    generated = []
    for i, (name, values) in enumerate(all_normalized.items()):
        vals = values + values[:1]
        color = FRAMEWORK_PALETTE[i % len(FRAMEWORK_PALETTE)]

        fig, ax = plt.subplots(figsize=(12 * scale, 12 * scale), subplot_kw=dict(projection="polar"))
        fig.patch.set_facecolor("white")

        # Plot single framework
        ax.plot(angles, vals, "o-", linewidth=4 * scale, markersize=12 * scale,
            color=color, zorder=3)
        ax.fill(angles, vals, alpha=0.25, color=color)

        # Display values on vertices
        for angle, value in zip(angles[:-1], values):
                 ax.text(angle, value + 0.08, f"{value:.2f}",
                     ha="center", va="center",
                     fontsize=int(15 * scale), fontweight="bold", color=color, zorder=4)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=int(20 * scale), fontweight="bold", ha="center")
        ax.set_ylim(0, 1.15)
        ax.tick_params(labelsize=int(15 * scale))
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=int(13 * scale), color="grey")
        ax.grid(True, linewidth=1.5, alpha=0.7)

        # Title
        fig.suptitle(f"{name}", fontsize=int(24 * scale), fontweight="bold", y=0.98)
        note = "Normalized scores [0–1] per dimension"
        fig.text(0.5, 0.02, note, ha="center", fontsize=int(14 * scale), color="grey", style="italic")

        fname = os.path.join(out_dir, f"03_Spider_{name}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        generated.append(fname)

    return generated


def plot_framework_box(frameworks: dict, metric_col: str, metric_label: str,
                       filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """Box-plot comparing distributions of one metric across frameworks."""
    import pandas as pd
    data  = [df[metric_col].dropna().values for df in frameworks.values() if metric_col in df.columns]
    names = [name for name, df in frameworks.items() if metric_col in df.columns]

    if not data:
        return filename

    fig, ax = plt.subplots(figsize=(max(8 * scale, len(names) * 2 * scale), 7 * scale))
    fig.patch.set_facecolor("white")

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)
    for patch, color in zip(bp["boxes"], FRAMEWORK_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=int(13 * scale), fontweight="bold", ha="center")
    ax.set_ylabel(metric_label, fontsize=int(14 * scale), fontweight="bold")
    ax.tick_params(labelsize=int(12 * scale))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL COMPARATIVE CHARTS FOR 2+ FRAMEWORKS
# ══════════════════════════════════════════════════════════════════════════════

def plot_framework_latency_comparison(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Lado a lado: Latencia promedio por framework con barras detalladas.
    Similar al ejemplo de FIGURA 4: TIME.
    """
    names = list(frameworks.keys())
    latencies = []
    stds = []
    
    for name, df in frameworks.items():
        lat = df["Execution Time (ms)"].dropna()
        latencies.append(lat.mean())
        stds.append(lat.std())
    
    fig, ax = plt.subplots(figsize=(max(10 * scale, len(names) * 3 * scale), 7 * scale))
    fig.patch.set_facecolor("white")
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, latencies, yerr=stds, capsize=15, 
                  color=FRAMEWORK_PALETTE[:len(names)], 
                  edgecolor="black", linewidth=2, alpha=0.85,
                  error_kw={"elinewidth": 2, "capthick": 3})
    
    # Añadir valores arriba de cada barra
    for i, (bar, val) in enumerate(zip(bars, latencies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + stds[i] + (max(latencies) * 0.02),
                f"{val:.0f} ms", ha="center", va="bottom", 
                fontsize=int(12 * scale), fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    
    ax.set_xlabel("Framework", fontsize=int(14 * scale), fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontsize=int(14 * scale), fontweight="bold")
    ax.set_title("Latency Comparison Across Frameworks", fontsize=int(16 * scale), fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=int(13 * scale), fontweight="bold")
    ax.tick_params(axis="y", labelsize=int(12 * scale))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_correctness_comparison(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Comparación de Correctness (LLM-as-Judge) entre frameworks.
    Muestra media y distribución.
    """
    names = list(frameworks.keys())
    correctness_data = []
    means = []
    stds = []
    
    for name, df in frameworks.items():
        corr = df["Correctness"].dropna()
        correctness_data.append(corr.values)
        means.append(corr.mean())
        stds.append(corr.std())
    
    fig, ax = plt.subplots(figsize=(max(10 * scale, len(names) * 3 * scale), 7 * scale))
    fig.patch.set_facecolor("white")
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=15,
                  color=FRAMEWORK_PALETTE[:len(names)],
                  edgecolor="black", linewidth=2, alpha=0.85,
                  error_kw={"elinewidth": 2, "capthick": 3})
    
    # Valores arriba de las barras
    for i, (bar, val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + stds[i] + 0.1,
                f"{val:.2f} / 5", ha="center", va="bottom",
                fontsize=int(12 * scale), fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))
    
    # Mostrar también los puntos individuales de forma suave
    for i, data in enumerate(correctness_data):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.25, s=40, color=FRAMEWORK_PALETTE[i], zorder=2)
    
    ax.set_xlabel("Framework", fontsize=int(14 * scale), fontweight="bold")
    ax.set_ylabel("Correctness Score [1–5]", fontsize=int(14 * scale), fontweight="bold")
    ax.set_title("Correctness Comparison (LLM-as-Judge)", fontsize=int(16 * scale), fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=int(13 * scale), fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.tick_params(axis="y", labelsize=int(12 * scale))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_tokens_comparison(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Desglose de tokens: Prompt, Completion y Total por framework.
    Gráfico de barras apiladas o lado a lado.
    """
    names = list(frameworks.keys())
    prompt_tokens = []
    completion_tokens = []
    
    for name, df in frameworks.items():
        prompt_tokens.append(df["Prompt Tokens"].mean())
        completion_tokens.append(df["Completion Tokens"].mean())
    
    fig, ax = plt.subplots(figsize=(max(11 * scale, len(names) * 3.5 * scale), 7 * scale))
    fig.patch.set_facecolor("white")
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    # Barras apiladas
    bars1 = ax.bar(x_pos - width/2, prompt_tokens, width, label="Prompt Tokens",
                   color="#3498db", edgecolor="black", linewidth=1.5, alpha=0.85)
    bars2 = ax.bar(x_pos + width/2, completion_tokens, width, label="Completion Tokens",
                   color="#e74c3c", edgecolor="black", linewidth=1.5, alpha=0.85)
    
    # Valores arriba de cada barra
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + (max(prompt_tokens + completion_tokens) * 0.02),
                    f"{int(height):,}", ha="center", va="bottom",
                    fontsize=int(10 * scale), fontweight="bold")
    
    ax.set_xlabel("Framework", fontsize=int(14 * scale), fontweight="bold")
    ax.set_ylabel("Token Count (Average)", fontsize=int(14 * scale), fontweight="bold")
    ax.set_title("Token Consumption Comparison", fontsize=int(16 * scale), fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=int(13 * scale), fontweight="bold")
    ax.tick_params(axis="y", labelsize=int(12 * scale))
    ax.legend(fontsize=int(13 * scale), loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_economic_comparison(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Comparación económica: Costo por query vs Economic Efficiency.
    Scatter plot: X=Costo, Y=Eficiencia
    """
    names = list(frameworks.keys())
    costs = []
    efficiencies = []
    
    for name, df in frameworks.items():
        cost = df["cost_per_query"].mean()
        eff = df["economic_eff_raw"].mean()
        costs.append(cost)
        efficiencies.append(eff)
    
    fig, ax = plt.subplots(figsize=(12 * scale, 8 * scale))
    fig.patch.set_facecolor("white")
    
    # Scatter plot con líneas conectando
    for i in range(len(names)):
        ax.scatter(costs[i], efficiencies[i], s=500, 
                  color=FRAMEWORK_PALETTE[i], edgecolor="black", linewidth=2, 
                  zorder=3, alpha=0.85)
        ax.text(costs[i], efficiencies[i] + (max(efficiencies) * 0.05),
                names[i], ha="center", va="bottom",
                fontsize=int(13 * scale), fontweight="bold")
        
        # Mostrar valores
        ax.text(costs[i] - (max(costs) * 0.02), efficiencies[i] - (max(efficiencies) * 0.05),
                f"${costs[i]:.6f}\nEff: {efficiencies[i]:.2f}",
                ha="right", va="top",
                fontsize=int(9 * scale),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel("Cost per Query (USD)", fontsize=int(14 * scale), fontweight="bold")
    ax.set_ylabel("Economic Efficiency (Correctness/USD)", fontsize=int(14 * scale), fontweight="bold")
    ax.set_title("Economic Analysis: Cost vs Efficiency", fontsize=int(16 * scale), fontweight="bold", pad=20)
    ax.tick_params(labelsize=int(12 * scale))
    ax.grid(alpha=0.3, linestyle="--")
    
    # Quadrants
    ax.axvline(np.median(costs), color="grey", linestyle=":", alpha=0.5, linewidth=1.5)
    ax.axhline(np.median(efficiencies), color="grey", linestyle=":", alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_performance_breakdown(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Desglose de rendimiento: 2x2 gráficos mostrando Correctness, Latency, Similarity y RAGking.
    """
    names = list(frameworks.keys())
    n_fw = len(names)
    
    data = {
        "Correctness": [frameworks[n]["Correctness"].mean() for n in names],
        "Latency (ms)": [frameworks[n]["Execution Time (ms)"].mean() for n in names],
        "Similarity": [frameworks[n]["String Similarity"].mean() for n in names],
        "RAGking Score": [frameworks[n]["RAGking_score"].mean() for n in names],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14 * scale, 10 * scale))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    
    metrics_info = [
        ("Correctness", data["Correctness"], "[1–5]", "#2ecc71", [1, 5]),
        ("Latency (ms)", data["Latency (ms)"], "[ms]", "#f39c12", None),
        ("String Similarity", data["Similarity"], "[0–1]", "#9b59b6", [0, 1]),
        ("RAGking Score", data["RAGking Score"], "[0–100]", "#3498db", [0, 100]),
    ]
    
    for idx, (metric_name, values, unit, color, y_limits) in enumerate(metrics_info):
        ax = axes[idx]
        x_pos = np.arange(n_fw)
        bars = ax.bar(x_pos, values, color=color, edgecolor="black", linewidth=2, alpha=0.85)
        
        # Valores arriba
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + (max(values) * 0.03),
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=int(11 * scale), fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        ax.set_ylabel(f"{metric_name} {unit}", fontsize=int(12 * scale), fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=int(11 * scale), fontweight="bold")
        if y_limits:
            ax.set_ylim(y_limits[0], y_limits[1] * 1.15)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=int(10 * scale))
    
    fig.suptitle("Performance Breakdown Comparison", fontsize=int(18 * scale), fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def plot_framework_similarity_distribution(frameworks: dict, filename: str, dpi: int = 180, scale: float = 1.0) -> str:
    """
    Distribución de String Similarity como histogramas superpuestos.
    """
    fig, ax = plt.subplots(figsize=(13 * scale, 7 * scale))
    fig.patch.set_facecolor("white")
    
    for i, (name, df) in enumerate(frameworks.items()):
        sim = df["String Similarity"].dropna()
        ax.hist(sim, bins=15, alpha=0.6, label=name, 
               color=FRAMEWORK_PALETTE[i % len(FRAMEWORK_PALETTE)],
               edgecolor="black", linewidth=1.5)
    
    ax.set_xlabel("String Similarity [0–1]", fontsize=int(14 * scale), fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=int(14 * scale), fontweight="bold")
    ax.set_title("String Similarity Distribution", fontsize=int(16 * scale), fontweight="bold", pad=20)
    ax.tick_params(labelsize=int(12 * scale))
    ax.legend(fontsize=int(13 * scale), loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename
