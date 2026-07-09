#!/usr/bin/env python3
"""Render publication-style Figures 3-5 from frozen TreatAgent results.

The script reads only results/final_results and produces editable SVG as the
primary output, plus PDF, 600 dpi TIFF, and PNG previews. It never invokes the
model or an external API.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "final_results"
OUT_DIR = Path(os.getenv("TREATAGENT_FIGURE_OUT_DIR", str(ROOT / "figure" / "final_results")))
SOURCE_DIR = OUT_DIR / "source_data"

# ---------------------------------------------------------------------------
# Nature-style visual contract
# ---------------------------------------------------------------------------

BLUE = "#0B3B8C"
BLUE_2 = "#2F6FB2"
BLUE_3 = "#BFD7EE"
BLUE_PALE = "#EEF5FB"
INK = "#111111"
MUTED = "#6E7781"
GRID = "#E6E6E6"
AXIS = "#111111"
GREEN = "#4E9A45"
GREEN_PALE = "#E9F4E4"
RED = "#9F1111"
RED_PALE = "#F7E0E0"
ORANGE = "#F2A000"
ORANGE_PALE = "#FFF0C7"
PURPLE = "#7356A8"
PURPLE_PALE = "#EEE8F7"
GREY = "#929292"
GREY_PALE = "#F0F0F0"
WHITE = "#FFFFFF"

METHOD_COLORS = {
    "Direct": "#A9A9A9",
    "CoT": PURPLE,
    "RAG": ORANGE,
    "TreatAgent": BLUE,
}
SPLIT_COLORS = {
    "Drug-disjoint": BLUE,
    "Temporal-submit": PURPLE,
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------------
# IO and shared helpers
# ---------------------------------------------------------------------------

def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color=GRID, lw=0.65, zorder=0)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", length=0, pad=2, colors=INK)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str, title: str | None = None, *, y: float = 1.02) -> None:
    ax.text(
        -0.10,
        y,
        label.lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    if title:
        ax.text(
            0.0,
            y,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.8,
            color=INK,
            clip_on=False,
        )


def annotate_value(ax: plt.Axes, x: float, y: float, value: float, *, dy: float = 0.018, color: str = INK) -> None:
    ax.text(x, y + dy, f"{value:.2f}", ha="center", va="bottom", fontsize=5.8, color=color)


def index_rows(rows: Iterable[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def metric(index: dict[tuple[str, ...], dict[str, str]], split: str, method: str, key: str) -> float:
    return as_float(index[(split, method)][key])


def short_method(method: str) -> str:
    return "TreatAgent" if method == "TreatAgent" else method


def label_with_n(text: str, n: int) -> str:
    return f"{text}\n(n={n})"


# ---------------------------------------------------------------------------
# Figure 3: main performance and triage value
# ---------------------------------------------------------------------------

def panel_f3a(ax: plt.Axes, main: dict[tuple[str, str], dict[str, str]]) -> None:
    style_axis(ax)
    add_panel_label(ax, "a", "Overall F1 performance")
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    x = np.arange(len(methods))
    for split, marker, offset in [("Drug-disjoint", "o", -0.055), ("Temporal-submit", "^", 0.055)]:
        values = [metric(main, split, method, "f1") for method in methods]
        ax.plot(x + offset, values, color=SPLIT_COLORS[split], lw=1.05, alpha=0.55)
        ax.scatter(
            x + offset,
            values,
            marker=marker,
            s=25,
            color=[METHOD_COLORS[method] if method != "TreatAgent" else SPLIT_COLORS[split] for method in methods],
            edgecolor=INK,
            linewidth=0.45,
            zorder=3,
            label=split,
        )
        annotate_value(ax, x[-1] + offset, values[-1], values[-1], color=SPLIT_COLORS[split])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 0.82)
    ax.legend(loc="upper left", borderaxespad=0.2, fontsize=5.5, handletextpad=0.45, labelspacing=0.30)
    ax.text(2.34, 0.76, "+0.15 / +0.20 vs best baseline", fontsize=5.8, color=BLUE)


def panel_f3b(ax: plt.Axes, main: dict[tuple[str, str], dict[str, str]]) -> None:
    style_axis(ax, grid_axis="x")
    add_panel_label(ax, "b", "Missed-positive rate")
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    y = np.arange(len(methods))
    height = 0.30
    for split, offset, color in [("Drug-disjoint", -height / 2, BLUE_2), ("Temporal-submit", height / 2, PURPLE)]:
        values = [1.0 - metric(main, split, method, "recall") for method in methods]
        ax.barh(y + offset, values, height=height, color=color, alpha=0.86, label=split, zorder=3)
        for pos, value in zip(y + offset, values):
            ax.text(value + 0.018, pos, f"{value:.0%}", va="center", ha="left", fontsize=5.7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("Missed-positive rate (1 - recall)")
    ax.text(0.98, 1.04, "lower is better", transform=ax.transAxes, ha="right", fontsize=5.7, color=MUTED)


def panel_f3c(ax: plt.Axes, main: dict[tuple[str, str], dict[str, str]]) -> None:
    style_axis(ax)
    add_panel_label(ax, "c", "Precision-recall profile")
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    for split, marker in [("Drug-disjoint", "o"), ("Temporal-submit", "^")]:
        for method in methods:
            precision = metric(main, split, method, "precision")
            recall = metric(main, split, method, "recall")
            ax.scatter(
                recall,
                precision,
                s=30 if method == "TreatAgent" else 22,
                marker=marker,
                color=METHOD_COLORS[method],
                edgecolor=INK,
                linewidth=0.45,
                zorder=3,
            )
            if method in {"RAG", "TreatAgent"}:
                ax.annotate(
                    f"{method}\n{'DD' if split == 'Drug-disjoint' else 'TS'}",
                    (recall, precision),
                    xytext=(4, 3 if split == "Drug-disjoint" else -12),
                    textcoords="offset points",
                    fontsize=5.3,
                    color=METHOD_COLORS[method],
                )
    ax.set_xlim(0, 0.90)
    ax.set_ylim(0.42, 0.79)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.axvspan(0.70, 0.90, color=GREEN_PALE, alpha=0.45, zorder=0)
    ax.text(0.88, 0.44, "triage-sensitive\nregion", ha="right", fontsize=5.4, color=GREEN)


def panel_f3d(ax: plt.Axes, main: dict[tuple[str, str], dict[str, str]]) -> None:
    style_axis(ax)
    add_panel_label(ax, "d", "Captured versus missed positives")
    labels = ["DD\nRAG", "DD\nTreatAgent", "TS\nRAG", "TS\nTreatAgent"]
    refs = [
        ("Drug-disjoint", "RAG"),
        ("Drug-disjoint", "TreatAgent"),
        ("Temporal-submit", "RAG"),
        ("Temporal-submit", "TreatAgent"),
    ]
    captured = [as_int(main[ref]["tp"]) for ref in refs]
    missed = [as_int(main[ref]["fn"]) for ref in refs]
    x = np.arange(len(labels))
    ax.bar(x, captured, color=GREEN, label="Captured positives", zorder=3)
    ax.bar(x, missed, bottom=captured, color=RED_PALE, edgecolor=RED, linewidth=0.55, label="Missed positives", zorder=3)
    for i, (tp, fn) in enumerate(zip(captured, missed)):
        ax.text(i, tp / 2, str(tp), color=WHITE, ha="center", va="center", fontsize=6.2, fontweight="bold")
        ax.text(i, tp + fn / 2, str(fn), color=RED, ha="center", va="center", fontsize=6.2)
    ax.text(0.50, 252, "+82 captured", ha="center", fontsize=5.8, color=GREEN)
    ax.text(2.50, 252, "+123 captured", ha="center", fontsize=5.8, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Positive pairs")
    ax.set_ylim(0, 270)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, borderaxespad=0.0, fontsize=5.5)


def panel_f3e(ax: plt.Axes, main: dict[tuple[str, str], dict[str, str]]) -> None:
    add_panel_label(ax, "e", "Compact metric profile")
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["DD RAG", "DD TreatAgent", "TS RAG", "TS TreatAgent"]
    refs = [
        ("Drug-disjoint", "RAG"),
        ("Drug-disjoint", "TreatAgent"),
        ("Temporal-submit", "RAG"),
        ("Temporal-submit", "TreatAgent"),
    ]
    values = np.array([[metric(main, split, method, item) for item in metrics] for split, method in refs])
    cmap = LinearSegmentedColormap.from_list("blue_profile", [WHITE, BLUE_PALE, BLUE])
    ax.imshow(values, vmin=0.20, vmax=0.80, cmap=cmap, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.0, color=WHITE if values[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_f3f(ax: plt.Axes) -> None:
    add_panel_label(ax, "f", "Evidence auditability")
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    capabilities = [
        "Provenance",
        "Typed tuples",
        "Support / conflict",
        "Missing evidence",
        "Planner trace",
        "Structured explanation",
    ]
    matrix = np.array(
        [
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5, 1.0],
        ]
    )
    cmap = LinearSegmentedColormap.from_list("audit", [GREY_PALE, BLUE_3, BLUE])
    ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    symbols = {0.0: "-", 0.5: "partial", 1.0: "yes"}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, symbols[matrix[i, j]], ha="center", va="center", fontsize=5.6, color=WHITE if matrix[i, j] == 1 else INK)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_yticks(range(len(capabilities)))
    ax.set_yticklabels(capabilities)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure3() -> None:
    main_rows = read_csv("main_results.csv")
    main = index_rows(main_rows, "split", "setting")
    fig = plt.figure(figsize=(7.20, 5.20))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.48, left=0.09, right=0.98, top=0.92, bottom=0.10)
    panel_f3a(fig.add_subplot(gs[0, 0]), main)
    panel_f3b(fig.add_subplot(gs[0, 1]), main)
    panel_f3c(fig.add_subplot(gs[1, 0]), main)
    panel_f3d(fig.add_subplot(gs[1, 1]), main)
    fig.suptitle(
        "Figure 3 | TreatAgent reduces missed positives while improving candidate-level treatment triage",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=9.0,
        color=INK,
    )
    save_bundle(fig, "figure3_main_performance_and_triage")


# ---------------------------------------------------------------------------
# Figure 4: complex evidence states
# ---------------------------------------------------------------------------

def subset_row(index: dict[tuple[str, str, str], dict[str, str]], split: str, subset: str, method: str) -> dict[str, str]:
    return index[(split, subset, method)]


def panel_f4a(ax: plt.Axes, stats: dict[tuple[str, str], dict[str, str]]) -> None:
    style_axis(ax, grid_axis="x")
    add_panel_label(ax, "a", "Prevalence of complex evidence states")
    subsets = [
        "no_direct_indication",
        "mechanism_supported_without_direct",
        "high_safety_conflict",
        "support_conflict_coexist",
        "planner_early_stop",
    ]
    labels = [
        "No direct indication",
        "Mechanism support\nwithout direct indication",
        "High safety conflict",
        "Support-conflict\ncoexistence",
        "Planner early STOP",
    ]
    y = np.arange(len(subsets))
    height = 0.32
    for split, offset, color in [("Drug-disjoint", -height / 2, BLUE_2), ("Temporal-submit", height / 2, PURPLE)]:
        values = [as_int(stats[(split, subset)]["n"]) for subset in subsets]
        ax.barh(y + offset, values, height=height, color=color, alpha=0.88, label=split, zorder=3)
        for pos, value in zip(y + offset, values):
            ax.text(value + 7, pos, str(value), va="center", fontsize=5.7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pairs")
    ax.set_xlim(0, 460)
    ax.legend(loc="lower right")


def grouped_subset_f1(ax: plt.Axes, index: dict[tuple[str, str, str], dict[str, str]], subset: str, label: str, panel: str) -> None:
    style_axis(ax)
    add_panel_label(ax, panel, label)
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    x = np.arange(len(methods))
    for split, marker, offset in [("Drug-disjoint", "o", -0.055), ("Temporal-submit", "^", 0.055)]:
        values = [as_float(subset_row(index, split, subset, method)["f1"]) for method in methods]
        ax.plot(x + offset, values, color=SPLIT_COLORS[split], lw=0.95, alpha=0.50)
        ax.scatter(
            x + offset,
            values,
            marker=marker,
            s=23,
            color=[METHOD_COLORS[method] if method != "TreatAgent" else SPLIT_COLORS[split] for method in methods],
            edgecolor=INK,
            linewidth=0.45,
            zorder=3,
        )
        annotate_value(ax, x[-1] + offset, values[-1], values[-1], color=SPLIT_COLORS[split])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 0.82)


def panel_f4d(ax: plt.Axes, diagnostics: dict[tuple[str, str, str], dict[str, str]]) -> None:
    style_axis(ax)
    add_panel_label(ax, "d", "DTI contribution under mechanism support")
    splits = ["Drug-disjoint", "Temporal-submit"]
    x = np.arange(len(splits))
    full = [as_float(diagnostics[(split, "mechanism_supported_without_direct", "TreatAgent")]["f1"]) for split in splits]
    ablated = [as_float(diagnostics[(split, "mechanism_supported_without_direct", "w/o DTI")]["f1"]) for split in splits]
    for i in range(len(splits)):
        ax.plot([x[i] - 0.10, x[i] + 0.10], [ablated[i], full[i]], color=GREY, lw=1.0, zorder=2)
        ax.scatter(x[i] - 0.10, ablated[i], color=GREY, s=28, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.scatter(x[i] + 0.10, full[i], color=BLUE, s=28, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.text(x[i], max(full[i], ablated[i]) + 0.035, f"+{full[i] - ablated[i]:.2f}", ha="center", fontsize=6.0, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(["Drug-disjoint", "Temporal-submit"])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.50, 0.78)
    ax.scatter([], [], color=BLUE, s=24, label="TreatAgent")
    ax.scatter([], [], color=GREY, s=24, label="w/o DTI")
    ax.legend(loc="lower left")


def render_figure4() -> None:
    subset_stats = index_rows(read_csv("subset_stats.csv"), "split", "subset")
    subset_metrics = index_rows(read_csv("subset_metrics.csv"), "split", "subset", "method")
    diagnostics = index_rows(read_csv("subset_ablation_diagnostics.csv"), "split", "subset", "setting")
    fig = plt.figure(figsize=(7.20, 5.00))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.46, left=0.13, right=0.98, top=0.92, bottom=0.11)
    panel_f4a(fig.add_subplot(gs[0, 0]), subset_stats)
    grouped_subset_f1(fig.add_subplot(gs[0, 1]), subset_metrics, "no_direct_indication", "Performance without direct indications", "b")
    grouped_subset_f1(
        fig.add_subplot(gs[1, 0]),
        subset_metrics,
        "mechanism_supported_without_direct",
        "Performance with mechanism support only",
        "c",
    )
    panel_f4d(fig.add_subplot(gs[1, 1]), diagnostics)
    fig.suptitle(
        "Figure 4 | TreatAgent retains value in complex evidence states without direct indication support",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=9.0,
        color=INK,
    )
    save_bundle(fig, "figure4_complex_evidence_states")


# ---------------------------------------------------------------------------
# Figure 5: planner efficiency and auditable case studies
# ---------------------------------------------------------------------------

def load_full_rows(split: str) -> list[dict[str, Any]]:
    name = "results_multiagent_dd_test.json" if split == "Drug-disjoint" else "results_multiagent_ts_test.json"
    return read_json(RESULT_DIR / "baselines" / name)["results"]


def panel_f5a(ax: plt.Axes) -> None:
    style_axis(ax)
    add_panel_label(ax, "a", "Expert-call distribution")
    splits = ["Drug-disjoint", "Temporal-submit"]
    rows = {split: load_full_rows(split) for split in splits}
    distributions = {split: Counter(len(row.get("expert_outputs") or {}) for row in split_rows) for split, split_rows in rows.items()}
    x = np.arange(len(splits))
    bottoms = np.zeros(len(splits))
    colors = {2: BLUE_PALE, 3: BLUE_3, 4: BLUE_2, 5: BLUE}
    for calls in [2, 3, 4, 5]:
        values = np.array([distributions[split][calls] for split in splits])
        ax.bar(x, values, bottom=bottoms, color=colors[calls], edgecolor=WHITE, linewidth=0.6, label=f"{calls} calls", zorder=3)
        for i, value in enumerate(values):
            if value >= 20:
                ax.text(i, bottoms[i] + value / 2, str(value), ha="center", va="center", fontsize=5.7, color=WHITE if calls >= 4 else INK)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Pairs")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        fontsize=5.2,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    ax.text(0, 492, "mean 4.59", ha="center", fontsize=5.8, color=BLUE)
    ax.text(1, 492, "mean 4.60", ha="center", fontsize=5.8, color=PURPLE)
    ax.set_ylim(0, 520)


def panel_f5b(ax: plt.Axes) -> None:
    style_axis(ax)
    add_panel_label(ax, "b", "Planner performance on early-STOP subset")
    diagnostics = index_rows(read_csv("subset_ablation_diagnostics.csv"), "split", "subset", "setting")
    splits = ["Drug-disjoint", "Temporal-submit"]
    x = np.arange(len(splits))
    full = [as_float(diagnostics[(split, "planner_early_stop", "TreatAgent")]["f1"]) for split in splits]
    no_planner = [as_float(diagnostics[(split, "planner_early_stop", "w/o Planner")]["f1"]) for split in splits]
    for i in range(len(splits)):
        ax.plot([x[i] - 0.10, x[i] + 0.10], [no_planner[i], full[i]], color=GREY, lw=1.0)
        ax.scatter(x[i] - 0.10, no_planner[i], color=GREY, s=28, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.scatter(x[i] + 0.10, full[i], color=BLUE, s=28, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.text(x[i], full[i] + 0.032, f"+{full[i] - no_planner[i]:.2f}", ha="center", fontsize=6.0, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.70, 0.99)
    ax.scatter([], [], color=BLUE, s=24, label="TreatAgent")
    ax.scatter([], [], color=GREY, s=24, label="w/o Planner")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.06, "~8% fewer calls overall", transform=ax.transAxes, ha="right", fontsize=5.8, color=BLUE)


def wrap_text(text: str, max_chars: int = 42) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if current and len(proposal) > max_chars:
            lines.append(current)
            current = word
        else:
            current = proposal
    if current:
        lines.append(current)
    return lines


def draw_text_lines(ax: plt.Axes, x: float, y: float, lines: list[str], *, size: float = 5.7, color: str = INK, dy: float = 0.045, weight: str = "normal") -> None:
    for i, line in enumerate(lines):
        ax.text(x, y - i * dy, line, transform=ax.transAxes, ha="left", va="top", fontsize=size, color=color, fontweight=weight)


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, *, face: str, edge: str, lw: float = 0.8) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc=face,
            ec=edge,
            lw=lw,
        )
    )


def truncate(text: str, length: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= length else value[: length - 3].rstrip() + "..."


def panel_case(ax: plt.Axes, case: dict[str, str], panel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, panel, case["case_type"], y=1.01)
    rounded_box(ax, 0.00, 0.03, 0.99, 0.88, face=WHITE, edge=BLUE_3, lw=0.9)

    decision = "prioritize" if as_int(case["prediction"]) == 1 else "deprioritize"
    decision_face = GREEN_PALE if decision == "prioritize" else RED_PALE
    decision_edge = GREEN if decision == "prioritize" else RED
    score = as_float(case["judge_score"])

    ax.text(0.04, 0.86, truncate(case["drug"], 32), transform=ax.transAxes, fontsize=6.7, fontweight="bold", color=BLUE)
    ax.text(0.04, 0.805, f"for {truncate(case['disease'], 38)}", transform=ax.transAxes, fontsize=6.1, color=INK)
    ax.text(0.04, 0.745, f"Planner: {truncate(case['planner_trace'], 54)}", transform=ax.transAxes, fontsize=5.3, color=MUTED)

    rounded_box(ax, 0.04, 0.48, 0.42, 0.21, face=GREEN_PALE, edge=GREEN)
    ax.text(0.065, 0.65, "Support", transform=ax.transAxes, fontsize=5.9, fontweight="bold", color=GREEN)
    card_text = {
        "PAIR-000828": {
            "support": "Direct indication overlap; disease-relevant mechanistic context.",
            "conflict": "No major safety conflict identified.",
            "judge": "Direct indication, mechanistic support and clinical feasibility justify prioritization.",
        },
        "PAIR-001194": {
            "support": "Indication context and CASR target convergence across sources.",
            "conflict": "No major safety conflict identified.",
            "judge": "Cross-source CASR grounding supports expert follow-up despite moderate clinical feasibility.",
        },
        "PAIR-001938": {
            "support": "No strong direct or disease-grounded support.",
            "conflict": "DILI, hERG and AMES safety risks accumulate.",
            "judge": "Missing grounding and multiple safety conflicts preclude prioritization.",
        },
    }.get(case["sample_id"], {})

    support = card_text.get("support") or truncate(case.get("support_claims") or "No strong direct support", 70)
    draw_text_lines(ax, 0.065, 0.60, wrap_text(support, 24)[:3], size=4.85, color=INK, dy=0.042)

    rounded_box(ax, 0.53, 0.48, 0.42, 0.21, face=RED_PALE, edge=RED)
    ax.text(0.555, 0.65, "Conflict / missing", transform=ax.transAxes, fontsize=5.9, fontweight="bold", color=RED)
    conflict = card_text.get("conflict") or truncate(case.get("conflict_claims") or "No major safety conflict identified", 70)
    draw_text_lines(ax, 0.555, 0.60, wrap_text(conflict, 24)[:3], size=4.85, color=INK, dy=0.042)

    rounded_box(ax, 0.04, 0.15, 0.58, 0.22, face=BLUE_PALE, edge=BLUE_2)
    ax.text(0.065, 0.335, "Constrained LLM judge", transform=ax.transAxes, fontsize=5.9, fontweight="bold", color=BLUE)
    reason = card_text.get("judge") or truncate(case.get("judge_reason") or "", 96)
    draw_text_lines(ax, 0.065, 0.285, wrap_text(reason, 31)[:4], size=4.75, color=INK, dy=0.039)

    rounded_box(ax, 0.68, 0.15, 0.27, 0.22, face=decision_face, edge=decision_edge)
    ax.text(0.815, 0.315, f"{score:.2f}", transform=ax.transAxes, ha="center", fontsize=12.0, fontweight="bold", color=decision_edge)
    ax.text(0.815, 0.238, decision, transform=ax.transAxes, ha="center", fontsize=5.3, color=decision_edge, fontweight="bold")
    ax.text(0.815, 0.185, f"label / pred: {case['label']} / {case['prediction']}", transform=ax.transAxes, ha="center", fontsize=4.9, color=MUTED)


def render_figure5() -> None:
    cases = read_csv("case_study_candidates.csv")[:3]
    fig = plt.figure(figsize=(7.20, 5.85))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.85, 1.48], hspace=0.58, wspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.06)
    panel_f5a(fig.add_subplot(gs[0, 0:3]))
    panel_f5b(fig.add_subplot(gs[0, 3:6]))
    for panel, case, cols in zip(["c", "d", "e"], cases, [(0, 2), (2, 4), (4, 6)]):
        panel_case(fig.add_subplot(gs[1, cols[0] : cols[1]]), case, panel)
    fig.suptitle(
        "Figure 5 | Planner efficiency and auditable evidence traces for candidate-level review",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=9.0,
        color=INK,
    )
    save_bundle(fig, "figure5_planner_efficiency_and_case_studies")


# ---------------------------------------------------------------------------
# Source-data traceability and rendering
# ---------------------------------------------------------------------------

def export_source_data() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "main_results.csv",
        "subset_stats.csv",
        "subset_metrics.csv",
        "subset_ablation_diagnostics.csv",
        "case_study_candidates.csv",
        "planner_efficiency.csv",
    ]:
        rows = read_csv(name)
        write_csv(SOURCE_DIR / name, rows)

    call_rows: list[dict[str, Any]] = []
    for split in ["Drug-disjoint", "Temporal-submit"]:
        distribution = Counter(len(row.get("expert_outputs") or {}) for row in load_full_rows(split))
        for calls in sorted(distribution):
            call_rows.append({"split": split, "expert_calls": calls, "pairs": distribution[calls]})
    write_csv(SOURCE_DIR / "figure5_expert_call_distribution.csv", call_rows)

    auditability = [
        {"capability": "Source provenance", "Direct": "No", "CoT": "No", "RAG": "Partial", "TreatAgent": "Yes"},
        {"capability": "Typed evidence tuples", "Direct": "No", "CoT": "No", "RAG": "No", "TreatAgent": "Yes"},
        {"capability": "Support / conflict separation", "Direct": "No", "CoT": "No", "RAG": "No", "TreatAgent": "Yes"},
        {"capability": "Missing-evidence tracking", "Direct": "No", "CoT": "No", "RAG": "No", "TreatAgent": "Yes"},
        {"capability": "Planner trajectory", "Direct": "No", "CoT": "No", "RAG": "No", "TreatAgent": "Yes"},
        {"capability": "Structured explanation", "Direct": "No", "CoT": "Partial", "RAG": "Partial", "TreatAgent": "Yes"},
    ]
    write_csv(SOURCE_DIR / "figure3_auditability_matrix.csv", auditability)


def write_contract() -> None:
    contract = """# Figure 2-5 contract and QA notes

## Core conclusions

- Figure 2: the benchmark is derived from clinical-trial records through explicit cleaning, canonicalization, conflict removal and leakage-controlled split construction.
- Figure 3: TreatAgent improves candidate-level treatment triage primarily by reducing missed positives across both leakage-controlled evaluations.
- Figure 4: TreatAgent remains useful when direct indication evidence is absent and mechanism evidence must be integrated across sources.
- Figure 5: the Planner supports budget-aware evidence acquisition and the final output can be audited at case level.

## Archetypes

- Figure 2: schematic-led composite.
- Figure 3: four-panel quantitative grid with non-redundant triage-oriented hero metrics. Complete metrics remain in Table 1; the auditability capability matrix belongs in the method figure or supplementary material.
- Figure 4: quantitative grid focused on complex evidence states.
- Figure 5: asymmetric mixed-modality figure with quantitative planner panels and case-level evidence cards.

## Export contract

- Backend: Python / matplotlib only.
- Primary output: editable SVG (`svg.fonttype = none`).
- Secondary outputs: PDF, 600 dpi TIFF and 300 dpi PNG preview.
- Font: Arial with sans-serif fallbacks.
- Source data: copied into `source_data/`.

## Reviewer-risk notes

- Baseline confidence intervals are not plotted because the frozen API evaluations are single runs.
- Direct, CoT and RAG expose binary predictions; their AUROC, AUPRC, Brier and ECE are not shown in Figure 3 as directly comparable continuous-score metrics.
- Auditability is presented as a capability matrix, not as an invented numerical score.
- ADMET is interpreted as a safety-review layer with an observable recall trade-off, not as a universal F1 improvement.
- Subset thresholds are fixed and interpretable but should be described as post-hoc behavior analysis, not preregistered confirmatory endpoints.
"""
    (OUT_DIR / "figure2_5_contract_and_qa.md").write_text(contract, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_source_data()
    write_contract()
    render_figure3()
    render_figure4()
    render_figure5()
    print(f"Wrote Figure 3-5 bundles to {OUT_DIR}")


if __name__ == "__main__":
    main()
