#!/usr/bin/env python3
"""Render Figure 4: complementary TreatAgent module contributions.

The figure separates global ablation results from context-specific diagnostics:
- Planner and EvidenceGraph are evaluated with the metrics they directly affect.
- All expert removals are reported transparently using global delta F1.
- DTI, ADMET, Clinical and Planner are then examined in interpretable hard subsets.

The script reads frozen CSV files only. It never invokes a model or external API.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "final_results"
OUT_DIR = ROOT / "figure" / "final_results"
SOURCE_DIR = OUT_DIR / "source_data"

# Figure 3-consistent palette.
BLUE = "#0B3B8C"
BLUE_PALE = "#D6E3F5"
GREEN = "#4E9A45"
GREEN_PALE = "#E9F4E4"
ORANGE = "#F2A000"
ORANGE_PALE = "#FFF0C7"
RED = "#9F1111"
RED_PALE = "#F7E0E0"
PURPLE = "#7356A8"
PURPLE_PALE = "#EEE8F7"
GREY = "#7A828B"
GREY_2 = "#A9B0B7"
GREY_PALE = "#F0F0F0"
INK = "#111111"
MUTED = "#6E7781"
GRID = "#E6E6E6"
WHITE = "#FFFFFF"

SPLITS = ["Drug-disjoint", "Temporal-submit"]
SPLIT_HATCH = {"Drug-disjoint": None, "Temporal-submit": "//"}
SPLIT_ALPHA = {"Drug-disjoint": 1.0, "Temporal-submit": 0.72}
SPLIT_SHORT = {"Drug-disjoint": "DD", "Temporal-submit": "TS"}

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
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    return float(value)


def index_rows(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(WHITE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", colors=INK, length=2.5, width=0.7, pad=2)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str, title: str, *, x_title: float = 0.0) -> None:
    ax.text(
        -0.10,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=8.8,
        fontweight="bold",
    )
    ax.text(
        x_title,
        1.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=7.8,
    )


def split_handles() -> list[Patch]:
    return [
        Patch(facecolor=GREY_PALE, edgecolor=GREY, label="Drug-disjoint"),
        Patch(facecolor=GREY_PALE, edgecolor=GREY, hatch="//", label="Temporal-submit"),
    ]


def pp(value: float) -> float:
    return value * 100.0


def fpr(row: dict[str, str]) -> float:
    """Recover the false-positive rate from subset-level rounded metrics."""
    positive = int(row["positive"])
    negative = int(row["negative"])
    recall = as_float(row["recall"])
    precision = as_float(row["precision"])
    true_positive = round(recall * positive)
    predicted_positive = round(true_positive / precision) if precision > 0 else 0
    false_positive = max(0, predicted_positive - true_positive)
    return false_positive / negative if negative else 0.0


def annotate_barh(ax: plt.Axes, value: float, y: float, *, suffix: str = "") -> None:
    offset = 0.32 if value >= 0 else -0.32
    ax.text(
        value + offset,
        y,
        f"{value:+.1f}{suffix}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=5.7,
        color=INK,
    )


def grouped_bars(
    ax: plt.Axes,
    values: dict[str, list[float]],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str,
    ylim: tuple[float, float],
    lower_is_better: bool = False,
) -> None:
    x = np.arange(len(labels))
    width = 0.34
    for split_index, split in enumerate(SPLITS):
        xpos = x + (split_index - 0.5) * width
        bars = ax.bar(
            xpos,
            values[split],
            width=width,
            facecolor=colors,
            edgecolor=colors,
            linewidth=1.15,
            hatch=SPLIT_HATCH[split],
            alpha=SPLIT_ALPHA[split],
            zorder=3,
        )
        for bar, value in zip(bars, values[split]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    style_axis(ax)
    if lower_is_better:
        ax.text(0.02, 0.96, "lower is better", transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=5.6)


def panel_a(ax: plt.Axes, components: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    """Metric-specific global contribution of core architecture modules."""
    style_axis(ax, grid_axis="x")
    panel_label(ax, "a", "Core architecture contributions")
    rows = [
        ("Planner", "F1", "f1", "w/o Planner"),
        ("EvidenceGraph", "Accuracy", "accuracy", "w/o EvidenceGraph"),
        ("EvidenceGraph", "Precision", "precision", "w/o EvidenceGraph"),
    ]
    labels = [f"{module}\n{metric}" for module, metric, _, _ in rows]
    y = np.arange(len(rows))
    height = 0.31
    exported: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        offset = (split_index - 0.5) * height
        values = []
        for module, metric_label, metric_key, ablation in rows:
            full = as_float(components[(split, "TreatAgent")][metric_key])
            removed = as_float(components[(split, ablation)][metric_key])
            value = pp(full - removed)
            values.append(value)
            exported.append(
                {
                    "panel": "a",
                    "split": split,
                    "comparison": f"{module}: TreatAgent vs {ablation}",
                    "metric": metric_label,
                    "value": value,
                }
            )
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            facecolor=[BLUE_PALE, GREEN_PALE, GREEN_PALE],
            edgecolor=[BLUE, GREEN, GREEN],
            linewidth=1.05,
            hatch=SPLIT_HATCH[split],
            alpha=SPLIT_ALPHA[split],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            annotate_barh(ax, value, bar.get_y() + bar.get_height() / 2, suffix=" pp")
    ax.axvline(0, color=INK, linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Performance gain after restoring module (percentage points)")
    ax.set_xlim(0, 9.5)
    return exported


def panel_b(ax: plt.Axes, sources: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    """Transparent global source ablation; mixed effects remain visible."""
    style_axis(ax, grid_axis="x")
    panel_label(ax, "b", "Global expert contribution")
    experts = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]
    labels = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]
    colors = [BLUE, PURPLE, ORANGE, RED, GREEN]
    fills = [BLUE_PALE, PURPLE_PALE, ORANGE_PALE, RED_PALE, GREEN_PALE]
    y = np.arange(len(experts))
    height = 0.31
    exported: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        offset = (split_index - 0.5) * height
        values = []
        for expert in experts:
            full = as_float(sources[(split, "TreatAgent")]["f1"])
            removed = as_float(sources[(split, f"w/o {expert}")]["f1"])
            value = pp(full - removed)
            values.append(value)
            exported.append(
                {
                    "panel": "b",
                    "split": split,
                    "comparison": f"TreatAgent vs w/o {expert}",
                    "metric": "F1",
                    "value": value,
                }
            )
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            facecolor=fills,
            edgecolor=colors,
            linewidth=1.05,
            hatch=SPLIT_HATCH[split],
            alpha=SPLIT_ALPHA[split],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            annotate_barh(ax, value, bar.get_y() + bar.get_height() / 2, suffix=" pp")
    ax.axvline(0, color=INK, linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("F1 gain after restoring expert (percentage points)")
    ax.set_xlim(-7.4, 10.5)
    return exported


def panel_c(ax: plt.Axes, diagnostics: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    panel_label(ax, "c", "Mechanism rescue without direct indication")
    subset = "mechanism_supported_without_direct"
    settings = ["TreatAgent", "w/o DTI", "w/o DrugKB"]
    labels = ["Full", "w/o DTI", "w/o DrugKB"]
    colors = [RED, ORANGE, BLUE]
    values = {
        split: [as_float(diagnostics[(split, subset, setting)]["f1"]) for setting in settings]
        for split in SPLITS
    }
    grouped_bars(ax, values, labels, colors, ylabel="F1 score", ylim=(0.50, 0.78))
    ax.text(0.98, 0.05, "n=245 / 271", transform=ax.transAxes, ha="right", color=MUTED, fontsize=5.5)
    return [
        {"panel": "c", "split": split, "comparison": setting, "metric": "F1", "value": value}
        for split in SPLITS
        for setting, value in zip(settings, values[split])
    ]


def panel_d(ax: plt.Axes, diagnostics: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    panel_label(ax, "d", "Safety guardrail under high conflict")
    subset = "high_safety_conflict"
    settings = ["TreatAgent", "w/o ADMET", "w/o EvidenceGraph"]
    labels = ["Full", "w/o ADMET", "w/o EG"]
    colors = [RED, ORANGE, GREY]
    values = {
        split: [fpr(diagnostics[(split, subset, setting)]) for setting in settings]
        for split in SPLITS
    }
    grouped_bars(ax, values, labels, colors, ylabel="False-positive rate", ylim=(0.0, 1.05), lower_is_better=True)
    ax.text(0.98, 0.05, "n=186 / 172", transform=ax.transAxes, ha="right", color=MUTED, fontsize=5.5)
    return [
        {"panel": "d", "split": split, "comparison": setting, "metric": "False-positive rate", "value": value}
        for split in SPLITS
        for setting, value in zip(settings, values[split])
    ]


def panel_e(ax: plt.Axes, diagnostics: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    panel_label(ax, "e", "Feasibility guardrail with low clinical prior")
    subset = "low_clinical_prior"
    settings = ["TreatAgent", "w/o Clinical"]
    labels = ["Full", "w/o Clinical"]
    colors = [RED, GREY]
    values = {
        split: [fpr(diagnostics[(split, subset, setting)]) for setting in settings]
        for split in SPLITS
    }
    grouped_bars(ax, values, labels, colors, ylabel="False-positive rate", ylim=(0.0, 0.72), lower_is_better=True)
    ax.text(0.98, 0.05, "n=117 / 138", transform=ax.transAxes, ha="right", color=MUTED, fontsize=5.5)
    return [
        {"panel": "e", "split": split, "comparison": setting, "metric": "False-positive rate", "value": value}
        for split in SPLITS
        for setting, value in zip(settings, values[split])
    ]


def panel_f(ax: plt.Axes, diagnostics: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    panel_label(ax, "f", "Planner benefit on early-STOP subset")
    subset = "planner_early_stop"
    settings = ["TreatAgent", "w/o Planner"]
    labels = ["Full", "w/o Planner"]
    colors = [RED, GREY]
    values = {
        split: [as_float(diagnostics[(split, subset, setting)]["f1"]) for setting in settings]
        for split in SPLITS
    }
    grouped_bars(ax, values, labels, colors, ylabel="F1 score", ylim=(0.70, 1.01))
    ax.text(0.98, 0.05, "n=103 / 97", transform=ax.transAxes, ha="right", color=MUTED, fontsize=5.5)
    return [
        {"panel": "f", "split": split, "comparison": setting, "metric": "F1", "value": value}
        for split in SPLITS
        for setting, value in zip(settings, values[split])
    ]


def main() -> None:
    components = index_rows(read_csv("component_ablation.csv"), "split", "setting")
    sources = index_rows(read_csv("source_ablation.csv"), "split", "setting")
    diagnostics = index_rows(read_csv("subset_ablation_diagnostics.csv"), "split", "subset", "setting")

    fig = plt.figure(figsize=(10.2, 6.5))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.91,
        wspace=0.55,
        hspace=0.56,
    )

    source_rows: list[dict[str, Any]] = []
    source_rows += panel_a(fig.add_subplot(grid[0, 0]), components)
    source_rows += panel_b(fig.add_subplot(grid[0, 1]), sources)
    source_rows += panel_c(fig.add_subplot(grid[0, 2]), diagnostics)
    source_rows += panel_d(fig.add_subplot(grid[1, 0]), diagnostics)
    source_rows += panel_e(fig.add_subplot(grid[1, 1]), diagnostics)
    source_rows += panel_f(fig.add_subplot(grid[1, 2]), diagnostics)

    fig.suptitle(
        "Figure 4 | TreatAgent modules contribute through complementary evidence-reasoning functions",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=10.0,
        color=INK,
    )
    fig.legend(
        handles=split_handles(),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.973),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.85,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "figure4_module_ablation.svg", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(OUT_DIR / "figure4_module_ablation.pdf", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(OUT_DIR / "figure4_module_ablation.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    write_csv(SOURCE_DIR / "figure4_module_ablation_summary.csv", source_rows)
    print(f"Wrote Figure 4 module ablation bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
