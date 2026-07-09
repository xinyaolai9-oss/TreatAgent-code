#!/usr/bin/env python3
"""Render Figure 3 with bootstrap confidence intervals.

Inputs:
- results/final_results/main_results.csv
- results/final_results/bootstrap_ci.csv
- figure/final_results/source_data/figure3_auditability_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "final_results"
SOURCE_DIR = ROOT / "figure" / "final_results" / "source_data"
OUT_DIR = ROOT / "figure" / "final_results"

METHODS = ["Direct", "CoT", "RAG", "TreatAgent"]
METHOD_LABELS = {"Direct": "Direct LLM", "CoT": "CoT", "RAG": "RAG", "TreatAgent": "TreatAgent"}
SPLITS = ["Drug-disjoint", "Temporal-submit"]
RADAR_METRICS = ["accuracy", "f1", "precision", "recall", "auroc", "auprc"]
RADAR_LABELS = ["Accuracy", "F1", "Precision", "Recall", "AUROC", "AUPRC"]

COLORS = {
    "Direct": "#164C9C",
    "CoT": "#4E914E",
    "RAG": "#F2A000",
    "TreatAgent": "#B30000",
}
LIGHT = {
    "Direct": "#EAF2FC",
    "CoT": "#EEF7EA",
    "RAG": "#FFF0C7",
    "TreatAgent": "#FCEAEA",
}
INK = "#111111"
MUTED = "#6E7781"
GRID = "#E6E6E6"
DD_LS = (0, (4, 2.2))
TS_LS = "solid"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_panel_label(ax, label: str, title: str, x: float = -0.08, y: float = 1.11) -> None:
    ax.text(x, y, label.lower(), transform=ax.transAxes, ha="left", va="center",
            fontsize=18, color=INK, fontweight="bold")
    ax.text(x + 0.13, y, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=13.5, color=INK)


def metric_row(df: pd.DataFrame, split: str, method: str) -> pd.Series:
    return df[(df["split"] == split) & (df["setting"] == method)].iloc[0]


def ci_row(ci: pd.DataFrame, split: str, method: str, metric: str) -> pd.Series:
    return ci[(ci["split"] == split) & (ci["method"] == method) & (ci["metric"] == metric)].iloc[0]


def split_bar_style(split: str, method: str) -> dict:
    """Visual split encoding for panels a/e.

    Drug-disjoint uses dark dashed borders with light method-colored fills.
    Temporal-submit uses dark solid borders with white fills.
    """
    if split == "Drug-disjoint":
        return {"facecolor": LIGHT[method], "edgecolor": COLORS[method], "linestyle": DD_LS}
    return {"facecolor": "white", "edgecolor": COLORS[method], "linestyle": TS_LS}


def draw_f1(ax, df: pd.DataFrame, ci: pd.DataFrame) -> None:
    add_panel_label(ax, "A", "F1 comparison")
    x = np.arange(len(METHODS))
    width = 0.34
    offsets = {"Drug-disjoint": -width / 2, "Temporal-submit": width / 2}

    for split in SPLITS:
        vals, lower, upper = [], [], []
        for method in METHODS:
            obs = metric_row(df, split, method)["f1"]
            row = ci_row(ci, split, method, "f1")
            vals.append(obs)
            lower.append(obs - row["ci_lower"])
            upper.append(row["ci_upper"] - obs)
        for i, method in enumerate(METHODS):
            style = split_bar_style(split, method)
            ax.bar(
                x[i] + offsets[split], vals[i], width=width,
                facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                linewidth=1.8, linestyle=style["linestyle"],
                label=split if i == 0 else None,
                zorder=3,
            )
            ax.errorbar(
                x[i] + offsets[split], vals[i],
                yerr=np.array([[lower[i]], [upper[i]]]),
                color="#333333", lw=1.0, capsize=2.5, zorder=5,
            )
            if method == "TreatAgent":
                ax.text(x[i] + offsets[split], vals[i] + upper[i] + 0.025,
                        f"{vals[i]:.3f}", ha="center", va="bottom",
                        fontsize=9.0, color=COLORS[method], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.84)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))


def _closed(vals: list[float]) -> np.ndarray:
    return np.array(vals + [vals[0]], dtype=float)


def draw_radar(ax, df: pd.DataFrame, ci: pd.DataFrame, split: str, label: str, title: str) -> None:
    add_panel_label(ax, label, title, x=-0.20, y=1.15)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    n = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = np.array(angles + [angles[0]])

    for method in METHODS:
        row = metric_row(df, split, method)
        vals = _closed([float(row[m]) for m in RADAR_METRICS])
        lo = _closed([float(ci_row(ci, split, method, m)["ci_lower"]) for m in RADAR_METRICS])
        hi = _closed([float(ci_row(ci, split, method, m)["ci_upper"]) for m in RADAR_METRICS])
        ax.fill_between(
            angles_closed, lo, hi,
            color=COLORS[method],
            alpha=0.11 if method == "TreatAgent" else 0.075,
            lw=0,
            zorder=1,
        )
        ax.plot(angles_closed, vals, color=COLORS[method],
                lw=2.2 if method == "TreatAgent" else 1.4,
                label=METHOD_LABELS[method],
                zorder=3 if method == "TreatAgent" else 2)

    ax.set_xticks(angles)
    ax.set_xticklabels(RADAR_LABELS)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=MUTED)
    ax.grid(color="#DDDDDD", lw=0.8)


def draw_pr(ax, df: pd.DataFrame) -> None:
    add_panel_label(ax, "D", "Precision–recall profile")
    markers = {"Drug-disjoint": "o", "Temporal-submit": "^"}
    for split in SPLITS:
        for method in METHODS:
            row = metric_row(df, split, method)
            ax.scatter(row["recall"], row["precision"], s=54,
                       marker=markers[split], facecolor=LIGHT[method],
                       edgecolor=COLORS[method], lw=1.3, zorder=4)
            dx = 0.010 if method != "TreatAgent" else -0.150
            dy = 0.010 if method != "TreatAgent" else 0.015
            ax.text(row["recall"] + dx, row["precision"] + dy,
                    f"{METHOD_LABELS[method]} ({'DD' if split == 'Drug-disjoint' else 'TS'})",
                    fontsize=8.6, color=INK)

    ax.add_patch(patches.Rectangle((0.64, 0.45), 0.20, 0.36,
                                   fc="#F2F2F2", ec="none", zorder=0))
    ax.text(0.635, 0.785, "higher-recall\ntriage region", ha="left", va="top",
            fontsize=9.0, color=INK)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0.45, 0.80)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_missed(ax, df: pd.DataFrame, ci: pd.DataFrame) -> None:
    add_panel_label(ax, "E", "Missed-positive rate")
    x = np.arange(len(METHODS))
    width = 0.34
    offsets = {"Drug-disjoint": -width / 2, "Temporal-submit": width / 2}
    for split in SPLITS:
        for i, method in enumerate(METHODS):
            row = ci_row(ci, split, method, "missed_positive_rate")
            obs = row["observed"]
            style = split_bar_style(split, method)
            ax.bar(x[i] + offsets[split], obs, width=width,
                   facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                   linewidth=1.8, linestyle=style["linestyle"],
                   label=split if i == 0 else None, zorder=3)
            ax.errorbar(x[i] + offsets[split], obs,
                        yerr=np.array([[obs - row["ci_lower"]], [row["ci_upper"] - obs]]),
                        color="#333333", lw=1.0, capsize=2.5, zorder=5)
            if method == "TreatAgent":
                ax.text(x[i] + offsets[split], row["ci_upper"] + 0.030,
                        f"{obs:.3f}", ha="center", va="bottom",
                        fontsize=9.0, color=COLORS[method], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Missed positive ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def draw_auditability(ax, audit: pd.DataFrame) -> None:
    add_panel_label(ax, "F", "Evidence auditability", x=-0.03, y=1.10)
    ax.axis("off")
    methods = ["Direct", "CoT", "RAG", "TreatAgent"]
    cols = ["Capability"] + methods
    rows = audit["capability"].tolist()
    table_data = []
    symbol = {"Yes": "Yes", "No": "No", "Partial": "Partial"}
    for _, r in audit.iterrows():
        table_data.append([r["capability"]] + [symbol.get(str(r[m]), str(r[m])) for m in methods])
    tbl = ax.table(
        cellText=table_data,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.42, 0.125, 0.125, 0.125, 0.19],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1.0, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.set_facecolor("#F7F7F7")
            text = cell.get_text().get_text()
            if text == "Yes":
                cell.set_text_props(color="#4E914E", fontsize=9.0, fontweight="bold")
            elif text == "No":
                cell.set_text_props(color="#B30000", fontsize=9.0, fontweight="bold")
            elif text == "Partial":
                cell.set_text_props(color="#E3B04B", fontsize=8.2, fontweight="bold")
    ax.text(0.18, -0.10, "No", transform=ax.transAxes, ha="center", va="center", color="#B30000", fontsize=10, fontweight="bold")
    ax.text(0.43, -0.10, "Partial", transform=ax.transAxes, ha="center", va="center", color="#E3B04B", fontsize=10, fontweight="bold")
    ax.text(0.72, -0.10, "Yes", transform=ax.transAxes, ha="center", va="center", color="#4E914E", fontsize=10, fontweight="bold")


def save(fig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "figure3_main_performance_and_triage"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    setup_style()
    df = pd.read_csv(RESULT_DIR / "main_results.csv")
    ci = pd.read_csv(RESULT_DIR / "bootstrap_ci.csv")
    audit = pd.read_csv(SOURCE_DIR / "figure3_auditability_matrix.csv")

    # Keep figure source data self-contained.
    ci.to_csv(SOURCE_DIR / "bootstrap_ci.csv", index=False)

    fig = plt.figure(figsize=(15.4, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.12, 1.0, 1.23],
                          hspace=0.43, wspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_f1(ax_a, df, ci)
    draw_radar(ax_b, df, ci, "Drug-disjoint", "B", "Drug-disjoint metric profile")
    draw_radar(ax_c, df, ci, "Temporal-submit", "C", "Temporal-submit metric profile")
    draw_pr(ax_d, df)
    draw_missed(ax_e, df, ci)
    draw_auditability(ax_f, audit)

    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
