from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure"


BLUE = "#2F6FB2"
BLUE_DARK = "#174A82"
BLUE_LIGHT = "#DDECFB"
GREEN = "#5D9B4E"
GREEN_LIGHT = "#E9F4E5"
RED = "#C94D4A"
RED_LIGHT = "#F7DEDC"
AMBER = "#D99A25"
AMBER_LIGHT = "#FFF2D6"
PURPLE = "#6C5B9A"
PURPLE_LIGHT = "#EEE9F8"
GRAY = "#6E7781"
GRAY_LIGHT = "#EEF1F4"
TEXT = "#1F2933"
BG = "#FBFCFD"


METHOD_COLORS = {
    "Direct": "#A7B1BD",
    "CoT": "#8896A6",
    "RAG": "#6F8FB8",
    "Raw Fusion": AMBER,
    "LLM Synth": PURPLE,
    "TreatAgent-ARG": BLUE,
}


def configure_style() -> None:
    for font_path in (
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            break

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.edgecolor": "#A7B8CC",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def load_json(path: str) -> Dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT / f"{name}.png", format="png", dpi=600, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def clean_ax(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B4C4D7")
    ax.spines["bottom"].set_color("#B4C4D7")
    if grid:
        ax.grid(axis="y", color="#E7EDF5", linewidth=0.8)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.26", facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=1.0),
    )


def set_title(ax: plt.Axes, title: str, subtitle: Optional[str] = None) -> None:
    ax.set_title(title, loc="left", color=BLUE_DARK, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=GRAY)


def main_metrics() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "Drug-disjoint": [
            {"method": "Direct", "f1": 0.0560, "auroc": None},
            {"method": "CoT", "f1": 0.4893, "auroc": None},
            {"method": "RAG", "f1": 0.3333, "auroc": None},
            {"method": "Raw Fusion", "f1": 0.7296, "auroc": 0.7317},
            {"method": "LLM Synth", "f1": 0.5735, "auroc": 0.6389},
            {"method": "TreatAgent-ARG", "f1": 0.7396, "auroc": 0.7120},
        ],
        "Temporal-submit": [
            {"method": "Direct", "f1": 0.0329, "auroc": None},
            {"method": "CoT", "f1": 0.4634, "auroc": None},
            {"method": "RAG", "f1": 0.2603, "auroc": None},
            {"method": "Raw Fusion", "f1": 0.7465, "auroc": 0.7661},
            {"method": "LLM Synth", "f1": 0.5353, "auroc": None},
            {"method": "TreatAgent-ARG", "f1": 0.7399, "auroc": 0.7717},
        ],
    }


def plot_main_bars(ax: plt.Axes, split: str, label: Optional[str] = None) -> None:
    data = main_metrics()[split]
    names = [d["method"] for d in data]
    x = np.arange(len(names))
    width = 0.36
    f1 = [d["f1"] for d in data]
    auroc = [np.nan if d["auroc"] is None else d["auroc"] for d in data]

    ax.bar(x - width / 2, f1, width, label="F1", color=BLUE, edgecolor=BLUE_DARK, linewidth=0.8)
    ax.bar(x + width / 2, np.nan_to_num(auroc, nan=0.0), width, label="AUROC", color=GREEN, edgecolor="#3F7B35", linewidth=0.8)
    for i, v in enumerate(auroc):
        if np.isnan(v):
            ax.text(i + width / 2, 0.035, "-", ha="center", va="center", fontsize=9, color=GRAY)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    set_title(ax, split, "Main predictive metrics")
    clean_ax(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    if label:
        panel_label(ax, label)


def plot_avg_calls(ax: plt.Axes, label: Optional[str] = None) -> None:
    splits = ["Drug-disjoint", "Temporal-submit"]
    all_calls = [5.0, 5.0]
    online = [3.8218, 3.6505]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, all_calls, width, color=GRAY_LIGHT, edgecolor="#A7B1BD", label="All experts")
    ax.bar(x + width / 2, online, width, color=BLUE, edgecolor=BLUE_DARK, label="Online planner")
    for i, v in enumerate(online):
        ax.text(i + width / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=BLUE_DARK)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Average expert calls")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    set_title(ax, "Cost-aware evidence acquisition", "Average expert calls per query")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper right")
    if label:
        panel_label(ax, label)


def plot_call_reduction(ax: plt.Axes, label: Optional[str] = None) -> None:
    vals = [23.56, 26.99]
    splits = ["Drug-disjoint", "Temporal-submit"]
    colors = [BLUE, GREEN]
    ax.bar(np.arange(2), vals, color=colors, edgecolor=[BLUE_DARK, "#3F7B35"], linewidth=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.0, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", color=colors[i])
    ax.set_ylim(0, 35)
    ax.set_ylabel("Call reduction (%)")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(splits)
    set_title(ax, "Planner saves expert calls", "Compared with all-expert acquisition")
    clean_ax(ax)
    if label:
        panel_label(ax, label)


def bootstrap_ci() -> Dict[str, Dict[str, Any]]:
    drug = load_json("results/bootstrap/final_drug_disjoint_online/bootstrap_final_drug_disjoint_online.json")
    temp = load_json("results/bootstrap/final_temporal_submit_online/bootstrap_final_temporal_submit_online.json")
    return {
        "Drug-disjoint": {
            "arg": drug["metrics_ci"]["TreatAgent-ARG"],
            "raw": drug["metrics_ci"]["Raw-Feature-Fusion"],
            "delta": drug["paired_delta_vs_primary"]["TreatAgent-ARG_minus_Raw-Feature-Fusion"],
        },
        "Temporal-submit": {
            "arg": temp["metrics_ci"]["TreatAgent-ARG"],
            "raw": temp["metrics_ci"]["Raw-Feature-Fusion"],
            "delta": temp["paired_delta_vs_primary"]["TreatAgent-ARG_minus_Raw-Feature-Fusion"],
        },
    }


def plot_bootstrap_forest(ax: plt.Axes, label: Optional[str] = None) -> None:
    ci = bootstrap_ci()
    rows = [
        ("Drug F1", ci["Drug-disjoint"]["arg"]["f1"], BLUE),
        ("Drug Raw F1", ci["Drug-disjoint"]["raw"]["f1"], AMBER),
        ("Temp F1", ci["Temporal-submit"]["arg"]["f1"], BLUE),
        ("Temp Raw F1", ci["Temporal-submit"]["raw"]["f1"], AMBER),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (name, stat, color) in zip(y, rows):
        point = stat["point"]
        lo = stat["ci95_low"]
        hi = stat["ci95_high"]
        ax.errorbar(point, yi, xerr=[[point - lo], [hi - point]], fmt="o", color=color, ecolor=color, capsize=3, lw=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.66, 0.80)
    ax.set_xlabel("F1 with 95% bootstrap CI")
    set_title(ax, "Bootstrap uncertainty", "TreatAgent-ARG vs Raw Feature Fusion")
    clean_ax(ax, grid=False)
    ax.axvline(0.74, color="#D8E2EF", lw=1.0, zorder=0)
    if label:
        panel_label(ax, label)


def planner_reports() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        load_json("results/planner_analysis/final_drug_disjoint_online/planner_final_drug_disjoint_online.json"),
        load_json("results/planner_analysis/final_temporal_submit_online/planner_final_temporal_submit_online.json"),
    )


def plot_trajectory(ax: plt.Axes, label: Optional[str] = None) -> None:
    drug, temp = planner_reports()
    traj_names = [
        "DrugKB -> DiseaseKB -> DTI -> ADMET -> Clinical",
        "Clinical",
        "DrugKB -> DiseaseKB -> DTI -> Clinical",
        "DiseaseKB -> DTI -> Clinical",
    ]
    def counts(report: Dict[str, Any]) -> List[int]:
        mapping = {d["trajectory"]: d["count"] for d in report["top_trajectories"]}
        return [mapping.get(t, 0) for t in traj_names]

    x = np.arange(len(traj_names))
    width = 0.36
    ax.bar(x - width / 2, counts(drug), width, color=BLUE, edgecolor=BLUE_DARK, label="Drug-disjoint")
    ax.bar(x + width / 2, counts(temp), width, color=GREEN, edgecolor="#3F7B35", label="Temporal")
    short = ["All core\nexperts", "Clinical\nonly", "No ADMET", "Disease+DTI\n+Clinical"]
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Number of queries")
    set_title(ax, "Planner trajectory distribution", "Top evidence acquisition routes")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper right")
    if label:
        panel_label(ax, label)


def read_reliability_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["avg_confidence"] and r["positive_rate"] and r["count"] != "0":
                rows.append({k: float(v) for k, v in r.items() if v != ""})
    return rows


def plot_reliability(ax: plt.Axes, label: Optional[str] = None) -> None:
    paths = {
        "Drug ARG": ("results/calibration/final_drug_disjoint_online/TreatAgent-ARG/calibration_final_drug_disjoint_online_TreatAgent-ARG_reliability_bins.csv", BLUE),
        "Drug Raw": ("results/calibration/final_drug_disjoint_online/Raw-Feature-Fusion/calibration_final_drug_disjoint_online_Raw-Feature-Fusion_reliability_bins.csv", AMBER),
        "Temp ARG": ("results/calibration/final_temporal_submit_online/TreatAgent-ARG/calibration_final_temporal_submit_online_TreatAgent-ARG_reliability_bins.csv", GREEN),
        "Temp Raw": ("results/calibration/final_temporal_submit_online/Raw-Feature-Fusion/calibration_final_temporal_submit_online_Raw-Feature-Fusion_reliability_bins.csv", PURPLE),
    }
    ax.plot([0, 1], [0, 1], "--", color="#B8C4D2", lw=1.0, label="Perfect")
    for name, (path, color) in paths.items():
        rows = read_reliability_csv(path)
        ax.plot([r["avg_confidence"] for r in rows], [r["positive_rate"] for r in rows], marker="o", lw=1.5, color=color, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    set_title(ax, "Reliability curves", "Calibration by probability bin")
    clean_ax(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    if label:
        panel_label(ax, label)


def plot_brier_ece(ax: plt.Axes, label: Optional[str] = None) -> None:
    cal_drug = load_json("results/calibration/final_drug_disjoint_online/calibration_final_drug_disjoint_online.json")
    cal_temp = load_json("results/calibration/final_temporal_submit_online/calibration_final_temporal_submit_online.json")
    categories = ["Drug\nARG", "Drug\nRaw", "Temp\nARG", "Temp\nRaw"]
    brier = [
        cal_drug["methods"]["TreatAgent-ARG"]["metrics"]["brier"],
        cal_drug["methods"]["Raw-Feature-Fusion"]["metrics"]["brier"],
        cal_temp["methods"]["TreatAgent-ARG"]["metrics"]["brier"],
        cal_temp["methods"]["Raw-Feature-Fusion"]["metrics"]["brier"],
    ]
    ece = [
        cal_drug["methods"]["TreatAgent-ARG"]["metrics"]["ece"],
        cal_drug["methods"]["Raw-Feature-Fusion"]["metrics"]["ece"],
        cal_temp["methods"]["TreatAgent-ARG"]["metrics"]["ece"],
        cal_temp["methods"]["Raw-Feature-Fusion"]["metrics"]["ece"],
    ]
    x = np.arange(len(categories))
    width = 0.36
    ax.bar(x - width / 2, brier, width, color=BLUE_LIGHT, edgecolor=BLUE, label="Brier")
    ax.bar(x + width / 2, ece, width, color=RED_LIGHT, edgecolor=RED, label="ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.28)
    ax.set_ylabel("Lower is better")
    set_title(ax, "Calibration-related metrics", "Brier score and ECE")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper right")
    if label:
        panel_label(ax, label)


def plot_component_ablation(ax: plt.Axes, label: Optional[str] = None) -> None:
    variants = ["Full", "w/o Clinical\nfeasibility", "w/o EvidenceGraph\nstructure"]
    drug = [0.7418, 0.7290, 0.7189]
    temp = [0.7496, 0.7085, 0.7256]
    x = np.arange(len(variants))
    width = 0.36
    ax.bar(x - width / 2, drug, width, color=BLUE, edgecolor=BLUE_DARK, label="Drug-disjoint")
    ax.bar(x + width / 2, temp, width, color=GREEN, edgecolor="#3F7B35", label="Temporal")
    ax.set_ylim(0.66, 0.77)
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    set_title(ax, "Key component ablation", "Clinical feasibility and graph structure")
    clean_ax(ax)
    ax.legend(frameon=False, loc="lower left")
    if label:
        panel_label(ax, label)


def plot_source_ablation(ax: plt.Axes, label: Optional[str] = None) -> None:
    variants = ["Full", "w/o Clinical", "w/o ADMET", "w/o DrugKB", "w/o DTI"]
    drug = [0.7183, 0.5392, 0.7159, 0.7219, 0.7180]
    temp = [0.7721, 0.5009, 0.7734, 0.7767, 0.7720]
    x = np.arange(len(variants))
    width = 0.36
    ax.bar(x - width / 2, drug, width, color=BLUE, edgecolor=BLUE_DARK, label="Drug-disjoint")
    ax.bar(x + width / 2, temp, width, color=GREEN, edgecolor="#3F7B35", label="Temporal")
    ax.set_ylim(0.46, 0.82)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    set_title(ax, "Source ablation summary", "Clinical evidence is the strongest source")
    clean_ax(ax)
    ax.legend(frameon=False, loc="lower right")
    if label:
        panel_label(ax, label)


def plot_hard_subset(ax: plt.Axes, label: Optional[str] = None) -> None:
    drug = load_json("results/hard_subset_analysis/final_drug_disjoint_online/hard_subset_final_drug_disjoint_online.json")
    temp = load_json("results/hard_subset_analysis/final_temporal_submit_online/hard_subset_final_temporal_submit_online.json")
    subsets = ["no_direct_indication", "low_clinical_prior", "admet_risk", "mechanism_only_support"]
    labels = ["No direct\nindication", "Low clinical\nprior", "ADMET\nrisk", "Mechanism-\nonly"]
    arg_vals = []
    raw_vals = []
    for s in subsets:
        arg_vals.append(np.mean([drug["methods"]["TreatAgent-ARG"][s]["f1"], temp["methods"]["TreatAgent-ARG"][s]["f1"]]))
        raw_vals.append(np.mean([drug["methods"]["Raw-Feature-Fusion"][s]["f1"], temp["methods"]["Raw-Feature-Fusion"][s]["f1"]]))
    x = np.arange(len(subsets))
    width = 0.36
    ax.bar(x - width / 2, arg_vals, width, color=BLUE, edgecolor=BLUE_DARK, label="TreatAgent-ARG")
    ax.bar(x + width / 2, raw_vals, width, color=AMBER, edgecolor="#B97B17", label="Raw Fusion")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean F1 across splits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    set_title(ax, "Hard subset analysis", "Evidence-state-specific behavior")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper right")
    if label:
        panel_label(ax, label)


def individual_figures() -> None:
    panels3 = [
        ("figure3a_drug_disjoint_main_metrics", lambda ax: plot_main_bars(ax, "Drug-disjoint", "A")),
        ("figure3b_temporal_main_metrics", lambda ax: plot_main_bars(ax, "Temporal-submit", "B")),
        ("figure3c_average_expert_calls", lambda ax: plot_avg_calls(ax, "C")),
        ("figure3d_expert_call_reduction", lambda ax: plot_call_reduction(ax, "D")),
        ("figure3e_bootstrap_ci", lambda ax: plot_bootstrap_forest(ax, "E")),
        ("figure3f_planner_trajectories", lambda ax: plot_trajectory(ax, "F")),
    ]
    for name, fn in panels3:
        fig, ax = plt.subplots(figsize=(4.2, 3.1))
        fn(ax)
        save(fig, name)

    panels5 = [
        ("figure5a_reliability_curves", lambda ax: plot_reliability(ax, "A")),
        ("figure5b_brier_ece", lambda ax: plot_brier_ece(ax, "B")),
        ("figure5c_component_ablation", lambda ax: plot_component_ablation(ax, "C")),
        ("figure5d_source_ablation", lambda ax: plot_source_ablation(ax, "D")),
        ("figure5e_hard_subsets", lambda ax: plot_hard_subset(ax, "E")),
    ]
    for name, fn in panels5:
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        fn(ax)
        save(fig, name)


def combined_figure3() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    plot_main_bars(axes[0, 0], "Drug-disjoint", "A")
    plot_main_bars(axes[0, 1], "Temporal-submit", "B")
    plot_avg_calls(axes[0, 2], "C")
    plot_call_reduction(axes[1, 0], "D")
    plot_bootstrap_forest(axes[1, 1], "E")
    plot_trajectory(axes[1, 2], "F")
    fig.suptitle("Figure 3. Main performance and cost-aware planning", x=0.02, ha="left", fontsize=16, color=BLUE_DARK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure3_combined")


def combined_figure5() -> None:
    fig = plt.figure(figsize=(14.2, 7.8))
    gs = fig.add_gridspec(2, 6, left=0.055, right=0.985, bottom=0.075, top=0.88, wspace=0.75, hspace=0.58)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])
    plot_reliability(ax1, "A")
    plot_brier_ece(ax2, "B")
    plot_component_ablation(ax3, "C")
    plot_source_ablation(ax4, "D")
    plot_hard_subset(ax5, "E")
    fig.suptitle("Figure 5. Reliability, ablation, and hard-subset behavior", x=0.02, ha="left", fontsize=16, color=BLUE_DARK, fontweight="bold")
    save(fig, "figure5_combined")


def main() -> None:
    configure_style()
    individual_figures()
    combined_figure3()
    combined_figure5()
    print(f"Wrote SVG and 600dpi PNG files to {OUT}")


if __name__ == "__main__":
    main()
