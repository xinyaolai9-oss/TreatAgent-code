#!/usr/bin/env python3
"""Render Nature Health-style benchmark construction/statistics SVGs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figure"

BLUE = "#164C9C"
BLUE2 = "#2B77C4"
LIGHT_BLUE = "#EAF4FF"
PALE_BLUE = "#F6FBFF"
TEAL = "#44A6A6"
GREEN = "#6DAA57"
PURPLE = "#6E4AA8"
ORANGE = "#E69F2E"
RED = "#D84B45"
INK = "#111827"
MUTED = "#5B677A"
GRID = "#D8E6F7"


def load_json(path: str):
    with open(ROOT / path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
    })


def add_panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.02, 0.96, label,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=13, color="white", fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.38", fc=BLUE, ec=BLUE, lw=0),
        zorder=10,
    )
    ax.text(
        0.07, 0.96, title,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=14, color=BLUE, fontweight="bold",
        zorder=10,
    )


def panel_frame(ax, edge: str = BLUE) -> None:
    ax.set_facecolor(PALE_BLUE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        fill=False,
        ec=edge,
        lw=1.35,
        linestyle=(0, (4, 4)),
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(rect)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)


def draw_pipeline(ax, stats: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_frame(ax)
    add_panel_label(ax, "A", "Benchmark construction pipeline")

    steps = [
        ("Raw clinical-trial\nrecords", f"{stats['extraction']['raw_rows']:,}\nrows"),
        ("Single-disease\nsingle-SMILES\nextraction", f"{stats['extraction']['kept_rows']:,}\nrows"),
        ("SMILES canonicalization\nand disease normalization", f"{stats['dedup']['valid_canonical_smiles_rows']:,}\nvalid"),
        ("Pair-level\ndeduplication", f"{stats['dedup']['unique_pair_candidates']:,}\ncandidates"),
        ("Conflict\nremoval", f"{stats['dedup']['conflict_pairs_removed']:,}\nremoved"),
        ("Final pair-level\nbenchmark", f"{stats['dedup']['pair_rows']:,}\npairs"),
    ]
    xs = [0.09, 0.25, 0.41, 0.57, 0.73, 0.89]
    y = 0.50
    w, h = 0.12, 0.48
    for idx, ((title, value), x) in enumerate(zip(steps, xs), 1):
        ec = RED if idx == 5 else BLUE
        fc = "#FFF8F7" if idx == 5 else "#FFFFFF"
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc=fc, ec=ec, lw=1.3, linestyle=(0, (4, 4)),
        )
        ax.add_patch(box)
        ax.text(x, y + 0.17, str(idx), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.35", fc=ec, ec=ec, lw=0))
        ax.text(x, y + 0.045, title, ha="center", va="center", color=INK, fontsize=9.2, fontweight="bold")
        ax.text(x, y - 0.165, value, ha="center", va="center", color=ec, fontsize=12, fontweight="bold")
        if idx < len(xs):
            ax.annotate(
                "", xy=(xs[idx] - w / 2 - 0.01, y), xytext=(x + w / 2 + 0.01, y),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=13),
            )
    ax.text(0.5, 0.10, "Clinical-trial-derived drug-disease pairs defined by canonical SMILES and normalized diseases; conflicting labels removed before split construction",
            ha="center", va="center", color=MUTED, fontsize=10, style="italic")


def draw_label_distribution(ax, stats: dict) -> None:
    panel_frame(ax)
    add_panel_label(ax, "B", "Label distribution by split")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    splits = ["train", "val", "test"]
    groups = ["drug_disjoint", "temporal_submit"]
    labels = []
    neg, pos = [], []
    for group in groups:
        source = stats["splits"]["drug_disjoint"] if group == "drug_disjoint" else load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]
        for split in splits:
            labels.append(f"{group.replace('_', '-')}\n{split}")
            counts = source[split]["label_counts"]
            neg.append(int(counts.get("0", 0)))
            pos.append(int(counts.get("1", 0)))
    x = list(range(len(labels)))
    ax.bar(x, neg, color="#DCE9F8", edgecolor=BLUE, linewidth=0.7, label="Negative")
    ax.bar(x, pos, bottom=neg, color=BLUE2, edgecolor=BLUE, linewidth=0.7, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Pairs")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.tick_params(length=0)


def draw_unique_coverage(ax, stats: dict) -> None:
    panel_frame(ax, edge=PURPLE)
    add_panel_label(ax, "C", "Coverage of unique drugs and diseases")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    splits = ["train", "val", "test"]
    dd = stats["splits"]["drug_disjoint"]
    ts = load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]
    labels = [f"drug-disjoint\n{s}" for s in splits] + [f"temporal-submit\n{s}" for s in splits]
    drugs = [dd[s]["unique_drugs"] for s in splits] + [ts[s]["unique_drugs"] for s in splits]
    diseases = [dd[s]["unique_diseases"] for s in splits] + [ts[s]["unique_diseases"] for s in splits]
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], drugs, width, color=TEAL, edgecolor="#237A7A", linewidth=0.7, label="Unique drugs")
    ax.bar([i + width / 2 for i in x], diseases, width, color=PURPLE, edgecolor="#55358D", linewidth=0.7, label="Unique diseases")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.5, 0.08, "Drug-disjoint split: zero drug overlap between train, validation, and test",
            transform=ax.transAxes, ha="center", va="center", color=PURPLE, fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.tick_params(length=0)


def draw_phase_composition(ax) -> None:
    panel_frame(ax, edge=GREEN)
    add_panel_label(ax, "D", "Clinical trial phase composition")
    rows = load_json("data/benchmark/processed/pair_level_dataset_with_submit_dates.json")
    phase_order = ["early phase 1", "phase 1", "phase 1/phase 2", "phase 2", "phase 2/phase 3", "phase 3", "phase 4", "n/a"]
    phase_counts = {phase: {"0": 0, "1": 0} for phase in phase_order}
    for row in rows:
        label = str(row.get("label", "0"))
        phases = row.get("phases") or ["n/a"]
        if not phases:
            phases = ["n/a"]
        for phase in phases:
            if phase not in phase_counts:
                phase = "n/a"
            phase_counts[phase][label] += 1
    neg = [phase_counts[phase]["0"] for phase in phase_order]
    pos = [phase_counts[phase]["1"] for phase in phase_order]
    x = list(range(len(phase_order)))
    ax.bar(x, neg, color="#DCE9F8", edgecolor=BLUE, linewidth=0.7, label="Negative")
    ax.bar(x, pos, bottom=neg, color=BLUE2, edgecolor=BLUE, linewidth=0.7, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels([phase.replace("phase", "Phase").replace("n/a", "Unknown") for phase in phase_order], rotation=45, ha="right")
    ax.set_ylabel("Pairs")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.5, 0.06, "Benchmark pairs span multiple clinical development phases, supporting clinical diversity.",
            transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.tick_params(length=0)


def split_for_date(date: datetime, temporal_stats: dict) -> str:
    for split in ["train", "val", "test"]:
        lo = datetime.fromisoformat(temporal_stats[split]["date_min"])
        hi = datetime.fromisoformat(temporal_stats[split]["date_max"])
        if lo <= date <= hi:
            return split
    return "other"


def draw_temporal_distribution(ax) -> None:
    panel_frame(ax)
    add_panel_label(ax, "E", "Temporal-submit distribution")
    temporal_stats = load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]
    rows = load_json("data/benchmark/processed/pair_level_dataset_with_submit_dates.json")
    counts = defaultdict(Counter)
    for row in rows:
        raw = row.get("pair_date")
        if not raw:
            continue
        dt = datetime.fromisoformat(raw)
        split = split_for_date(dt, temporal_stats)
        counts[dt.year][split] += 1
    years = list(range(min(counts), max(counts) + 1))
    bottom = [0] * len(years)
    colors = {"train": BLUE2, "val": GREEN, "test": PURPLE}
    for split in ["train", "val", "test"]:
        values = [counts[y][split] for y in years]
        ax.bar(years, values, bottom=bottom, color=colors[split], edgecolor="white", linewidth=0.4, label=split)
        bottom = [b + v for b, v in zip(bottom, values)]
    val_start = datetime.fromisoformat(temporal_stats["val"]["date_min"]).year
    test_start = datetime.fromisoformat(temporal_stats["test"]["date_min"]).year
    ax.axvline(val_start - 0.5, color=ORANGE, lw=1.5, ls="--")
    ax.axvline(test_start - 0.5, color=ORANGE, lw=1.5, ls="--")
    ax.text(val_start - 0.6, max(bottom) * 0.9, "val cutoff", rotation=90, va="top", ha="right", color=ORANGE, fontsize=8)
    ax.text(test_start - 0.6, max(bottom) * 0.9, "test cutoff", rotation=90, va="top", ha="right", color=ORANGE, fontsize=8)
    ax.set_xlabel("Study first submit year")
    ax.set_ylabel("Pairs")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def render_all() -> None:
    setup_style()
    stats = load_json("data/benchmark/processed/benchmark_stats.json")

    fig, ax = plt.subplots(figsize=(14.5, 4.1))
    draw_pipeline(ax, stats)
    save(fig, "figure2A_benchmark_pipeline.svg")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_label_distribution(ax, stats)
    save(fig, "figure2B_label_distribution.svg")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_unique_coverage(ax, stats)
    save(fig, "figure2C_unique_coverage.svg")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_phase_composition(ax)
    save(fig, "figure2D_phase_composition.svg")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_temporal_distribution(ax)
    save(fig, "figure2E_temporal_distribution.svg")

    fig = plt.figure(figsize=(15.5, 13.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.0, 1.0], hspace=0.24, wspace=0.14)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_pipeline(ax_a, stats)
    draw_label_distribution(ax_b, stats)
    draw_unique_coverage(ax_c, stats)
    draw_phase_composition(ax_d)
    draw_temporal_distribution(ax_e)
    fig.suptitle("Figure 2. Benchmark construction, split characterization, and clinical trial diversity",
                 color=INK, fontsize=17, fontweight="bold", y=0.995)
    save(fig, "figure2_benchmark_full_5panel.svg")


if __name__ == "__main__":
    render_all()
