#!/usr/bin/env python3
"""Render Nature Health-style benchmark construction/statistics SVGs.

Design changes versus the earlier draft:
- Keep the benchmark construction pipeline as an infographic-style panel.
- Remove dashed frames around statistical plots (Panels B-E).
- Use Figure 1-derived colors with dark borders and very light fills for all bar charts.
- Keep B-E statistical panels clean without dashed outer frames.
- Draw temporal-submit distribution as equal-width categorical bins with exact cutoff labels, avoiding visually uneven date-width bars.
- Use black regular-weight titles/text, except circular panel letters.
- Use vertically stacked legends with larger labels.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    """Find a project root that contains data/benchmark/processed.

    The original script assumed it lived two levels below the project root.
    This helper keeps that behaviour but also works when the script is launched
    from the project root or after being copied to another folder.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = [Path.cwd(), here.parent, *here.parents]
    for root in candidates:
        if (root / "data" / "benchmark" / "processed").exists():
            return root
    # Fallback to the original project layout assumption.
    return here.parents[2]


ROOT = find_project_root()
OUT_DIR = Path(os.getenv("TREATAGENT_FIGURE_OUT_DIR", str(ROOT / "figure" / "final_results")))


# -----------------------------------------------------------------------------
# Figure 1-consistent palette
# -----------------------------------------------------------------------------

BLUE = "#0B3B8C"          # deep navy / main titles and arrows
BLUE2 = "#2F6FB2"         # DrugKB / train edge
LIGHT_BLUE = "#D6E3F5"    # Figure 1 pale blue
VERY_LIGHT_BLUE = "#EAF2FC"
PALE_BLUE = "#F7FBFF"     # very pale panel fill for infographic panel A
TEAL = "#5AA9A6"          # optional teal accent
GREEN = "#4E9A45"         # support / validation edge
LIGHT_GREEN = "#E9F4E4"
PURPLE = "#7356A8"        # clinical / test edge
LIGHT_PURPLE = "#EEE8F7"
ORANGE = "#F2A000"        # DTI / cutoff / highlight
LIGHT_ORANGE = "#FFF0C7"
RED = "#9F1111"           # DiseaseKB / negative / conflict edge
LIGHT_RED = "#F7E0E0"
VERY_LIGHT_RED = "#FCEAEA"
GREY = "#A9A9A9"
LIGHT_GREY = "#F2F2F2"
INK = "#111111"
MUTED = "#6E7781"
GRID = "#E6E6E6"
AXIS = "#111111"

# Bar-chart rule requested by user:
#   dark Figure-1-derived border + very light fill.
NEG_FILL = "#FDEDEA"
POS_FILL = "#EEF7EA"
DRUG_FILL = "#EAF4FF"
DISEASE_FILL = "#FBE8E6"
TRAIN_FILL = "#EAF4FF"
VAL_FILL = "#EEF7EA"
TEST_FILL = "#F1EEF8"

NEG_EDGE = RED
POS_EDGE = GREEN
DRUG_EDGE = BLUE2
DISEASE_EDGE = RED
TRAIN_EDGE = BLUE2
VAL_EDGE = GREEN
TEST_EDGE = PURPLE
BAR_LW = 1.15
MOLECULE_DIR = ROOT / "figure" / "final_results" / "molecules" / "drug_disjoint"

# -----------------------------------------------------------------------------
# IO / style utilities
# -----------------------------------------------------------------------------

def load_json(path: str):
    with open(ROOT / path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_panel_label(ax, label: str, title: str, *, y: float = 0.985) -> None:
    # Nature-style panel labels are compact lowercase letters near the top-left.
    ax.text(
        0.01, y, label.lower(),
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=13.0, color=INK, fontweight="bold",
        zorder=20,
    )
    ax.text(
        0.055, y, title,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=13.5, color=INK, fontweight="normal",
        zorder=20,
    )


def add_panel_label_compact(ax, label: str, title: str, *, y: float = 1.08) -> None:
    ax.text(
        0.00, y, label.lower(),
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=12.2, color=INK, fontweight="bold",
        zorder=20,
    )
    ax.text(
        0.075, y, title,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=11.1, color=INK, fontweight="normal",
        zorder=20,
    )


def panel_frame(ax, edge: str = BLUE) -> None:
    """Infographic frame used only for Panel A."""
    ax.set_facecolor(PALE_BLUE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = patches.FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.94,
        transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        fill=False,
        ec=edge,
        lw=1.15,
        linestyle=(0, (4, 4)),
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(rect)


def style_stat_axis(ax) -> None:
    """Clean Nature-style statistical panel without dashed outer frame."""
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=INK, pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, format="svg", bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / name.replace(".svg", ".pdf"), bbox_inches="tight", transparent=False)
    fig.savefig(OUT_DIR / name.replace(".svg", ".tiff"), dpi=600, bbox_inches="tight", transparent=False)
    # PNG preview is useful for quick inspection in slides/notes.
    fig.savefig(OUT_DIR / name.replace(".svg", ".png"), dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def split_source(stats: dict, group: str) -> dict:
    if group == "drug_disjoint":
        return stats["splits"]["drug_disjoint"]
    return load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]


def short_split_labels(groups: Iterable[str], splits: Iterable[str]) -> list[str]:
    prefix = {"drug_disjoint": "DD", "temporal_submit": "TS"}
    return [f"{prefix[g]}-{s}" for g in groups for s in splits]


def _blend(hex_color: str, alpha: float = 0.86) -> str:
    """Blend a color toward white."""
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    out = [round(c * (1 - alpha) + 255 * alpha) for c in rgb]
    return "#" + "".join(f"{c:02X}" for c in out)


def draw_split_pie(ax, center: tuple[float, float], counts: dict, radius: float = 0.045) -> None:
    """Draw a compact positive/negative pie in axes coordinates."""
    neg = int(counts.get("0", 0))
    pos = int(counts.get("1", 0))
    total = max(pos + neg, 1)
    theta_pos = 360 * pos / total
    x, y = center
    ax.add_patch(patches.Wedge(center, radius, 90, 90 + theta_pos,
                               transform=ax.transAxes, fc=POS_FILL, ec=POS_EDGE, lw=1.05))
    ax.add_patch(patches.Wedge(center, radius, 90 + theta_pos, 450,
                               transform=ax.transAxes, fc=NEG_FILL, ec=NEG_EDGE, lw=1.05))
    ax.add_patch(patches.Circle(center, radius, transform=ax.transAxes, fc="none", ec="white", lw=0.6))
    ax.text(x, y - radius - 0.018, f"{pos:,}/{neg:,}", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.6, color=INK)


def draw_pie_legend(ax, x: float, y: float) -> None:
    ax.add_patch(patches.Rectangle((x, y - 0.012), 0.018, 0.018,
                                   transform=ax.transAxes, fc=POS_FILL, ec=POS_EDGE, lw=0.9))
    ax.text(x + 0.024, y - 0.003, "Positive", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.6, color=INK)
    ax.add_patch(patches.Rectangle((x + 0.13, y - 0.012), 0.018, 0.018,
                                   transform=ax.transAxes, fc=NEG_FILL, ec=NEG_EDGE, lw=0.9))
    ax.text(x + 0.154, y - 0.003, "Negative", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.6, color=INK)


def load_molecule_examples() -> dict[str, list[Path]]:
    """Return representative molecule PNG paths by split."""
    meta = MOLECULE_DIR / "molecule_examples_metadata.csv"
    examples: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if not meta.exists():
        return examples
    import csv
    with meta.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            if split not in examples:
                continue
            png = ROOT / row.get("png", "")
            if png.exists():
                examples[split].append(png)
    return examples


def draw_molecule_thumbnail(ax, path: Path, extent: tuple[float, float, float, float],
                            tint: str) -> None:
    """Draw a molecule thumbnail with a faint border."""
    if not path.exists():
        x0, x1, y0, y1 = extent
        ax.add_patch(patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            transform=ax.transAxes,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            fc=_blend(tint, 0.94), ec=tint, lw=0.8,
        ))
        return
    img = mpimg.imread(path)
    ax.imshow(img, extent=extent, transform=ax.transAxes, zorder=2, aspect="auto")


def draw_clipboard_icon(ax, x: float, y: float, color: str = GREY) -> None:
    ax.add_patch(patches.FancyBboxPatch((x - 0.030, y - 0.055), 0.060, 0.095,
                                        boxstyle="round,pad=0.004,rounding_size=0.006",
                                        transform=ax.transAxes, fc="white", ec=color, lw=2.0))
    ax.add_patch(patches.FancyBboxPatch((x - 0.018, y + 0.035), 0.036, 0.018,
                                        boxstyle="round,pad=0.002,rounding_size=0.005",
                                        transform=ax.transAxes, fc="white", ec=color, lw=2.0))
    for i in range(4):
        yy = y + 0.020 - i * 0.020
        ax.plot([x - 0.017, x + 0.008], [yy, yy], transform=ax.transAxes, color=BLUE2, lw=1.8)
        ax.plot([x + 0.014, x + 0.020, x + 0.030], [yy - 0.002, yy - 0.010, yy + 0.006],
                transform=ax.transAxes, color=BLUE2, lw=1.5)


def draw_network_icon(ax, x: float, y: float) -> None:
    pts = [(x - 0.035, y - 0.020), (x, y + 0.020), (x + 0.035, y - 0.018), (x - 0.006, y - 0.050)]
    for p0, p1 in [(0, 1), (1, 2), (1, 3)]:
        ax.plot([pts[p0][0], pts[p1][0]], [pts[p0][1], pts[p1][1]],
                transform=ax.transAxes, color=BLUE2, lw=1.8)
    for px, py in pts:
        ax.add_patch(patches.Circle((px, py), 0.010, transform=ax.transAxes, fc=BLUE2, ec="white", lw=0.6))
    for k in range(8):
        import math
        ang = 2 * math.pi * k / 8
        px = x + 0.070 + 0.030 * math.cos(ang)
        py = y + 0.005 + 0.030 * math.sin(ang)
        ax.plot([x + 0.070, px], [y + 0.005, py], transform=ax.transAxes, color=RED, lw=1.4)
        ax.add_patch(patches.Circle((px, py), 0.006, transform=ax.transAxes, fc=RED, ec=RED, lw=0))
    ax.add_patch(patches.Circle((x + 0.070, y + 0.005), 0.018, transform=ax.transAxes, fc=RED, ec="white", lw=0.7))


def draw_cube_icon(ax, x: float, y: float) -> None:
    for dx, dy in [(-0.030, -0.020), (0.030, -0.020), (0.0, 0.035)]:
        ax.add_patch(patches.RegularPolygon((x + dx, y + dy), numVertices=6, radius=0.021,
                                            orientation=0.52, transform=ax.transAxes,
                                            fc=LIGHT_GREY, ec=GREY, lw=1.4))
    ax.plot([x - 0.010, x + 0.028], [y + 0.010, y + 0.032], transform=ax.transAxes, color=BLUE2, lw=2.0)
    ax.plot([x - 0.010, x - 0.035], [y + 0.010, y - 0.007], transform=ax.transAxes, color=BLUE2, lw=2.0)
    ax.plot([x + 0.010, x + 0.035], [y + 0.010, y - 0.007], transform=ax.transAxes, color=BLUE2, lw=2.0)


def draw_dedup_icon(ax, x: float, y: float) -> None:
    for dx, dy, col in [(-0.038, 0.024, BLUE), (-0.038, -0.025, RED), (0.048, 0.000, BLUE)]:
        ax.add_patch(patches.Rectangle((x + dx - 0.012, y + dy - 0.012), 0.024, 0.024,
                                       transform=ax.transAxes, fc=col, ec=col, lw=0))
    ax.plot([x - 0.004, x + 0.022], [y + 0.025, y + 0.025], transform=ax.transAxes, color=GREY, lw=2.4)
    ax.plot([x - 0.004, x + 0.022], [y - 0.025, y - 0.025], transform=ax.transAxes, color=GREY, lw=2.4)
    ax.plot([x + 0.022, x + 0.022], [y - 0.025, y + 0.025], transform=ax.transAxes, color=GREY, lw=2.4)


def draw_conflict_icon(ax, x: float, y: float) -> None:
    ax.add_patch(patches.Rectangle((x - 0.040, y - 0.045), 0.080, 0.040,
                                   transform=ax.transAxes, fc=GREY, ec=GREY, lw=0))
    ax.add_patch(patches.Polygon([[x - 0.030, y - 0.005], [x - 0.020, y + 0.030],
                                  [x + 0.030, y + 0.030], [x + 0.040, y - 0.005]],
                                 transform=ax.transAxes, fc=GREY, ec=GREY, lw=0))
    ax.add_patch(patches.Circle((x + 0.035, y + 0.030), 0.026, transform=ax.transAxes, fc=RED, ec=RED, lw=0))
    ax.plot([x + 0.025, x + 0.045], [y + 0.020, y + 0.040], transform=ax.transAxes, color="white", lw=2.3)
    ax.plot([x + 0.045, x + 0.025], [y + 0.020, y + 0.040], transform=ax.transAxes, color="white", lw=2.3)


def draw_calendar_icon(ax, x: float, y: float) -> None:
    ax.add_patch(patches.Rectangle((x - 0.040, y - 0.045), 0.080, 0.085,
                                   transform=ax.transAxes, fc="white", ec=GREY, lw=2.0))
    ax.add_patch(patches.Rectangle((x - 0.040, y + 0.018), 0.080, 0.020,
                                   transform=ax.transAxes, fc=LIGHT_GREY, ec=GREY, lw=0))
    for dx in [-0.022, 0.022]:
        ax.plot([x + dx, x + dx], [y + 0.035, y + 0.055], transform=ax.transAxes,
                color=GREY, lw=3, solid_capstyle="round")
    for i in range(3):
        for j in range(2):
            ax.add_patch(patches.Rectangle((x - 0.026 + i * 0.026, y - 0.025 + j * 0.022),
                                           0.013, 0.010, transform=ax.transAxes,
                                           fc=LIGHT_GREY, ec=LIGHT_GREY, lw=0))
    ax.add_patch(patches.Circle((x + 0.040, y - 0.030), 0.024, transform=ax.transAxes, fc=GREEN, ec=GREEN, lw=0))
    ax.plot([x + 0.029, x + 0.037, x + 0.054], [y - 0.030, y - 0.040, y - 0.018],
            transform=ax.transAxes, color="white", lw=2.2)


# -----------------------------------------------------------------------------
# Panel A: pipeline
# -----------------------------------------------------------------------------

def draw_pipeline(ax, stats: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "A", "Benchmark construction pipeline", y=0.94)

    steps = [
        ("Raw clinical-\ntrial records", f"{stats['extraction']['raw_rows']:,} rows", draw_clipboard_icon),
        ("Single-pair\nextraction", f"{stats['extraction']['kept_rows']:,} rows", draw_network_icon),
        ("Drug / disease\nnormalization", f"{stats['dedup']['valid_canonical_smiles_rows']:,} valid", draw_cube_icon),
        ("Pair-level\ndeduplication", f"{stats['dedup']['unique_pair_candidates']:,} pairs", draw_dedup_icon),
        ("Conflict\nremoval", f"{stats['dedup']['conflict_pairs_removed']:,} removed", draw_conflict_icon),
        ("Final pair-level\nbenchmark", f"{stats['dedup']['pair_rows']:,} pairs", draw_calendar_icon),
    ]
    xs = [0.085, 0.250, 0.415, 0.580, 0.745, 0.910]
    y = 0.47
    w, h = 0.135, 0.60

    for idx, ((title, value, icon_fn), x) in enumerate(zip(steps, xs), 1):
        is_conflict = idx == 5
        ec = RED if is_conflict else BLUE
        fc = VERY_LIGHT_RED if is_conflict else "#FFFFFF"
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            fc=fc, ec=ec, lw=1.15, linestyle=(0, (3, 4)),
        )
        ax.add_patch(box)
        ax.text(
            x - w * 0.34, y + h * 0.32, str(idx),
            ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.32", fc=ec, ec=ec, lw=0),
        )
        ax.text(
            x + 0.020, y + h * 0.31, title,
            ha="center", va="center", color=INK,
            fontsize=9.7, fontweight="normal", linespacing=1.05,
        )
        icon_fn(ax, x, y - 0.030)
        ax.text(
            x, y - h * 0.36, value,
            ha="center", va="center", color=ec,
            fontsize=10.5, fontweight="normal", linespacing=1.05,
        )
        if idx < len(xs):
            ax.annotate(
                "",
                xy=(xs[idx] - w / 2 - 0.018, y),
                xytext=(x + w / 2 + 0.010, y),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.25, mutation_scale=12),
            )


def _summary_text(source: dict, split: str) -> str:
    counts = source[split]["label_counts"]
    return (
        f"{source[split]['rows']:,} pairs\n"
        f"{source[split]['unique_drugs']:,} drugs / {source[split]['unique_diseases']:,} diseases\n"
        f"{int(counts.get('1', 0)):,} pos / {int(counts.get('0', 0)):,} neg"
    )


def draw_drug_disjoint_design(ax, stats: dict) -> None:
    """Visual definition of the drug-disjoint split."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "B", "Drug-disjoint split design", y=0.98)

    dd = split_source(stats, "drug_disjoint")
    molecule_examples = load_molecule_examples()
    xs = [0.19, 0.50, 0.81]
    names = ["Train", "Validation", "Test"]
    fills = [VERY_LIGHT_BLUE, LIGHT_ORANGE, VERY_LIGHT_RED]
    edges = [TRAIN_EDGE, VAL_EDGE, TEST_EDGE]

    for x, name, fill, edge, split in zip(xs, names, fills, edges, ["train", "val", "test"]):
        ax.add_patch(patches.FancyBboxPatch(
            (x - 0.085, 0.795), 0.170, 0.070,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            fc=fill, ec="none", lw=0,
        ))
        ax.text(x, 0.830, name, ha="center", va="center", fontsize=12.0, color=edge, fontweight="bold")

        examples = molecule_examples.get(split, [])[:3]
        extents = [
            (x - 0.155, x + 0.005, 0.470, 0.745),
            (x - 0.015, x + 0.145, 0.470, 0.745),
            (x - 0.090, x + 0.090, 0.315, 0.575),
        ]
        for path, extent in zip(examples, extents):
            draw_molecule_thumbnail(ax, path, extent, edge)

        draw_split_pie(ax, (x, 0.205), dd[split]["label_counts"], radius=0.041)
        ax.text(x, 0.082, f"{dd[split]['rows']:,} pairs", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.4, color=MUTED)

    overlap = dd.get("drug_overlap", {})
    ax.text(
        0.5, 0.030,
        f"Zero drug overlap: train-val {overlap.get('train_val', 0)}, train-test {overlap.get('train_test', 0)}, val-test {overlap.get('val_test', 0)}",
        ha="center", va="center", fontsize=9.4, color=BLUE, fontweight="bold",
    )


def draw_temporal_submit_design(ax) -> None:
    """Visual definition of the temporal-submit split."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "C", "Temporal-submit split design", y=0.98)

    temporal = load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]
    segments = [
        ("train", "Train", VERY_LIGHT_BLUE, TRAIN_EDGE, 0.10, 0.47),
        ("val", "Validation", LIGHT_ORANGE, ORANGE, 0.47, 0.65),
        ("test", "Test", VERY_LIGHT_RED, RED, 0.65, 0.92),
    ]
    label_centers = {"train": 0.27, "val": 0.56, "test": 0.78}
    for split, name, fill, edge, x0, x1 in segments:
        ax.add_patch(patches.FancyBboxPatch(
            (label_centers[split] - 0.085, 0.795), 0.170, 0.070,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            fc=fill, ec="none", lw=0,
        ))
        ax.text(label_centers[split], 0.830, name, ha="center", va="center",
                fontsize=10.8, color=edge, fontweight="bold")

    # grey clinical-trial records above the timeline
    clip_x = [0.17, 0.29, 0.41, 0.55, 0.66, 0.77, 0.88]
    for x in clip_x:
        draw_clipboard_icon(ax, x, 0.610, color="#DDDDDD")

    # exact split boundaries and timeline
    val_cut = temporal["val"]["date_min"]
    test_cut = temporal["test"]["date_min"]
    ax.annotate("", xy=(0.93, 0.420), xytext=(0.08, 0.420),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.65, mutation_scale=14))
    for xpos, cut, edge in [(0.50, val_cut, ORANGE), (0.66, test_cut, RED)]:
        ax.plot([xpos, xpos], [0.410, 0.720], color=edge, lw=1.25, ls=(0, (3, 3)))
        ax.add_patch(patches.Circle((xpos, 0.420), 0.013, transform=ax.transAxes, fc=edge, ec=edge, lw=0))
        ax.text(xpos, 0.360, cut, ha="center", va="top", fontsize=10.2, color=edge)
    ax.add_patch(patches.Circle((0.08, 0.420), 0.009, transform=ax.transAxes, fc=BLUE, ec=BLUE, lw=0))
    ax.text(0.08, 0.360, temporal["train"]["date_min"], ha="left", va="top", fontsize=10.2, color=INK)

    for split, _, _, edge, _, _ in segments:
        x = label_centers[split]
        draw_split_pie(ax, (x, 0.205), temporal[split]["label_counts"], radius=0.041)
        ax.text(x, 0.082, f"{temporal[split]['rows']:,} pairs", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.4, color=MUTED)


# -----------------------------------------------------------------------------
# Panels B-E: statistical characterization
# -----------------------------------------------------------------------------

def draw_label_distribution(ax, stats: dict) -> None:
    style_stat_axis(ax)
    add_panel_label(ax, "B", "Label distribution by split", y=1.08)

    splits = ["train", "val", "test"]
    groups = ["drug_disjoint", "temporal_submit"]
    labels = short_split_labels(groups, splits)
    neg, pos = [], []
    for group in groups:
        source = split_source(stats, group)
        for split in splits:
            counts = source[split]["label_counts"]
            neg.append(int(counts.get("0", 0)))
            pos.append(int(counts.get("1", 0)))

    x = list(range(len(labels)))
    ax.bar(x, neg, color=NEG_FILL, edgecolor=NEG_EDGE, linewidth=BAR_LW, label="Negative")
    ax.bar(x, pos, bottom=neg, color=POS_FILL, edgecolor=POS_EDGE, linewidth=BAR_LW, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Pairs", color=INK)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.10), handlelength=1.8, labelspacing=0.45)


def draw_label_distribution_for_split(ax, stats: dict, group: str, label: str, title: str) -> None:
    """Stacked label counts for one split protocol."""
    style_stat_axis(ax)
    add_panel_label_compact(ax, label, title, y=1.08)

    source = split_source(stats, group)
    splits = ["train", "val", "test"]
    neg = [int(source[s]["label_counts"].get("0", 0)) for s in splits]
    pos = [int(source[s]["label_counts"].get("1", 0)) for s in splits]
    x = list(range(len(splits)))

    ax.bar(x, neg, color=NEG_FILL, edgecolor=NEG_EDGE, linewidth=BAR_LW, label="Negative")
    ax.bar(x, pos, bottom=neg, color=POS_FILL, edgecolor=POS_EDGE, linewidth=BAR_LW, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_ylabel("Pairs", color=INK)
    ax.set_ylim(0, max([n + p for n, p in zip(neg, pos)]) * 1.24)
    for i, (n, p) in enumerate(zip(neg, pos)):
        ax.text(i, n + p + max(neg + pos) * 0.030, f"{n+p:,}",
                ha="center", va="bottom", fontsize=8.6, color=MUTED)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.13),
              handlelength=1.5, labelspacing=0.35)


def draw_unique_coverage(ax, stats: dict, label: str = "C", title: str = "Coverage of unique drugs and diseases") -> None:
    style_stat_axis(ax)
    add_panel_label_compact(ax, label, title, y=1.08)

    splits = ["train", "val", "test"]
    dd = split_source(stats, "drug_disjoint")
    ts = split_source(stats, "temporal_submit")
    labels = ["DD\ntrain", "DD\nval", "DD\ntest", "TS\ntrain", "TS\nval", "TS\ntest"]
    drugs = [dd[s]["unique_drugs"] for s in splits] + [ts[s]["unique_drugs"] for s in splits]
    diseases = [dd[s]["unique_diseases"] for s in splits] + [ts[s]["unique_diseases"] for s in splits]

    x = list(range(len(labels)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], drugs, width, color=DRUG_FILL, edgecolor=DRUG_EDGE, linewidth=BAR_LW, label="Unique drugs")
    ax.bar([i + width / 2 for i in x], diseases, width, color=DISEASE_FILL, edgecolor=DISEASE_EDGE, linewidth=BAR_LW, label="Unique diseases")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count", color=INK)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.02, 1.13), handlelength=1.5, labelspacing=0.35)


def draw_phase_composition(ax, label: str = "F", title: str = "Clinical trial phase composition") -> None:
    style_stat_axis(ax)
    add_panel_label_compact(ax, label, title, y=1.08)

    rows = load_json("data/benchmark/processed/pair_level_dataset_with_submit_dates.json")
    phase_order = [
        "early phase 1", "phase 1", "phase 1/phase 2", "phase 2",
        "phase 2/phase 3", "phase 3", "phase 4", "n/a",
    ]
    phase_counts = {phase: {"0": 0, "1": 0} for phase in phase_order}
    for row in rows:
        label = str(row.get("label", "0"))
        phases = row.get("phases") or ["n/a"]
        for phase in phases:
            phase = phase if phase in phase_counts else "n/a"
            phase_counts[phase][label] += 1

    neg = [phase_counts[phase]["0"] for phase in phase_order]
    pos = [phase_counts[phase]["1"] for phase in phase_order]
    labels = [
        "Early\nP1", "P1", "P1/2", "P2", "P2/3", "P3", "P4", "Unknown"
    ]
    x = list(range(len(phase_order)))
    ax.bar(x, neg, color=NEG_FILL, edgecolor=NEG_EDGE, linewidth=BAR_LW, label="Negative")
    ax.bar(x, pos, bottom=neg, color=POS_FILL, edgecolor=POS_EDGE, linewidth=BAR_LW, label="Positive")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Pairs", color=INK)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.05),
              handlelength=1.5, labelspacing=0.32, fontsize=8.5)


def split_for_date(date: datetime, temporal_stats: dict) -> str:
    """Return the temporal split for an exact date."""
    for split in ["train", "val", "test"]:
        lo = datetime.fromisoformat(temporal_stats[split]["date_min"])
        hi = datetime.fromisoformat(temporal_stats[split]["date_max"])
        if lo <= date <= hi:
            return split
    return "other"


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _year_label(year: int) -> str:
    return str(year)


def _temporal_year_counts(rows: list[dict], temporal_stats: dict) -> tuple[list[int], dict[str, Counter], dict[str, datetime]]:
    """Aggregate temporal split counts by year.

    The earlier date-width solution used exact cutoff dates as bin edges. This
    made the bars near cutoff dates visually narrower/wider. For the main figure,
    equal-width yearly bars are easier to read. To keep the split assignment
    faithful, each row is first assigned to the exact temporal split by date, and
    then counted by year within that split. If a cutoff falls inside a year, the
    same calendar year can have adjacent split-colored bars, but they are drawn as
    equal-width grouped bars rather than stacked in one bar.
    """
    counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    for row in rows:
        raw = row.get("pair_date")
        if not raw:
            continue
        dt = datetime.fromisoformat(raw)
        split = split_for_date(dt, temporal_stats)
        if split in counts:
            counts[split][dt.year] += 1

    years_all = sorted({year for c in counts.values() for year in c})
    cutoffs = {
        "val": datetime.fromisoformat(temporal_stats["val"]["date_min"]),
        "test": datetime.fromisoformat(temporal_stats["test"]["date_min"]),
    }
    return years_all, counts, cutoffs


def draw_temporal_distribution(ax, label: str = "E", title: str = "Temporal-submit distribution") -> None:
    style_stat_axis(ax)
    add_panel_label_compact(ax, label, title, y=1.08)

    temporal_stats = load_json("data/benchmark/processed/temporal_submit_split_stats.json")["temporal"]
    rows = load_json("data/benchmark/processed/pair_level_dataset_with_submit_dates.json")
    years, counts, cutoffs = _temporal_year_counts(rows, temporal_stats)

    if not years:
        ax.text(0.5, 0.5, "No temporal data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED)
        return

    split_style = {
        "train": (TRAIN_FILL, TRAIN_EDGE, "Train"),
        "val": (VAL_FILL, VAL_EDGE, "Validation"),
        "test": (TEST_FILL, TEST_EDGE, "Test"),
    }

    # Equal-width categorical positions. Years that contain records from two
    # temporal splits are shown as neighboring narrow bars within the same year,
    # instead of one stacked multi-split bar.
    x_positions = {year: i for i, year in enumerate(years)}
    width = 0.24
    offsets = {"train": -width, "val": 0.0, "test": width}
    used_labels: set[str] = set()

    for split in ["train", "val", "test"]:
        xs, vals = [], []
        for year in years:
            v = counts[split][year]
            if v > 0:
                xs.append(x_positions[year] + offsets[split])
                vals.append(v)
        if not xs:
            continue
        fill, edge, label = split_style[split]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        ax.bar(xs, vals, width=width * 0.88, color=fill, edgecolor=edge,
               linewidth=BAR_LW, label=legend_label)

    # Draw cutoff markers at the boundary between years if possible. If the
    # cutoff occurs inside a year, place the dashed line within that year's
    # grouped-bar region and write the exact date above it.
    ymax = max([counts[s][y] for s in counts for y in years] + [1])
    for split_name, text in [("val", "val cutoff"), ("test", "test cutoff")]:
        cutoff = cutoffs[split_name]
        year = cutoff.year
        if year in x_positions:
            # Approximate within-year location inside the categorical year slot.
            day_of_year = cutoff.timetuple().tm_yday
            frac = (day_of_year - 1) / 366.0
            xpos = x_positions[year] - 0.45 + 0.90 * frac
        else:
            # If no data in the cutoff year, place between closest neighboring years.
            earlier = [y for y in years if y < year]
            later = [y for y in years if y > year]
            if earlier and later:
                xpos = (x_positions[max(earlier)] + x_positions[min(later)]) / 2
            elif earlier:
                xpos = x_positions[max(earlier)] + 0.5
            else:
                xpos = x_positions[min(later)] - 0.5
        ax.axvline(xpos, color=ORANGE, lw=1.25, ls="--", zorder=5)
        ax.text(xpos + 0.035, ymax * 0.98,
                text,
                rotation=90, va="top", ha="left", color=ORANGE, fontsize=7.6)

    ax.set_xlabel("Study first submit year", color=INK)
    ax.set_ylabel("Pairs", color=INK)
    ax.set_xlim(-0.8, len(years) - 0.2)
    # Avoid overly dense year ticks when many years are present.
    step = max(1, len(years) // 7)
    tick_years = years[::step]
    if years[-1] not in tick_years:
        tick_years.append(years[-1])
    ax.set_xticks([x_positions[y] for y in tick_years])
    ax.set_xticklabels([_year_label(y) for y in tick_years], rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    dedup = {lab: h for h, lab in zip(handles, labels) if not lab.startswith("_")}
    ax.legend(dedup.values(), dedup.keys(), frameon=False, ncol=1,
              loc="upper right", bbox_to_anchor=(1.0, 1.01),
              handlelength=1.35, labelspacing=0.28, fontsize=8.4)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_all() -> None:
    setup_style()
    stats = load_json("data/benchmark/processed/benchmark_stats.json")

    # Figure logic:
    #   Row 1: benchmark construction pipeline.
    #   Row 2: split design schematics with embedded label-distribution pies.
    #   Row 3: supporting statistics for coverage, time and clinical diversity.
    fig = plt.figure(figsize=(16.4, 11.4))
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[0.96, 1.06, 0.90],
        hspace=0.42,
        wspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0:3])
    ax_c = fig.add_subplot(gs[1, 3:6])
    ax_d = fig.add_subplot(gs[2, 0:2])
    ax_e = fig.add_subplot(gs[2, 2:4])
    ax_f = fig.add_subplot(gs[2, 4:6])

    draw_pipeline(ax_a, stats)
    draw_drug_disjoint_design(ax_b, stats)
    draw_temporal_submit_design(ax_c)
    draw_unique_coverage(
        ax_d, stats, "D",
        "Drug/disease coverage",
    )
    draw_temporal_distribution(
        ax_e, "E",
        "Temporal-submit years",
    )
    draw_phase_composition(ax_f, "F", "Clinical trial phases")

    fig.suptitle(
        "Figure 2 | Clinical-trial-derived benchmark construction and leakage-controlled evaluation splits",
        color=INK, fontsize=16.5, fontweight="normal", y=0.975,
    )
    fig.subplots_adjust(top=0.92, bottom=0.075, left=0.055, right=0.985)
    save(fig, "figure2_benchmark.svg")


if __name__ == "__main__":
    render_all()
