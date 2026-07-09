#!/usr/bin/env python3
from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "figure1_treatagent_arg_overview_4k.png"

S = 2
W, H = 1920 * S, 1080 * S


def p(v: int | float) -> int:
    return int(round(v * S))


def xy(box: tuple[int | float, int | float, int | float, int | float]) -> tuple[int, int, int, int]:
    return tuple(p(v) for v in box)  # type: ignore[return-value]


FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/mnt/c/Windows/Fonts"),
    Path("C:/Windows/Fonts"),
]


def first_existing(names: list[str]) -> str | None:
    for directory in FONT_DIRS:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return None


def font(size: int, *, bold: bool = False, serif: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        names = ["DejaVuSerif-Bold.ttf", "timesbd.ttf", "georgiab.ttf", "arialbd.ttf"] if bold else ["DejaVuSerif.ttf", "times.ttf", "georgia.ttf", "arial.ttf"]
    elif italic:
        names = ["DejaVuSans-Oblique.ttf", "ariali.ttf", "Arial Italic.ttf"]
    else:
        names = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
    path = first_existing(names)
    if path:
        return ImageFont.truetype(path, p(size))
    return ImageFont.load_default(size=p(size))


F = {
    "title": font(30, bold=True, serif=True),
    "subtitle": font(13, italic=True),
    "section": font(17, bold=True),
    "card_title": font(12, bold=True),
    "card_sub": font(8, italic=True),
    "text": font(9),
    "small": font(7),
    "tiny": font(6),
    "num": font(12, bold=True),
    "module": font(15, bold=True),
}


BLUE = "#0b4aa2"
DEEP_BLUE = "#083b8e"
PURPLE = "#5b2fa2"
ORANGE = "#f15a24"
GREEN = "#178f3a"
RED = "#f02b2b"
GRAY = "#6b7280"
LIGHT_BG = "#fbfdff"
INK = "#0f172a"


def text(draw: ImageDraw.ImageDraw, pos: tuple[int | float, int | float], value: str, fill=INK, f=None, anchor=None):
    draw.text((p(pos[0]), p(pos[1])), value, fill=fill, font=f or F["text"], anchor=anchor)


def centered(draw: ImageDraw.ImageDraw, box: tuple[int | float, int | float, int | float, int | float], value: str, fill=INK, f=None):
    draw.text(((p(box[0]) + p(box[2])) // 2, (p(box[1]) + p(box[3])) // 2), value, fill=fill, font=f or F["text"], anchor="mm")


def wrap_lines(value: str, width: int, max_lines: int | None = None) -> list[str]:
    lines = textwrap.wrap(value, width=width)
    if max_lines and len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: max(0, width - 3)] + "..."]
    return lines or [""]


def wrapped(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, width: int, *, fill=INK, f=None, line_h=12, max_lines=None):
    for i, line in enumerate(wrap_lines(value, width, max_lines)):
        text(draw, (x, y + i * line_h), line, fill=fill, f=f)


def rr(draw: ImageDraw.ImageDraw, box, radius=10, fill="white", outline=BLUE, width=1):
    draw.rounded_rectangle(xy(box), radius=p(radius), fill=fill, outline=outline, width=p(width))


def line(draw: ImageDraw.ImageDraw, points, fill=INK, width=1, joint="curve"):
    draw.line([(p(x), p(y)) for x, y in points], fill=fill, width=p(width), joint=joint)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=INK, width=1.4):
    x1, y1 = start
    x2, y2 = end
    line(draw, [(x1, y1), (x2, y2)], fill=fill, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 8
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45)),
        (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45)),
    ]
    draw.polygon([(p(x), p(y)) for x, y in pts], fill=fill)


def dashed(draw: ImageDraw.ImageDraw, start, end, fill=GRAY, width=1, dash=8):
    x1, y1 = start
    x2, y2 = end
    dist = math.hypot(x2 - x1, y2 - y1)
    steps = max(1, int(dist / dash))
    for i in range(0, steps, 2):
        a = i / steps
        b = min(1, (i + 1) / steps)
        line(draw, [(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill=fill, width=width)


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, n: int, color: str):
    draw.ellipse(xy((x, y, x + 28, y + 28)), fill=color, outline="white", width=p(2))
    centered(draw, (x, y, x + 28, y + 28), str(n), fill="white", f=F["num"])


def card(draw: ImageDraw.ImageDraw, box, title, subtitle, n, color=BLUE):
    rr(draw, box, 14, fill="white", outline=color, width=1.6)
    badge(draw, box[0] + 16, box[1] + 18, n, color)
    text(draw, (box[0] + 52, box[1] + 24), title, fill=color, f=F["card_title"])
    if subtitle:
        text(draw, (box[0] + 52, box[1] + 47), subtitle, fill=INK, f=F["card_sub"])


def simple_molecule(draw: ImageDraw.ImageDraw, x: int, y: int):
    pts = [(x, y), (x + 24, y - 16), (x + 48, y), (x + 72, y - 16), (x + 96, y), (x + 124, y)]
    line(draw, pts, fill="black", width=1.2)
    line(draw, [(x + 96, y), (x + 96, y - 22)], fill="black", width=1)
    text(draw, (x + 90, y - 38), "O", f=F["text"])
    text(draw, (x + 126, y - 4), "OH", f=F["text"])
    line(draw, [(x + 20, y - 45), (x + 52, y - 63), (x + 84, y - 45), (x + 84, y - 10), (x + 52, y + 8), (x + 20, y - 10), (x + 20, y - 45)], fill="black", width=1)
    text(draw, (x + 62, y - 65), "N", f=F["small"])


def icon_database(draw: ImageDraw.ImageDraw, x: int, y: int, color=BLUE):
    draw.ellipse(xy((x, y, x + 32, y + 12)), outline=color, width=p(2), fill="#dbeafe")
    draw.rectangle(xy((x, y + 6, x + 32, y + 38)), outline=color, width=p(2), fill="#dbeafe")
    draw.ellipse(xy((x, y + 30, x + 32, y + 42)), outline=color, width=p(2), fill="#dbeafe")


def icon_target(draw: ImageDraw.ImageDraw, x: int, y: int, color=ORANGE):
    for r in [24, 14, 5]:
        draw.ellipse(xy((x + 24 - r, y + 24 - r, x + 24 + r, y + 24 + r)), outline=color, width=p(2))
    line(draw, [(x, y + 24), (x + 48, y + 24)], fill=color, width=1.5)
    line(draw, [(x + 24, y), (x + 24, y + 48)], fill=color, width=1.5)


def icon_shield(draw: ImageDraw.ImageDraw, x: int, y: int, color=RED):
    pts = [(x + 24, y), (x + 45, y + 9), (x + 40, y + 38), (x + 24, y + 54), (x + 8, y + 38), (x + 3, y + 9)]
    draw.polygon([(p(a), p(b)) for a, b in pts], fill="#fff1f2", outline=color)
    text(draw, (x + 24, y + 31), "!", fill=color, f=font(20, bold=True), anchor="mm")


def expert_box(draw, x, y, label, icon, color):
    rr(draw, (x, y, x + 185, y + 50), 7, fill="white", outline=color, width=1)
    if icon == "db":
        icon_database(draw, x + 18, y + 8, color)
    elif icon == "target":
        icon_target(draw, x + 11, y + 1, color)
    elif icon == "shield":
        icon_shield(draw, x + 13, y + 2, color)
    else:
        text(draw, (x + 30, y + 30), "✚", fill=color, f=font(20, bold=True), anchor="mm")
    centered(draw, (x + 66, y, x + 170, y + 50), label, fill=color if color == RED else INK, f=F["module"])


def evidence_tuple(draw, x, y, source, sign, claim, r, color):
    rr(draw, (x, y, x + 285, y + 38), 7, fill="#fbfffb" if color == GREEN else "#fffafa" if color == RED else "#f8fafc", outline=color, width=1)
    rr(draw, (x + 10, y + 8, x + 78, y + 30), 4, fill="white", outline=color, width=1)
    centered(draw, (x + 10, y + 8, x + 78, y + 30), source, fill=color, f=F["small"])
    draw.ellipse(xy((x + 92, y + 6, x + 124, y + 32)), outline=color, width=p(1.5), fill="white")
    centered(draw, (x + 92, y + 6, x + 124, y + 32), sign, fill=color, f=font(13, bold=True))
    wrapped(draw, x + 138, y + 9, claim, 14, f=F["small"], line_h=10)
    text(draw, (x + 220, y + 15), f"r = {r:.2f}", fill=INK, f=F["small"])
    text(draw, (x + 260, y + 16), "▣", fill=BLUE, f=F["small"])


def arg_node(draw, x, y, label, color, direction="support"):
    rr(draw, (x, y, x + 105, y + 72), 7, fill="#f7fff9" if color == GREEN else "#fff7f7" if color == RED else "white", outline=color, width=1)
    wrapped(draw, x + 12, y + 16, label, 13, fill=color if color != GRAY else INK, f=F["small"], line_h=10)


def rule_row(draw, x, y, title, sub, color, symbol):
    rr(draw, (x, y, x + 210, y + 42), 7, fill="#ffffff", outline=color, width=1)
    centered(draw, (x + 10, y + 8, x + 38, y + 34), symbol, fill=color, f=font(16, bold=True))
    text(draw, (x + 48, y + 9), title, fill=color, f=F["card_title"])
    text(draw, (x + 48, y + 27), sub, fill=INK, f=F["tiny"])


def draw_pipeline(draw: ImageDraw.ImageDraw):
    text(draw, (24, 96), "OVERVIEW PIPELINE", fill=BLUE, f=F["section"])
    boxes = [
        (20, 125, 225, 575),
        (275, 125, 505, 575),
        (540, 125, 780, 575),
        (815, 125, 1125, 575),
        (1160, 125, 1455, 575),
        (1490, 125, 1705, 575),
        (1740, 125, 1900, 575),
    ]
    titles = [
        ("SMILES +\nDisease", "", BLUE),
        ("Constrained\nPlanner", "route next\nevidence action", PURPLE),
        ("Evidence Agents", "knowledge and data sources", BLUE),
        ("Typed Evidence Tuples", "standardized evidence cards", ORANGE),
        ("Argument EvidenceGraph", "claim-centered argument map", BLUE),
        ("Reliability-aware\nARG Rule", "", BLUE),
        ("Treatment\nScore +\nDecision", "", BLUE),
    ]
    for i, (box, (title, sub, color)) in enumerate(zip(boxes, titles), start=1):
        card(draw, box, title, sub, i, color)
    for a, b in zip(boxes, boxes[1:]):
        arrow(draw, (a[2] + 10, (a[1] + a[3]) / 2), (b[0] - 8, (b[1] + b[3]) / 2), fill=INK, width=1.2)

    simple_molecule(draw, 55, 285)
    dashed(draw, (42, 345), (205, 345), fill=GRAY)
    rr(draw, (36, 388, 108, 478), 6, fill="#f8fbff", outline=BLUE)
    centered(draw, (36, 388, 108, 478), "⚕", fill=BLUE, f=font(38))
    wrapped(draw, 125, 405, "Disease (e.g., breast cancer)", 14, f=F["small"], line_h=11)

    rr(draw, (328, 250, 445, 280), 6, fill="#ffffff", outline=GRAY)
    centered(draw, (328, 250, 445, 280), "Observe state", f=F["text"])
    draw.ellipse(xy((345, 312, 440, 407)), outline=PURPLE, width=p(1.5), fill="#faf7ff")
    for x, y in [(392, 336), (368, 381), (416, 381)]:
        rr(draw, (x - 9, y - 9, x + 9, y + 9), 2, fill=PURPLE, outline=PURPLE)
        arrow(draw, (392, 345), (x, y - 12), fill=PURPLE, width=1)
    arrow(draw, (386, 280), (386, 310), fill=INK)
    arrow(draw, (386, 407), (386, 472), fill=INK)
    rr(draw, (300, 520, 365, 555), 6, fill="#eaffee", outline=GREEN)
    centered(draw, (300, 520, 365, 555), "STOP", fill=GREEN, f=F["card_title"])
    text(draw, (300, 484), "Yes", fill=GREEN, f=F["small"])
    text(draw, (442, 482), "No", fill=INK, f=F["small"])

    expert_box(draw, 558, 218, "DrugKB", "db", BLUE)
    expert_box(draw, 558, 288, "DiseaseKB", "db", BLUE)
    expert_box(draw, 558, 358, "DTI", "target", ORANGE)
    expert_box(draw, 558, 428, "ADMET", "shield", RED)
    expert_box(draw, 558, 498, "Clinical", "plus", PURPLE)
    for y in [243, 313, 383, 453, 523]:
        dashed(draw, (743, y), (815, y), fill=INK)

    evidence_tuple(draw, 830, 218, "DrugKB", "+", "known indication", 0.88, GREEN)
    evidence_tuple(draw, 830, 284, "DTI", "+", "target support", 0.63, GREEN)
    evidence_tuple(draw, 830, 350, "ADMET", "−", "toxicity risk", 0.70, RED)
    evidence_tuple(draw, 830, 416, "Clinical", "?", "weak prior", 0.42, GRAY)
    text(draw, (830, 540), "⊕ Support   ⊖ Conflict   ? Missing   ▣ Provenance", fill=INK, f=F["tiny"])

    arg_node(draw, 1180, 210, "DrugKB\nknown indication\nr=0.88", GREEN)
    arg_node(draw, 1354, 210, "DTI\ntarget support\nr=0.63", GREEN)
    arg_node(draw, 1180, 392, "ADMET\ntoxicity risk\nr=0.70", RED)
    arg_node(draw, 1354, 420, "Clinical\nweak prior\nr=0.42", GRAY)
    draw.ellipse(xy((1278, 318, 1348, 388)), fill="#eff6ff", outline=BLUE, width=p(1.5))
    centered(draw, (1278, 318, 1348, 388), "Drug\ntreats\ndisease?", fill=BLUE, f=F["small"])
    arrow(draw, (1235, 275), (1288, 332), fill=GREEN)
    arrow(draw, (1370, 275), (1337, 332), fill=GREEN)
    arrow(draw, (1238, 428), (1285, 384), fill=RED)
    dashed(draw, (1354, 455), (1340, 385), fill=GRAY)
    text(draw, (1175, 540), "→ Support    → Conflict   --→ Missing   ▣ Provenance", fill=INK, f=F["tiny"])

    rule_row(draw, 1508, 210, "Support aggregate", "(reliability-weighted)", GREEN, "Σ")
    rule_row(draw, 1508, 284, "Conflict penalty", "(reliability-weighted)", RED, "−")
    rule_row(draw, 1508, 358, "Consistency bonus", "(cross-source)", BLUE, "↔")
    rule_row(draw, 1508, 432, "Missing-evidence penalty", "(uncertainty cost)", GRAY, "?")
    text(draw, (1528, 525), "Score = Support", fill=INK, f=F["small"])
    text(draw, (1518, 545), "− Conflict + Consistency − Missing", fill=INK, f=F["tiny"])

    draw.arc(xy((1780, 238, 1870, 328)), start=190, end=350, fill="#facc15", width=p(5))
    draw.arc(xy((1780, 238, 1870, 328)), start=190, end=240, fill=GREEN, width=p(5))
    draw.arc(xy((1780, 238, 1870, 328)), start=310, end=350, fill=RED, width=p(5))
    line(draw, [(1825, 300), (1852, 265)], fill="black", width=3)
    draw.ellipse(xy((1816, 291, 1834, 309)), fill="black")
    centered(draw, (1740, 360, 1900, 380), "Score", f=F["text"])
    centered(draw, (1740, 392, 1900, 412), "+0.73", fill=GREEN, f=F["module"])
    dashed(draw, (1758, 430), (1882, 430), fill=GRAY)
    rr(draw, (1778, 450, 1862, 520), 8, fill="#f7fff9", outline=GREEN)
    centered(draw, (1778, 450, 1862, 520), "Decision\nTreat", fill=GREEN, f=F["text"])
    rr(draw, (1718, 600, 1900, 680), 8, fill="#f8fbff", outline=BLUE)
    centered(draw, (1718, 600, 1900, 680), "Graph-grounded\nExplanation", fill=BLUE, f=F["card_title"])


def draw_key_modules(draw: ImageDraw.ImageDraw):
    rr(draw, (10, 605, 1910, 1065), 12, fill="white", outline="#cbd5e1")
    text(draw, (30, 640), "KEY MODULES", fill=BLUE, f=F["section"])
    panels = [
        (20, 675, 425, 1040),
        (440, 675, 895, 1040),
        (910, 675, 1350, 1040),
        (1365, 675, 1900, 1040),
    ]
    titles = [
        ("Constrained Planner", "state-aware routing", PURPLE),
        ("Typed Evidence Tuples", "standardized evidence cards", ORANGE),
        ("Argument EvidenceGraph", "support / conflict / missing", BLUE),
        ("ARG Rule", "reliability-aware aggregation", BLUE),
    ]
    for panel, (title, sub, color) in zip(panels, titles):
        rr(draw, panel, 13, fill="white", outline=color, width=1)
        centered(draw, (panel[0], panel[1] + 18, panel[2], panel[1] + 45), title, fill=color, f=F["module"])
        centered(draw, (panel[0], panel[1] + 50, panel[2], panel[1] + 70), sub, fill=INK, f=F["card_sub"])

    # Planner panel
    x, y = 36, 760
    for i, label in enumerate(["Observe\nstate", "Choose\nnext\naction"]):
        rr(draw, (x + i * 105, y, x + 65 + i * 105, y + 102), 7, fill="#f8fbff", outline=GRAY)
        centered(draw, (x + i * 105, y, x + 65 + i * 105, y + 102), label, f=F["small"])
    arrow(draw, (101, 810), (140, 810))
    arrow(draw, (206, 810), (250, 810))
    draw.polygon([(p(285), p(755)), (p(335), p(810)), (p(285), p(865)), (p(235), p(810))], fill="white", outline=GRAY)
    centered(draw, (235, 755, 335, 865), "Enough\nevidence?", f=F["small"])
    rr(draw, (342, 804, 405, 842), 7, fill="#eaffee", outline=GREEN)
    centered(draw, (342, 804, 405, 842), "STOP", fill=GREEN, f=F["text"])
    text(draw, (338, 770), "Yes", fill=GREEN, f=F["small"])
    rr(draw, (240, 900, 325, 972), 7, fill="#faf7ff", outline=PURPLE)
    centered(draw, (240, 900, 325, 972), "Continue\nrouting", fill=PURPLE, f=F["small"])
    line(draw, [(282, 865), (282, 900)], fill=INK)
    line(draw, [(282, 972), (36, 972), (36, 862)], fill=PURPLE, width=1.3)
    for i, item in enumerate(["fixed action space (experts)", "query-specific routing", "stop when sufficient evidence"]):
        text(draw, (45, 960 + i * 24), f"• {item}", f=F["small"])

    # Tuple panel
    evidence_tuple(draw, 455, 760, "DrugKB", "+", "known indication", 0.88, GREEN)
    evidence_tuple(draw, 455, 828, "DTI", "+", "target support", 0.63, GREEN)
    evidence_tuple(draw, 455, 896, "ADMET", "−", "toxicity risk", 0.70, RED)
    evidence_tuple(draw, 455, 964, "Clinical", "?", "weak prior", 0.42, GRAY)
    for y2, label in [(779, "source\n(provenance)"), (848, "direction"), (915, "claim\n(short statement)"), (982, "reliability\n(0–1)")]:
        dashed(draw, (740, y2), (825, y2), fill=GRAY)
        wrapped(draw, 830, y2 - 10, label, 16, f=F["tiny"], line_h=9)
    text(draw, (455, 1020), "⊕ Support    ⊖ Conflict    ? Missing    ▣ Provenance", fill=INK, f=F["small"])

    # Graph panel
    arg_node(draw, 935, 760, "Support\n(DrugKB)\nr=0.88", GREEN)
    arg_node(draw, 1210, 760, "Support\n(DTI)\nr=0.63", GREEN)
    arg_node(draw, 935, 900, "Conflict\n(ADMET)\nr=0.70", RED)
    arg_node(draw, 1210, 900, "Missing\n(Clinical)\nr=0.42", GRAY)
    draw.ellipse(xy((1068, 825, 1160, 917)), fill="#eff6ff", outline=BLUE, width=p(1.5))
    centered(draw, (1068, 825, 1160, 917), "Drug\ntreats\ndisease?", fill=BLUE, f=F["small"])
    arrow(draw, (1040, 795), (1075, 843), fill=GREEN)
    arrow(draw, (1220, 795), (1150, 843), fill=GREEN)
    arrow(draw, (1038, 936), (1078, 900), fill=RED)
    dashed(draw, (1210, 936), (1160, 905), fill=GRAY)
    line(draw, [(925, 990), (1330, 990)], fill="#cbd5e1")
    text(draw, (925, 1018), "→ Support      → Conflict      --→ Missing      ▣ Provenance", fill=INK, f=F["small"])

    # Rule panel
    rule_row(draw, 1385, 760, "Direct support", "known indication, target support", GREEN, "Σ")
    rule_row(draw, 1385, 828, "Clinical feasibility", "clinical outcomes, real-world use", GREEN, "●")
    rule_row(draw, 1385, 896, "Cross-source consistency", "agreement across independent sources", BLUE, "↔")
    rule_row(draw, 1385, 964, "Safety / mechanism / knowledge\nconflict penalty", "toxicity risk, contradictory evidence", RED, "!")
    text(draw, (1775, 825), "{", fill=INK, f=font(150))
    rr(draw, (1810, 795, 1885, 920), 10, fill="#f8fbff", outline=BLUE)
    centered(draw, (1810, 795, 1885, 920), "Treatment\nscore\nS", fill=BLUE, f=F["module"])


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(image)
    text(draw, (960, 18), "Figure 1. Overview of TreatAgent-ARG", fill="black", f=F["title"], anchor="ma")
    text(
        draw,
        (960, 66),
        "Constrained multi-agent evidence acquisition and reliability-aware argument reasoning for drug-disease treatment prediction",
        fill=INK,
        f=F["subtitle"],
        anchor="ma",
    )
    draw_pipeline(draw)
    draw_key_modules(draw)
    image.save(OUT_PATH, quality=95, dpi=(300, 300))
    print(OUT_PATH)


if __name__ == "__main__":
    render()
