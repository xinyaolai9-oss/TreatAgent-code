#!/usr/bin/env python3
"""Render representative molecule structures from drug-disjoint splits.

The script samples balanced positive/negative examples from train/val/test,
draws the largest covalent fragment for salts/mixtures, and records metadata
so the figure assets are traceable to benchmark rows.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "benchmark" / "splits"
OUT_DIR = ROOT / "figure" / "final_results" / "molecules" / "drug_disjoint"
N_PER_LABEL = 3
IMAGE_SIZE = (520, 390)


COLORS = {
    "train": (47, 111, 178),
    "val": (78, 154, 69),
    "test": (115, 86, 168),
    "positive": (78, 154, 69),
    "negative": (159, 17, 17),
    "ink": (17, 17, 17),
    "muted": (110, 119, 129),
    "panel": (248, 251, 255),
}


def slugify(text: str, max_len: int = 42) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return text[:max_len] or "unknown"


def load_split(split: str) -> list[dict]:
    path = SPLIT_DIR / f"drug_disjoint_{split}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def largest_fragment_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    # Prefer chemically informative fragments; avoid drawing isolated ions when
    # a salt/counterion is present.
    frag = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))
    if frag.GetNumHeavyAtoms() < 8:
        return None
    return Chem.MolToSmiles(frag, canonical=True)


def representative_name(row: dict) -> str:
    drugs = row.get("drugs") or []
    if drugs:
        return str(drugs[0]).strip()
    return row.get("pair_id", "unknown")


def choose_examples(rows: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for label in [1, 0]:
        label_rows = [r for r in rows if int(r.get("label", -1)) == label]
        count = 0
        for row in label_rows:
            if not row.get("drugs"):
                continue
            smiles = row.get("canonical_smiles") or row.get("example_smiles")
            draw_smiles = largest_fragment_smiles(smiles)
            if not draw_smiles:
                continue
            row = dict(row)
            row["draw_smiles"] = draw_smiles
            chosen.append(row)
            count += 1
            if count >= N_PER_LABEL:
                break
    return chosen


def mol_to_svg(mol: Chem.Mol, path: Path) -> None:
    drawer = rdMolDraw2D.MolDraw2DSVG(IMAGE_SIZE[0], IMAGE_SIZE[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.8
    opts.minFontSize = 12
    opts.maxFontSize = 20
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    path.write_text(svg, encoding="utf-8")


def mol_to_png(mol: Chem.Mol, path: Path) -> None:
    img = Draw.MolToImage(mol, size=IMAGE_SIZE, kekulize=True)
    img.save(path)


def draw_card(row: dict, split: str, png_path: Path) -> Image.Image:
    card_w, card_h = 640, 560
    card = Image.new("RGB", (card_w, card_h), "white")
    draw = ImageDraw.Draw(card)
    split_color = COLORS[split]
    label_color = COLORS["positive"] if int(row["label"]) == 1 else COLORS["negative"]
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_mid = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font_title = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=18, outline=split_color, width=3)
    draw.rectangle([0, 0, card_w, 46], fill=tuple(int(c * 0.12 + 255 * 0.88) for c in split_color))
    draw.text((18, 13), split.upper(), fill=split_color, font=font_mid)
    draw.text((128, 13), row["pair_id"], fill=COLORS["ink"], font=font_mid)
    label_text = "Positive" if int(row["label"]) == 1 else "Negative"
    draw.rounded_rectangle([card_w - 142, 10, card_w - 18, 36], radius=10, outline=label_color, width=2)
    draw.text((card_w - 128, 15), label_text, fill=label_color, font=font_small)

    mol_img = Image.open(png_path).convert("RGB")
    card.paste(mol_img, (60, 62))

    name = representative_name(row)
    disease = str(row.get("normalized_disease") or row.get("example_disease") or "")
    draw.text((18, 464), name[:54], fill=COLORS["ink"], font=font_title)
    draw.text((18, 500), disease[:64], fill=COLORS["muted"], font=font_mid)
    return card


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] = []
    cards: list[Image.Image] = []

    for split in ["train", "val", "test"]:
        rows = load_split(split)
        for idx, row in enumerate(choose_examples(rows), start=1):
            mol = Chem.MolFromSmiles(row["draw_smiles"])
            if mol is None:
                continue
            Chem.rdDepictor.Compute2DCoords(mol)
            name = representative_name(row)
            stem = f"{split}_{idx:02d}_{row['pair_id']}_{slugify(name)}"
            svg_path = OUT_DIR / f"{stem}.svg"
            png_path = OUT_DIR / f"{stem}.png"
            mol_to_svg(mol, svg_path)
            mol_to_png(mol, png_path)
            cards.append(draw_card(row, split, png_path))
            metadata.append({
                "split": split,
                "pair_id": row.get("pair_id", ""),
                "label": row.get("label", ""),
                "drug": name,
                "disease": row.get("normalized_disease", ""),
                "canonical_smiles": row.get("canonical_smiles", ""),
                "draw_smiles_largest_fragment": row.get("draw_smiles", ""),
                "svg": str(svg_path.relative_to(ROOT)).replace("\\", "/"),
                "png": str(png_path.relative_to(ROOT)).replace("\\", "/"),
            })

    with (OUT_DIR / "molecule_examples_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)

    # Three rows (train/val/test) x six columns (3 pos + 3 neg).
    if cards:
        cols = 6
        rows = (len(cards) + cols - 1) // cols
        pad = 18
        sheet = Image.new(
            "RGB",
            (cols * 640 + (cols + 1) * pad, rows * 560 + (rows + 1) * pad),
            COLORS["panel"],
        )
        for i, card in enumerate(cards):
            x = pad + (i % cols) * (640 + pad)
            y = pad + (i // cols) * (560 + pad)
            sheet.paste(card, (x, y))
        sheet.save(OUT_DIR / "drug_disjoint_molecule_examples_contact_sheet.png")


if __name__ == "__main__":
    render()
