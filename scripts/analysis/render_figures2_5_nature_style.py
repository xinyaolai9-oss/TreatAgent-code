#!/usr/bin/env python3
"""Render the final Nature-style Figure 2-5 publication bundle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figure" / "final_results",
        help="Directory for SVG, PDF, TIFF, PNG previews and source data.",
    )
    args = parser.parse_args()
    os.environ["TREATAGENT_FIGURE_OUT_DIR"] = str(args.out_dir.resolve())

    from scripts.analysis.render_benchmark_figure2_nature_style import render_all as render_figure2
    from scripts.analysis.render_final_result_figures_nature_style import (
        export_source_data,
        render_figure3,
        render_figure4,
        render_figure5,
        write_contract,
    )

    render_figure2()
    export_source_data()
    write_contract()
    render_figure3()
    render_figure4()
    render_figure5()
    print(f"Wrote Figure 2-5 publication bundle to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
