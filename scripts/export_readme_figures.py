#!/usr/bin/env python3
"""Export paper Figures 2–4 as README-ready PNGs under docs/figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

EXPORTS = [
    ("Fig2.pdf", "fig2_construction_pipeline.png"),
    ("Fig3.pdf", "fig3_sfq_agent.png"),
    ("Fig4.pdf", "fig4_dataset_statistics.png"),
]


def _render_pdf(pdf_path: Path, out_path: Path, *, dpi: int, max_width: int) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required: pip install pymupdf\n"
            "Or copy existing PNGs into docs/figures/ manually."
        ) from exc

    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    w, h = img.size
    if w > max_width:
        img = img.resize((max_width, int(h * max_width / w)), Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True, quality=88)
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path.relative_to(ROOT)} ({img.size[0]}x{img.size[1]}, {mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        required=True,
        help="Directory containing Fig2.pdf … Fig4.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-width", type=int, default=2000)
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    if not paper_dir.is_dir():
        raise SystemExit(f"Paper directory not found: {paper_dir}")

    for pdf_name, png_name in EXPORTS:
        pdf_path = paper_dir / pdf_name
        if not pdf_path.exists():
            raise SystemExit(f"Missing {pdf_path}")
        _render_pdf(pdf_path, OUT_DIR / png_name, dpi=args.dpi, max_width=args.max_width)


if __name__ == "__main__":
    main()
