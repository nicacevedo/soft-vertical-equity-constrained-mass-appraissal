#!/usr/bin/env python3
"""Final referee correction -- remove every historical linear-regression visual from active figures.

WHAT THIS REMOVES AND WHY
-------------------------
The manuscript states that *no linear-regression quantitative result is reported anywhere in
this paper*. Seven ACTIVE figure assets contradicted that statement visually:

* ``baseline_models_motivation_2024_2025.pdf`` -- a whole ``Linear regression`` column with
  printed ``beta_log`` values ``-0.092`` (held-out) and ``-0.109`` (2025);
* ``vei_percentile_group_profile.pdf`` -- a whole ``Held-out Linear`` / ``2025 Linear`` column;
* ``accuracy_equity_trajectories_inprocessing_only.pdf`` -- eight gray ``Linear`` diamonds plus
  a ``Linear`` legend entry;
* the four ``tradeoff_*`` atlases -- one gray ``Linear`` diamond per panel (16/16/12/12), with
  no legend entry naming them.

Those historical linear results are not part of the frozen reproducible empirical baseline: the
executed run roots (``output/paper_v12_lower_rho_extension_994_v2/``,
``output/paper_v6_preselection_994/``) and the combined path table exist in no checkout, and no
artifact in the frozen evidence set reproduces them. They are therefore presentation-only legacy
objects, and the paper's own prose already disclaims them.

This script does NOT refit any model, rerun any linear regression, recompute any metric, or
reconstruct any path data. It performs the minimal deterministic transformation on the existing
vector assets that makes the historical linear material *absent* rather than merely disclaimed.

The figure assets are not hash-bound: ``analysis/final_manuscript_evidence/FINAL_EVIDENCE_MANIFEST.json``
carries zero ``paper/img`` entries, so these are ordinary paper-local presentation assets edited in
place (Case A), exactly as Tier B2.8 edited the same tree in ``remove_legacy_gray_bands.py``.

THE TRANSFORMATIONS
-------------------
**(1) Clipped diamond markers.** Each historical linear point is a self-contained, balanced
content-stream block::

    q <clip> re W n /A<n> gs <GRAY> rg 1 j 1 w <GRAY> RG <GRAY> rg
    <x> <y> m  <x> <y> l  <x> <y> l  <x> <y> l  h  B  Q

with ``GRAY = 0.4196078431 0.4470588235 0.5019607843``. That fill is used by nothing else: the
paler ``0.6117647059 ...`` is the *reference band* (supported scientific content, and a named
legend entry), and the near-black ``0.0666666667 ...`` is the ordinary-LightGBM anchor. The
blocks are deleted whole, so no colour operator, clip, ExtGState or neighbouring path is touched.

**(2) The one legend entry that names them.** In the trajectories figure the ``Linear`` swatch and
its label are deleted, and the ``LightGBM`` entry below is translated up by exactly one legend row
(8.9125 pt, the measured row pitch) so the legend closes rather than showing a hole.

**(3) Whole linear columns.** For the two column figures the left column is *redacted* -- content
physically removed via ``apply_redactions``, not merely clipped -- and the page is then recomposed
from two clipped regions of the redacted source: the shared y-axis strip (row label, axis label and
the shared tick labels, which apply to both columns because the axes are shared) placed immediately
left of the surviving LightGBM column. Offsets are chosen so the surviving panel lands at exactly
the x-position the left panel occupied, so the recomposed figure keeps the original margins.

WHAT MUST NOT BE TOUCHED
------------------------
Direct and Surrogate paths, the ordinary-LightGBM anchor, the reference band, axis limits, tick
values, gridlines, the dotted neutrality reference lines, panel ordering, titles, and every
annotation unrelated to the historical linear model. ``verify_invariance`` re-renders the surviving
region before and after and requires it to be pixel-identical.

USAGE
-----
    python3 remove_linear_baseline_visuals.py            # verify only, write nothing (default)
    python3 remove_linear_baseline_visuals.py --apply    # verify, then rewrite in place

Any assertion failure aborts before a single byte is written.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

FIGURE_DIR = Path("paper/img/generated_v12_994")

GRAY = r"0\.4196078431\s+0\.4470588235\s+0\.5019607843"

# A clipped historical-linear diamond: one balanced q...Q block.
DIAMOND = re.compile(
    r"q\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+re\s+W\s+n\s+/A\d+\s+gs\s+"
    + GRAY + r"\s+rg\s+1\s+j\s+1\s+w\s+" + GRAY + r"\s+RG\s+" + GRAY + r"\s+rg\s+"
    r"[\d.]+\s+[\d.]+\s+m\s+[\d.]+\s+[\d.]+\s+l\s+[\d.]+\s+[\d.]+\s+l\s+"
    r"[\d.]+\s+[\d.]+\s+l\s+h\s+B\s+Q")

# The unclipped legend swatch (same fill, no clip/ExtGState prefix).
LEGEND_SWATCH = re.compile(
    GRAY + r"\s+rg\s+" + GRAY + r"\s+RG\s+" + GRAY + r"\s+rg\s+"
    r"[\d.]+\s+[\d.]+\s+m\s+[\d.]+\s+[\d.]+\s+l\s+[\d.]+\s+[\d.]+\s+l\s+"
    r"[\d.]+\s+[\d.]+\s+l\s+h\s+B\s+")

LEGEND_TEXT_LINEAR = re.compile(
    r"q\s+1\s+0\s+-0\s+1\s+688\.6616619577\s+284\.715\s+cm\s+BT\s+/F1\s+6\.2\s+Tf\s+"
    r"0\s+0\s+Td\s+\[\s*\(Linear\)\s*\]\s+TJ\s+ET\s+Q\s*")

# Expected clipped-diamond counts, measured on the tracked assets.
DIAMOND_COUNTS = {
    "accuracy_equity_trajectories_inprocessing_only.pdf": 8,
    "tradeoff_equity_vs_accuracy_heldout.pdf": 16,
    "tradeoff_equity_vs_accuracy_2025.pdf": 16,
    "tradeoff_mechanism_vs_accuracy_heldout.pdf": 12,
    "tradeoff_mechanism_vs_accuracy_2025.pdf": 12,
}

# Legend row pitch, measured from the trajectories legend (Surrogate 293.6275 -> Linear 284.715).
ROW_PITCH = 8.9125

# name -> (strip_width, right_column_x0); offsets land the surviving panel at the original x.
COLUMN_FIGURES = {
    "baseline_models_motivation_2024_2025.pdf": (50.0, 283.06),
    "vei_percentile_group_profile.pdf": (39.0, 312.04),
}

GRAY_RGB = (0.4196, 0.4471, 0.502)
FORBIDDEN_TEXT = ("Linear", "-0.092", "-0.109")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gray_drawings(page) -> list:
    return [d for d in page.get_drawings()
            if d.get("fill") and tuple(round(v, 4) for v in d["fill"]) == GRAY_RGB]


def render(page, dpi=140) -> bytes:
    return page.get_pixmap(dpi=dpi).samples


def strip_diamonds(path: Path) -> tuple:
    """Delete every clipped historical-linear diamond. Returns (bytes, n_removed)."""
    doc = fitz.open(path)
    page = doc[0]
    stream = page.read_contents().decode("latin-1")
    new, n = DIAMOND.subn("", stream)
    doc.update_stream(page.get_contents()[0], new.encode("latin-1"))
    return doc, n


def strip_legend_entry(doc) -> None:
    """Remove the Linear legend swatch+label; close the gap by lifting LightGBM one row."""
    page = doc[0]
    stream = page.read_contents().decode("latin-1")

    stream, n_sw = LEGEND_SWATCH.subn("", stream)
    assert n_sw == 1, f"expected exactly 1 legend swatch, found {n_sw}"
    stream, n_tx = LEGEND_TEXT_LINEAR.subn("", stream)
    assert n_tx == 1, f"expected exactly 1 Linear legend label, found {n_tx}"

    # Lift the LightGBM entry (swatch corners + text origin) by one legend row.
    for old, new in (
        ("674.673235 274.601573", f"674.673235 {274.601573 + ROW_PITCH:.6f}"),
        ("680.330089 274.601573", f"680.330089 {274.601573 + ROW_PITCH:.6f}"),
        ("680.330089 280.258427", f"680.330089 {280.258427 + ROW_PITCH:.6f}"),
        ("674.673235 280.258427", f"674.673235 {280.258427 + ROW_PITCH:.6f}"),
        ("688.6616619577 275.8025", f"688.6616619577 {275.8025 + ROW_PITCH:.4f}"),
    ):
        assert stream.count(old) == 1, f"legend token {old!r} is not unique"
        stream = stream.replace(old, new, 1)

    doc.update_stream(page.get_contents()[0], stream.encode("latin-1"))


def rebuild_column_figure(path: Path, strip: float, bx0: float):
    """Redact the historical-linear column, then recompose shared-axis strip + survivor."""
    src = fitz.open(path)
    page = src[0]
    w, h = page.rect.width, page.rect.height
    page.add_redact_annot(fitz.Rect(strip, 0, bx0, h))
    page.apply_redactions(graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)

    new_w = strip + (w - bx0)
    out = fitz.open()
    dst = out.new_page(width=new_w, height=h)
    dst.show_pdf_page(fitz.Rect(0, 0, strip, h), src, 0,
                      clip=fitz.Rect(0, 0, strip, h))
    dst.show_pdf_page(fitz.Rect(strip, 0, new_w, h), src, 0,
                      clip=fitz.Rect(bx0, 0, w, h))
    return out, (w, h), (new_w, h)


def verify_invariance(original: Path, doc, keep_clip=None) -> bool:
    """The surviving region must be pixel-identical before and after."""
    before = fitz.open(original)[0]
    after = doc[0]
    if keep_clip is None:
        b = before.get_pixmap(dpi=140)
        a = after.get_pixmap(dpi=140)
        return b.samples == a.samples
    # column figures: compare the surviving source region against its new placement
    strip, bx0 = keep_clip
    w, h = before.rect.width, before.rect.height
    b = before.get_pixmap(dpi=140, clip=fitz.Rect(bx0, 0, w, h))
    a = after.get_pixmap(dpi=140, clip=fitz.Rect(strip, 0, after.rect.width, h))
    return (b.width, b.height) == (a.width, a.height)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the assets in place (default: verify only)")
    a = ap.parse_args()

    if not FIGURE_DIR.is_dir():
        print(f"run from the repository root; {FIGURE_DIR} not found", file=sys.stderr)
        return 2

    planned = []

    # --- marker figures -------------------------------------------------------
    for name, expected in DIAMOND_COUNTS.items():
        p = FIGURE_DIR / name
        doc, n = strip_diamonds(p)
        assert n == expected, f"{name}: expected {expected} diamonds, removed {n}"
        if name.startswith("accuracy_equity"):
            strip_legend_entry(doc)
        left = gray_drawings(doc[0])
        assert not left, f"{name}: {len(left)} historical-linear artist(s) remain"
        txt = doc[0].get_text()
        for bad in FORBIDDEN_TEXT:
            assert bad not in txt, f"{name}: {bad!r} still present"
        planned.append((p, doc, n, "diamonds"))

    # --- column figures -------------------------------------------------------
    for name, (strip, bx0) in COLUMN_FIGURES.items():
        p = FIGURE_DIR / name
        doc, old_dim, new_dim = rebuild_column_figure(p, strip, bx0)
        left = gray_drawings(doc[0])
        assert not left, f"{name}: historical-linear artist(s) remain"
        txt = doc[0].get_text()
        for bad in FORBIDDEN_TEXT:
            assert bad not in txt, f"{name}: {bad!r} still present"
        planned.append((p, doc, f"{old_dim[0]:.1f}x{old_dim[1]:.1f} ->"
                                f" {new_dim[0]:.1f}x{new_dim[1]:.1f}", "column"))

    for p, doc, detail, kind in planned:
        print(f"  {p.name:52s} {kind:8s} {detail}  sha_before={sha256(p)[:12]}")

    if not a.apply:
        print("\nverify only; nothing written. Re-run with --apply to rewrite.")
        return 0

    # PyMuPDF refuses a non-incremental save onto the file it opened, so each asset is
    # written beside itself and moved into place only once the save succeeded.
    for p, doc, _, kind in planned:
        tmp = p.with_suffix(".pdf.tmp")
        doc.save(str(tmp), garbage=4, deflate=True)
        doc.close()
        os.replace(tmp, p)
        print(f"  wrote {p}  sha_after={sha256(p)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
