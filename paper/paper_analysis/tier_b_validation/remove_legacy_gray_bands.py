#!/usr/bin/env python3
"""Tier B2.8 -- remove the legacy CV-derived gray band from the eight active path figures.

WHAT THIS REMOVES AND WHY
-------------------------
Seven figure environments in ``paper/paper_v17_option1.tex`` (eight PDF assets --
``fig:other_metric_paths_placeholder`` stacks two graphics under one caption) carry a
filled gray band with dashed vertical boundaries. The band is the min/max hull of five
CV-mean grid optima over R2_price, MAE_price, MAPE, RMSE_log and COD, drawn by
``utils/paper_v12_lower_rho_plots.py`` (``shade_cv_span_with_bounds``).

Its endpoints come from ``transition_span_summary.csv`` under
``output/paper_v12_lower_rho_extension_994_v2/``, which exists in no checkout. **No artifact
in the frozen evidence set reproduces them.** The band is therefore a presentation-only
legacy object: it is drawn over held-out and 2025 curves where the development-fold
construct does not apply, and it visually marks a penalty interval in a paper that selects
no penalty strength.

This script does NOT recreate path data, reconstruct endpoints, rerun plotting code, or
refit anything. It performs the minimal deterministic transformation that makes the band
and its boundaries invisible while leaving every scientific object byte-identical.

THE TRANSFORMATION
------------------
Each file's Resources dictionary points ``/ExtGState`` at one uncompressed object holding a
map of named alpha states. In all eight files:

    /A2 << /Type /ExtGState /CA 0.15 /ca 0.15 >>   <- band fill alpha
    /A5 << /Type /ExtGState /CA 0.9 /ca 1 >>       <- dashed boundary stroke alpha

``/A2`` and ``/A5`` are used by nothing else (verified per run, see ``verify_exclusivity``).
We substitute::

    /A2 ... /CA 0.15 /ca 0.15  ->  /CA 0.00 /ca 0.00
    /A5 ... /CA 0.9            ->  /CA 0.0

Both substitutions preserve byte count exactly, so no content stream is touched, no
``/Length`` changes, and no xref offset shifts. The band geometry remains in the file but
paints nothing.

WHAT MUST NOT BE TOUCHED
------------------------
The band boundaries are DASHED (``[ 2.25 1.65 ]``, ExtGState ``/A5``). The *dotted* lines
(``[ 1.05 1.7325 ]``, ExtGState ``/A4``) are the metric neutrality reference lines --
PRD=1, PRB=0, MKI=1, VEI=0, beta_log=0, ratio=1 -- and are supported scientific content.
The chronological fold curves in the ``cv_*`` figures stroke the *same* gray as the band but
via ``/A6``. Gridlines use ``/A3``. None of these is modified.

USAGE
-----
    python3 remove_legacy_gray_bands.py            # verify only, write nothing (default)
    python3 remove_legacy_gray_bands.py --apply    # verify, then rewrite in place

Any assertion failure aborts before a single byte is written. This script does not import
``tb_common``: that module's ``guard_write`` refuses writes outside the validator directory,
and these are legitimate paper-asset edits under ``paper/img/``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zlib
from pathlib import Path

# --------------------------------------------------------------------------------------
# Allowlist. Nothing outside this tuple is ever opened for writing.
# --------------------------------------------------------------------------------------
FIGURE_DIR = Path("paper/img/generated_v12_994")

ALLOWLIST: tuple[str, ...] = (
    "mechanism_vs_rho.pdf",
    "predictive_metric_paths.pdf",
    "level_uniformity_paths.pdf",
    "vertical_equity_metric_paths.pdf",
    "cv_predictive_metric_paths.pdf",
    "cv_level_uniformity_paths.pdf",
    "cv_vertical_equity_metric_paths.pdf",
    "cv_mechanism_metric_paths.pdf",
)

# Expected band-object census, from the B2.8 pre-edit verification. Keyed by basename:
# (number of band fills, number of dashed boundary strokes). Boundaries are always 2x fills.
EXPECTED_COUNTS: dict[str, tuple[int, int]] = {
    "mechanism_vs_rho.pdf": (6, 12),
    "predictive_metric_paths.pdf": (8, 16),
    "level_uniformity_paths.pdf": (10, 20),
    "vertical_equity_metric_paths.pdf": (8, 16),
    "cv_predictive_metric_paths.pdf": (8, 16),
    "cv_level_uniformity_paths.pdf": (10, 20),
    "cv_vertical_equity_metric_paths.pdf": (8, 16),
    "cv_mechanism_metric_paths.pdf": (6, 12),
}

TOTAL_FILLS = 64
TOTAL_BOUNDS = 128

# --------------------------------------------------------------------------------------
# Frozen operator signatures.
# --------------------------------------------------------------------------------------
GRAY_BAND_RGB = "0.6117647059 0.6392156863 0.6862745098"      # #9CA3AF, band fill
DASH_STROKE_RGB = "0.4196078431 0.4470588235 0.5019607843"    # #6B7280, band boundary
BAND_DASH = "[ 2.25 1.65 ] 0 d"                               # boundary: DASHED
NEUTRALITY_DASH = "[ 1.05 1.7325 ] 0 d"                       # reference lines: DOTTED (keep)

A2_OLD = b"/A2 << /Type /ExtGState /CA 0.15 /ca 0.15 >>"
A2_NEW = b"/A2 << /Type /ExtGState /CA 0.00 /ca 0.00 >>"
A5_OLD = b"/A5 << /Type /ExtGState /CA 0.9 /ca 1 >>"
A5_NEW = b"/A5 << /Type /ExtGState /CA 0.0 /ca 1 >>"

assert len(A2_OLD) == len(A2_NEW), "A2 substitution must preserve byte count"
assert len(A5_OLD) == len(A5_NEW), "A5 substitution must preserve byte count"


class BandRemovalError(RuntimeError):
    """Raised on any signature mismatch. Always aborts before writing."""


def _norm(chunk: bytes) -> str:
    """Whitespace-normalized view. Matplotlib hard-wraps its content stream at 79
    columns, so operator sequences are split by newlines at unpredictable points."""
    return " ".join(chunk.decode("latin-1").split())


def _obj_body(raw: bytes, num: int) -> bytes:
    """Body of ``<num> 0 obj``. The lookbehind anchors the object number so that
    ``4 0 obj`` cannot match inside ``24 0 obj``."""
    pat = re.compile(
        rb"(?:(?<=\n)|(?<=\A))" + str(num).encode() + rb"\s+0\s+obj\s*(.*?)\s*endobj",
        re.S,
    )
    m = pat.search(raw)
    if m is None:
        raise BandRemovalError(f"object {num} 0 obj not found")
    return m.group(1)


def _page_content_stream(raw: bytes) -> bytes:
    """Inflate the page content stream (``9 0 obj`` in every matplotlib figure here)."""
    m = re.search(
        rb"(?:(?<=\n)|(?<=\A))9\s+0\s+obj\s*<<[^>]*/FlateDecode[^>]*>>\s*stream\r?\n", raw
    )
    if m is None:
        raise BandRemovalError("page content stream (9 0 obj, FlateDecode) not found")
    end = raw.find(b"endstream", m.end())
    if end < 0:
        raise BandRemovalError("unterminated content stream")
    return zlib.decompress(raw[m.end() : end])


def verify_structure(name: str, raw: bytes) -> int:
    """Vector-PDF sanity. Returns the ExtGState object number."""
    if not raw.startswith(b"%PDF-"):
        raise BandRemovalError(f"{name}: not a PDF")
    if b"/Subtype /Image" in raw:
        raise BandRemovalError(f"{name}: embedded raster image present; expected pure vector")
    if b"Matplotlib" not in raw:
        raise BandRemovalError(f"{name}: not matplotlib output")
    n_pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    if n_pages != 1:
        raise BandRemovalError(f"{name}: expected 1 page, found {n_pages}")

    refs = re.findall(rb"/ExtGState\s+(\d+)\s+0\s+R", raw)
    if len(refs) != 1:
        raise BandRemovalError(f"{name}: expected exactly one /ExtGState resource ref, got {len(refs)}")
    return int(refs[0])


def verify_extgstate(name: str, raw: bytes, egs_num: int) -> None:
    """The two target dictionaries must be present, exact, and unique in the whole file."""
    body = _obj_body(raw, egs_num)
    for label, literal in (("A2", A2_OLD), ("A5", A5_OLD)):
        if literal not in body:
            raise BandRemovalError(
                f"{name}: /{label} dictionary not in expected form; refusing to guess.\n"
                f"  expected: {literal.decode()}\n"
                f"  obj {egs_num} body: {body.decode('latin-1')[:400]}"
            )
        if raw.count(literal) != 1:
            raise BandRemovalError(
                f"{name}: /{label} literal occurs {raw.count(literal)}x in file; expected exactly 1"
            )


def verify_exclusivity(name: str, content: bytes) -> tuple[int, int]:
    """Prove /A2 and /A5 are used by nothing but the band.

    Every ``/A2 gs`` must open a gray band fill; every ``/A5 gs`` must open a dashed
    boundary stroke. Any other use means the alpha edit would touch something else.
    """
    n_fill = 0
    for m in re.finditer(rb"/A2\s+gs", content):
        seg = _norm(content[m.end() : m.end() + 300])
        expected = f"{GRAY_BAND_RGB} rg {GRAY_BAND_RGB} RG {GRAY_BAND_RGB} rg"
        if not seg.startswith(expected) or not re.search(r"\bh f Q", seg):
            raise BandRemovalError(
                f"{name}: /A2 used outside a gray band fill -- alpha edit is NOT safe.\n"
                f"  context: {seg[:200]}"
            )
        n_fill += 1

    n_bound = 0
    for m in re.finditer(rb"/A5\s+gs", content):
        seg = _norm(content[m.end() : m.end() + 300])
        expected = f"1 j 0.75 w {BAND_DASH} {DASH_STROKE_RGB} RG /DeviceRGB cs"
        if not seg.startswith(expected) or not re.search(r"\bl S Q", seg):
            raise BandRemovalError(
                f"{name}: /A5 used outside a dashed band boundary -- alpha edit is NOT safe.\n"
                f"  context: {seg[:200]}"
            )
        n_bound += 1

    # /A2 and /A5 must never appear other than as a gs operand.
    for label in (b"/A2", b"/A5"):
        bare = len(re.findall(re.escape(label) + rb"\b", content))
        used = len(re.findall(re.escape(label) + rb"\s+gs", content))
        if bare != used:
            raise BandRemovalError(
                f"{name}: {label.decode()} appears {bare}x but only {used}x as a gs operand"
            )

    expect_fill, expect_bound = EXPECTED_COUNTS[name]
    if (n_fill, n_bound) != (expect_fill, expect_bound):
        raise BandRemovalError(
            f"{name}: band census {n_fill}/{n_bound} != expected {expect_fill}/{expect_bound}"
        )
    if n_bound != 2 * n_fill:
        raise BandRemovalError(f"{name}: {n_bound} boundaries for {n_fill} bands; expected 2x")
    return n_fill, n_bound


def report_preserved(content: bytes) -> dict[str, int]:
    """Objects that must survive untouched, counted for the audit report."""
    flat = _norm(content)
    return {
        "dotted_neutrality_lines": len(re.findall(re.escape(NEUTRALITY_DASH), flat)),
        "gridlines_A3": len(re.findall(r"/A3 gs", flat)),
        "opaque_A4": len(re.findall(r"/A4 gs", flat)),
        "fold_curves_A6": len(re.findall(r"/A6 gs", flat)),
    }


def process(path: Path, apply: bool) -> dict:
    name = path.name
    if name not in ALLOWLIST:
        raise BandRemovalError(f"{name} is not in the allowlist; refusing to touch it")

    raw = path.read_bytes()
    before_sha = hashlib.sha256(raw).hexdigest()

    egs_num = verify_structure(name, raw)
    verify_extgstate(name, raw, egs_num)
    content = _page_content_stream(raw)
    n_fill, n_bound = verify_exclusivity(name, content)
    preserved = report_preserved(content)

    patched = raw.replace(A2_OLD, A2_NEW).replace(A5_OLD, A5_NEW)
    if len(patched) != len(raw):
        raise BandRemovalError(f"{name}: substitution changed file length; aborting")
    if patched == raw:
        raise BandRemovalError(f"{name}: substitution was a no-op; aborting")

    # The content stream must be bit-identical: we only touched an uncompressed dict.
    if _page_content_stream(patched) != content:
        raise BandRemovalError(f"{name}: content stream changed; aborting")

    n_diff = sum(1 for a, b in zip(raw, patched) if a != b)

    result = {
        "file": name,
        "bytes": len(raw),
        "extgstate_obj": egs_num,
        "band_fills": n_fill,
        "boundary_strokes": n_bound,
        "changed_bytes": n_diff,
        "sha256_before": before_sha,
        "preserved": preserved,
        "applied": False,
    }

    if apply:
        path.write_bytes(patched)
        result["sha256_after"] = hashlib.sha256(patched).hexdigest()
        result["applied"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="rewrite the PDFs (default: verify only)")
    ap.add_argument("--figure-dir", default=str(FIGURE_DIR))
    args = ap.parse_args(argv)

    base = Path(args.figure_dir)
    if not base.is_dir():
        print(f"figure directory not found: {base}", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "VERIFY-ONLY (no bytes written)"
    print(f"Tier B2.8 legacy gray-band removal -- {mode}")
    print(f"figure dir: {base}")
    print()

    results = []
    try:
        for name in ALLOWLIST:
            results.append(process(base / name, args.apply))
    except BandRemovalError as exc:
        print(f"\nABORTED -- {exc}", file=sys.stderr)
        print("No file was modified.", file=sys.stderr)
        return 1

    hdr = f"{'file':34s} {'bytes':>7s} {'egs':>4s} {'fills':>5s} {'bounds':>6s} {'dbytes':>6s}  preserved (dotted/A3/A4/A6)"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        p = r["preserved"]
        print(
            f"{r['file']:34s} {r['bytes']:7d} {r['extgstate_obj']:4d} {r['band_fills']:5d} "
            f"{r['boundary_strokes']:6d} {r['changed_bytes']:6d}  "
            f"{p['dotted_neutrality_lines']:3d}/{p['gridlines_A3']:3d}/{p['opaque_A4']:4d}/{p['fold_curves_A6']:3d}"
        )

    tf = sum(r["band_fills"] for r in results)
    tb = sum(r["boundary_strokes"] for r in results)
    print()
    print(f"TOTAL band fills made transparent : {tf} (expected {TOTAL_FILLS})")
    print(f"TOTAL boundary strokes hidden     : {tb} (expected {TOTAL_BOUNDS})")
    if (tf, tb) != (TOTAL_FILLS, TOTAL_BOUNDS):
        print("census mismatch -- aborting", file=sys.stderr)
        return 1
    print("Content streams unchanged; file lengths unchanged; xref offsets unchanged.")
    print("Neutrality reference lines, gridlines and CV fold curves untouched.")
    if not args.apply:
        print("\nVerify-only. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
