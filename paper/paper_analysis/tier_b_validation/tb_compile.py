#!/usr/bin/env python3
"""Compile the manuscript and read the log for the things that actually matter.

``latexmk`` exiting 0 is necessary and not sufficient. Three failure modes are
silent in the exit code and load-bearing for this revision:

* **Undefined references.** B1.1 deletes tables and figures. If any surviving
  passage still ``\\ref``s one of them the reference resolves to ``??``, which
  is a broken document, not a warning to tolerate. Note that a ``\\ref`` inside
  an ``oldrevisionblock`` DOES resolve -- the Tier-A redefinition typesets the
  body into a discarded box rather than gobbling it -- so a suppressed sentence
  can still break a reference. ``\\oldtext``, by contrast, gobbles its argument
  and cannot.
* **Undefined citations.** B1.2 adds ten citation keys. A key that is cited but
  not present in a loaded ``.bib`` prints nothing at all, which is precisely the
  pre-existing defect this stage repairs.
* **Overfull boxes introduced by this pass.** The baseline already has some, so
  the useful quantity is the DELTA against the recorded baseline, not the count.

Nothing here alters scientific content to silence a formatting warning; the
report distinguishes structural errors, which must be fixed, from cosmetic
warnings, which are reported and left alone.

The build runs in a scratch output directory so no ``.aux``/``.log`` litter
lands in ``paper/`` and no stray file can violate the write-scope check.
"""
from __future__ import annotations

import sys

# Set before ANY project import: running the gate must not leave a stray
# __pycache__ anywhere, least of all under analysis/.
sys.dont_write_bytecode = True

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import tb_common as tb


def compile_manuscript(outdir=None, keep: bool = False) -> dict:
    tmp = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix="tierb_latexmk_"))
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error",
           f"-outdir={tmp}", "-cd", str(tb.TEX)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tb.REPO)
    log_path = tmp / (tb.TEX.stem + ".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") \
        if log_path.exists() else ""
    blg_path = tmp / (tb.TEX.stem + ".blg")
    blg = blg_path.read_text(encoding="utf-8", errors="replace") \
        if blg_path.exists() else ""

    undefined_refs = sorted(set(re.findall(
        r"Reference `([^']+)' on page \d+ undefined", log)))
    undefined_cites = sorted(set(re.findall(
        r"Citation `([^']+)' on page \d+ undefined", log)))
    # biblatex reports a key present in no datasource separately
    missing_bib = sorted(set(re.findall(
        r"I didn't find a database entry for '([^']+)'", log)
        + re.findall(r"WARN - I didn't find a database entry for '([^']+)'", blg)))
    errors = [l for l in log.split("\n")
              if re.match(r"^(?:\./)?[^:]*:\d+:", l) or l.startswith("! ")]
    overfull_h = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\) "
                            r"in paragraph at lines (\d+)--(\d+)", log)
    overfull_v = re.findall(r"Overfull \\vbox \(([\d.]+)pt too high\)", log)
    underfull = len(re.findall(r"Underfull \\[hv]box", log))
    latex_warnings = sorted(set(
        w.strip() for w in re.findall(r"^LaTeX Warning: (.+)$", log, re.M)))
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    biber_warn = sorted(set(re.findall(r"^WARN - (.+)$", blg, re.M)))

    result = {
        "exit_code": r.returncode,
        "pdf_written": (tmp / (tb.TEX.stem + ".pdf")).exists(),
        "pages": int(pages.group(1)) if pages else None,
        "undefined_references": undefined_refs,
        "undefined_citations": undefined_cites,
        "missing_bib_entries": missing_bib,
        "latex_errors": errors[:40],
        "n_latex_errors": len(errors),
        "latex_warnings": [w for w in latex_warnings
                           if "undefined" not in w.lower()
                           and "rerun" not in w.lower()],
        "biber_warnings": biber_warn,
        "overfull_hbox": [{"pt": float(a), "from": int(b), "to": int(c)}
                          for a, b, c in overfull_h],
        "n_overfull_hbox": len(overfull_h),
        "n_overfull_vbox": len(overfull_v),
        "n_underfull": underfull,
        "outdir": str(tmp),
        "stdout_tail": r.stdout[-2000:],
    }
    result["ok"] = (r.returncode == 0 and result["pdf_written"]
                    and not undefined_refs and not undefined_cites
                    and not missing_bib and not errors)
    if not keep and not outdir:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


if __name__ == "__main__":
    import json
    res = compile_manuscript()
    res.pop("stdout_tail", None)
    print(json.dumps(res, indent=2, sort_keys=True))
