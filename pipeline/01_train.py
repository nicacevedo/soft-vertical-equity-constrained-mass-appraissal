#!/usr/bin/env python3
"""
Train (CCAO ``01-train`` analog).

Cook County trains baselines + tuned LightGBM, writes workflows and holdout test
predictions for monitoring.

This repository performs **rolling-origin temporal CV**, optional bootstrap
metrics, and held-out evaluation via::

  run_temporal_cv.py

This wrapper forwards CLI arguments to that script (same interface) and records
``data_id`` / ``split_id`` in ``pipeline/pipeline_last_context.json`` so later
stages can resolve paths without copying identifiers. Identifiers are captured
from the underlying script's stdout (the printed ``data_id=… | split_id=…``
line) and then double-checked against the latest ``test_eval_status.json``.

Usage::

  python pipeline/01_train.py
  python pipeline/01_train.py --parallel --sample-frac 0.2
  python pipeline/01_train.py --result-root ./my_out --data-path data/CCAO/2025/training_data.parquet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import (
    DEFAULT_RESULT_ROOT,
    discover_latest_completed_run,
    extract_arg_value,
    run_repo_script_capture,
    write_context,
)


_ID_LINE_RE = re.compile(r"data_id=(\S+)\s*\|\s*split_id=(\S+)")


def _parse_ids_from_stdout(text: str) -> tuple[str | None, str | None]:
    matches = _ID_LINE_RE.findall(text)
    if not matches:
        return None, None
    data_id, split_id = matches[-1]
    return data_id.strip(), split_id.strip()


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        argv = argv[:i] + argv[i + 1 :]

    output = run_repo_script_capture("run_temporal_cv.py", argv)

    result_root_raw = extract_arg_value(argv, "--result-root", default=str(DEFAULT_RESULT_ROOT))
    result_root = Path(result_root_raw).resolve()

    data_id, split_id = _parse_ids_from_stdout(output)
    status_path: Path | None = None
    if data_id and split_id:
        candidate = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "test_eval_status.json"
        status_path = candidate if candidate.is_file() else None
    else:
        found = discover_latest_completed_run(result_root)
        if found:
            data_id, split_id, status_path = found

    if not (data_id and split_id):
        print(
            "[pipeline 01_train] WARNING: could not determine data_id / split_id from "
            "run_temporal_cv.py output or analysis directory. Context file not updated.",
            flush=True,
        )
        return

    payload = {
        "stage": "train",
        "data_id": data_id,
        "split_id": split_id,
        "result_root": str(result_root),
        "test_eval_status_json": str(status_path) if status_path else None,
    }
    out = write_context(payload)
    print(f"[pipeline 01_train] wrote context: {out}", flush=True)
    print(f"[pipeline 01_train] data_id={data_id} split_id={split_id}", flush=True)


if __name__ == "__main__":
    main()
