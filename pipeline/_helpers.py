"""
Small shared helpers for the numbered pipeline scripts.

Keep this module lightweight: path constants, optional JSON context passing
between stages, and discovery of the latest CV run under ``result_root``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_TRAINING_PARQUET = Path("data/CCAO/2025/training_data.parquet")
DEFAULT_RESULT_ROOT = Path("./output/robust_rolling_origin_cv")
CONTEXT_FILENAME = "pipeline_last_context.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def context_path() -> Path:
    return repo_root() / "pipeline" / CONTEXT_FILENAME


def write_context(payload: Dict[str, Any]) -> Path:
    path = context_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)
    return path


def read_context() -> Dict[str, Any]:
    path = context_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_latest_completed_run(result_root: Path) -> Optional[Tuple[str, str, Path]]:
    """
    Find the most recently modified ``test_eval_status.json`` under
    ``result_root / analysis / data_id=* / split_id=*``.

    Returns (data_id, split_id, status_json_path) or None if nothing matches.
    """
    analysis = Path(result_root) / "analysis"
    if not analysis.is_dir():
        return None
    best: Optional[Tuple[float, str, str, Path]] = None
    for status_path in analysis.glob("data_id=*/split_id=*/test_eval_status.json"):
        try:
            mtime = status_path.stat().st_mtime
        except OSError:
            continue
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in status_path.parts if "=" in p}
        data_id = parts.get("data_id")
        split_id = parts.get("split_id")
        if not data_id or not split_id:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, str(data_id), str(split_id), status_path)
    if best is None:
        return None
    return best[1], best[2], best[3]


def run_repo_script(script_name: str, argv_tail: list[str], *, cwd: Optional[Path] = None) -> None:
    """Run ``python <repo>/<script_name>`` with the given extra CLI args."""
    root = repo_root()
    script = root / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script), *argv_tail]
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "default")
    subprocess.run(cmd, cwd=str(cwd or root), check=True, env=env)


def run_repo_script_capture(
    script_name: str,
    argv_tail: list[str],
    *,
    cwd: Optional[Path] = None,
) -> str:
    """
    Same as ``run_repo_script`` but tees the merged stdout/stderr to the parent
    terminal *and* returns the full output text. Used by stages that must parse
    identifiers (e.g. ``data_id`` / ``split_id``) emitted by the underlying
    script.
    """
    root = repo_root()
    script = root / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script), *argv_tail]
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "default")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return "".join(captured)


def extract_arg_value(argv: list[str], flag: str, *, default: Optional[str] = None) -> Optional[str]:
    """
    Find the value for ``--flag`` in an argv list, supporting both
    ``--flag value`` and ``--flag=value`` forms. Returns ``default`` if absent.
    """
    eq_prefix = f"{flag}="
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(eq_prefix):
            return token[len(eq_prefix) :]
    return default


def parse_data_split_ids(
    *,
    data_id: Optional[str],
    split_id: Optional[str],
    result_root: Path,
    prefer_context: bool,
) -> Tuple[str, str]:
    """Resolve identifiers from explicit args, saved context, or latest run."""
    if data_id and split_id:
        return str(data_id), str(split_id)
    if prefer_context:
        ctx = read_context()
        cid = ctx.get("data_id")
        sid = ctx.get("split_id")
        if cid and sid:
            return str(cid), str(sid)
    found = discover_latest_completed_run(result_root)
    if found:
        return found[0], found[1]
    raise ValueError(
        "Could not resolve data_id / split_id. Run train first, pass --data-id/--split-id, "
        f"or ensure {context_path()} lists data_id and split_id."
    )
