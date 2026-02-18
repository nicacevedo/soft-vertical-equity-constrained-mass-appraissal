"""
Entry point alias for the full CV pipeline.

This file exists to make the pipeline easier to discover:
  - `run_temporal_cv.py` runs the full rolling-origin CV + test evaluation.
  - The legacy entry point `main.py` is still supported.
"""

from __future__ import annotations

from main import run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()

