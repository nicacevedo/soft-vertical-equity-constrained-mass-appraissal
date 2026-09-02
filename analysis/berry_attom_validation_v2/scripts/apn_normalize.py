#!/usr/bin/env python3
"""Documented APN normalization. Always preserve raw IDs alongside these keys."""
from __future__ import annotations

import re

import pandas as pd

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_apn_raw(value: str | None) -> str | None:
    """Controlled normalization used for DOCUMENTED_NORMALIZED_APN matches.

    Deterministic: uppercase, strip, drop non-alphanumeric characters.
    Empty after cleaning -> None. Does not pad, unpad, or guess check digits.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE", "NULL", "<NA>"}:
        return None
    s = _NON_ALNUM.sub("", s)
    return s or None


def normalize_apn_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_apn_raw).astype("string")
