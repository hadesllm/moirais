"""Input validation / sanitization metrics. 'I protect that which matters most.' -- Seraph"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def validate_inputs(
    data: pd.DataFrame,
    *,
    rules: dict | None = None,
) -> DescriptiveResult:
    """Score data quality by checking common sanitization issues.

    Checks per column: missing values, out-of-range numerics, whitespace-only
    strings, duplicate rows, and type consistency. Optionally applies user
    rules (min/max bounds, regex patterns).

    Parameters
    ----------
    data : DataFrame
        Input data to validate.
    rules : dict or None
        Optional per-column rules. Format::

            {"col": {"min": 0, "max": 100, "pattern": r"^[A-Z]"}}

    Returns
    -------
    DescriptiveResult
        ``value`` is the overall quality score (0--1); ``extra`` has
        per-column violation counts.
    """
    n_rows, n_cols = data.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("DataFrame must have at least 1 row and 1 column")

    if rules is None:
        rules = {}

    violations = {}
    total_checks = 0
    total_pass = 0

    for col in data.columns:
        col_v = {}
        s = data[col]

        n_missing = int(s.isna().sum())
        col_v["missing"] = n_missing
        total_checks += len(s)
        total_pass += len(s) - n_missing

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            n_inf = int(np.isinf(clean.to_numpy(dtype=float)).sum()) if len(clean) > 0 else 0
            col_v["inf"] = n_inf
            total_checks += len(clean)
            total_pass += len(clean) - n_inf

            if col in rules:
                r = rules[col]
                if "min" in r:
                    n_below = int((clean < r["min"]).sum())
                    col_v["below_min"] = n_below
                    total_checks += len(clean)
                    total_pass += len(clean) - n_below
                if "max" in r:
                    n_above = int((clean > r["max"]).sum())
                    col_v["above_max"] = n_above
                    total_checks += len(clean)
                    total_pass += len(clean) - n_above
        elif pd.api.types.is_string_dtype(s):
            clean = s.dropna().astype(str)
            n_ws = int((clean.str.strip() == "").sum())
            col_v["whitespace_only"] = n_ws
            total_checks += len(clean)
            total_pass += len(clean) - n_ws

            if col in rules and "pattern" in rules[col]:
                pat = rules[col]["pattern"]
                n_mismatch = int((~clean.str.match(pat)).sum())
                col_v["pattern_mismatch"] = n_mismatch
                total_checks += len(clean)
                total_pass += len(clean) - n_mismatch

        violations[col] = col_v

    n_dup = int(data.duplicated().sum())
    quality = total_pass / total_checks if total_checks > 0 else 1.0

    return DescriptiveResult(
        name="InputValidation",
        value=round(quality, 4),
        extra={
            "violations": violations,
            "duplicate_rows": n_dup,
            "total_checks": total_checks,
            "total_pass": total_pass,
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
    )


sraph = validate_inputs


def cheatsheet() -> str:
    return "validate_inputs({}) -> Input validation / sanitization metrics. 'I protect that whi"
