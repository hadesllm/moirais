# morie.fn — function file (hadesllm/morie)
"""Waitlist analysis (instrumental for causal)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def program_waitlist(
    df,
    *,
    waitlist_col: str = "waitlisted",
    outcome_col: str = "outcome",
    treatment_col: str = "enrolled",
) -> ESRes:
    """Estimate program effect using waitlist as instrument (Wald estimator).

    Parameters
    ----------
    df : DataFrame
    waitlist_col : str
        Binary instrument (1 = on waitlist / offered).
    outcome_col : str
    treatment_col : str

    Returns
    -------
    ESRes
    """
    for c in [waitlist_col, outcome_col, treatment_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")
    z = df[waitlist_col].values.astype(float)
    y = df[outcome_col].values.astype(float)
    d = df[treatment_col].values.astype(float)

    cov_zy = np.mean(z * y) - np.mean(z) * np.mean(y)
    cov_zd = np.mean(z * d) - np.mean(z) * np.mean(d)
    if abs(cov_zd) < 1e-12:
        raise ValueError("Instrument has no effect on treatment (weak instrument)")
    wald = float(cov_zy / cov_zd)
    n = len(y)
    se = float(np.std(y, ddof=1) / (abs(cov_zd) * np.sqrt(n)))
    return ESRes(
        measure="waitlist_iv_late",
        estimate=wald,
        ci_lower=wald - 1.96 * se,
        ci_upper=wald + 1.96 * se,
        se=se,
        n=n,
        extra={"cov_zy": float(cov_zy), "cov_zd": float(cov_zd)},
    )


prgwt = program_waitlist


def cheatsheet() -> str:
    return "program_waitlist({}) -> Waitlist analysis (instrumental for causal)."
