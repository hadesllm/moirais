# moirais.fn — function file (hadesllm/moirais)
"""R v Jordan compliance (18/30 month ceiling)."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import CrimeResult


def court_jordan(
    days_to_trial: np.ndarray | list[float],
    *,
    ceiling_months: float = 18.0,
) -> CrimeResult:
    """Assess R v Jordan compliance rate.

    R v Jordan [2016] 1 SCR 631 sets presumptive ceilings:
    18 months for provincial court, 30 months for superior court.

    Parameters
    ----------
    days_to_trial : array-like
        Days from charge to completion.
    ceiling_months : float
        Presumptive ceiling in months (18 or 30).

    Returns
    -------
    CrimeResult
    """
    d = np.asarray(days_to_trial, dtype=float)
    d = d[np.isfinite(d) & (d >= 0)]
    if len(d) == 0:
        raise ValueError("No valid durations")
    ceiling_days = ceiling_months * 30.44
    n_over = int(np.sum(d > ceiling_days))
    rate = n_over / len(d)
    return CrimeResult(
        name="jordan_non_compliance",
        rate=rate,
        n=n_over,
        population=len(d),
        extra={
            "ceiling_months": ceiling_months,
            "ceiling_days": ceiling_days,
            "median_days": float(np.median(d)),
            "pct_over_ceiling": rate,
        },
    )


crtjr = court_jordan


def cheatsheet() -> str:
    return "court_jordan({}) -> R v Jordan compliance (18/30 month ceiling)."
