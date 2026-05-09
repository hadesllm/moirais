# moirais.fn — function file (hadesllm/moirais)
"""Occupational injury rate per 100 FTE."""

import scipy.stats as stats

from ._containers import ESRes


def occupational_injury_rate(
    n_injuries: int,
    n_fte: float,
    hours_per_fte: float = 2000.0,
    confidence: float = 0.95,
) -> ESRes:
    """Compute occupational injury rate per 100 full-time equivalents.

    Parameters
    ----------
    n_injuries : int
    n_fte : float
        Number of full-time equivalent workers.
    hours_per_fte : float
        Hours per FTE per year (default 2000).
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_fte <= 0:
        raise ValueError("n_fte must be positive")

    rate = n_injuries / n_fte * 100
    total_hours = n_fte * hours_per_fte
    alpha = 1 - confidence
    ci_lo = stats.chi2.ppf(alpha / 2, 2 * n_injuries) / (2 * n_fte) * 100 if n_injuries > 0 else 0.0
    ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (n_injuries + 1)) / (2 * n_fte) * 100

    return ESRes(
        measure="injury_rate_per_100_fte",
        estimate=float(rate),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n_injuries,
        extra={"total_hours": float(total_hours)},
    )


occrt = occupational_injury_rate


def cheatsheet() -> str:
    return "occupational_injury_rate({}) -> Occupational injury rate per 100 FTE."
