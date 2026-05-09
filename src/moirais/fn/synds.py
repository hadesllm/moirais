"""Syndromic surveillance (EARS algorithm)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def syndromic_surveillance(
    counts: list[int] | np.ndarray,
    method: str = "C2",
    baseline_days: int = 7,
    threshold: float = 2.0,
) -> ESRes:
    """Syndromic surveillance using the EARS (Early Aberration
    Reporting System) C1/C2/C3 algorithms.

    Parameters
    ----------
    counts : array-like of int
        Daily syndrome counts.
    method : str, default 'C2'
        EARS method: 'C1', 'C2', or 'C3'.
    baseline_days : int, default 7
        Number of baseline days for mean/SD calculation.
    threshold : float, default 2.0
        Alert threshold in standard deviations.

    Returns
    -------
    ESRes
        estimate is number of alerts detected.

    References
    ----------
    Hutwagner, L. et al. (2003). The bioterrorism preparedness and
    response Early Aberration Reporting System (EARS). Journal of
    Urban Health, 80(suppl 1), i89-i96.
    """
    c = np.asarray(counts, dtype=float)
    n = len(c)
    if n < baseline_days + 2:
        raise ValueError(f"Need at least {baseline_days + 2} days of data")

    gap = 2 if method in ("C2", "C3") else 0
    alerts = []
    test_stats = np.zeros(n)

    for t in range(baseline_days + gap, n):
        start = t - baseline_days - gap
        end = t - gap
        baseline = c[start:end]
        mu = np.mean(baseline)
        sigma = np.std(baseline, ddof=1)

        if sigma > 0:
            test_stats[t] = (c[t] - mu) / sigma
        else:
            test_stats[t] = 0.0

        if test_stats[t] > threshold:
            alerts.append(t)

    if method == "C3" and len(alerts) > 0:
        c3_stat = np.zeros(n)
        for t in range(baseline_days + gap + 2, n):
            c3_stat[t] = max(0, test_stats[t] - threshold) + max(0, c3_stat[t - 1])
        alerts_c3 = [t for t in range(n) if c3_stat[t] > threshold]
        alerts = alerts_c3

    return ESRes(
        measure="syndromic_surveillance",
        estimate=float(len(alerts)),
        n=n,
        extra={
            "method": method,
            "alerts": alerts,
            "test_statistics": test_stats.tolist(),
            "threshold": threshold,
            "baseline_days": baseline_days,
        },
    )


synds = syndromic_surveillance


def cheatsheet() -> str:
    return "syndromic_surveillance({}) -> EARS syndromic surveillance algorithm."
