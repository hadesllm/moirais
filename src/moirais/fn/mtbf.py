# moirais.fn — function file (hadesllm/moirais)
"""Mean Time Between Failures. 'Real knowledge is to know the extent of one's ignorance. -- Confucius'"""
from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mtbf_estimate(
    failure_times: np.ndarray,
    total_time: float | None = None,
) -> DescriptiveResult:
    r"""Estimate Mean Time Between Failures (MTBF).

    For repairable systems, MTBF is the expected operating time
    between consecutive failures:

    .. math::

        \widehat{MTBF} = \frac{T}{n}

    where *T* is total operating time and *n* is number of failures.

    Parameters
    ----------
    failure_times : ndarray
        Times at which failures occurred (cumulative or inter-arrival).
    total_time : float or None
        Total observation time. If None, uses max(failure_times).

    Returns
    -------
    DescriptiveResult
        name='MTBF', value=MTBF estimate,
        extra has 'mtbf', 'failure_rate', 'n_failures',
        'total_time', 'ci_lo', 'ci_hi' (95% chi-squared CI).

    References
    ----------
    O'Connor, P.D.T. & Kleyner, A. (2012). *Practical Reliability
    Engineering* (5th ed.). Wiley. Ch. 5.
    """
    ft = np.sort(np.asarray(failure_times, dtype=np.float64).ravel())
    n = len(ft)

    if n == 0:
        return DescriptiveResult(
            name="MTBF",
            value=float("inf"),
            extra={"mtbf": float("inf"), "failure_rate": 0.0, "n_failures": 0, "total_time": total_time or 0.0},
        )

    T = total_time if total_time is not None else float(ft[-1])

    mtbf_val = T / n
    failure_rate = n / T if T > 0 else float("inf")

    from scipy.stats import chi2

    ci_lo = 2 * T / chi2.ppf(0.975, 2 * n + 2) if n > 0 else 0.0
    ci_hi = 2 * T / chi2.ppf(0.025, max(2 * n, 2)) if n > 0 else float("inf")

    return DescriptiveResult(
        name="MTBF",
        value=float(mtbf_val),
        extra={
            "mtbf": float(mtbf_val),
            "failure_rate": float(failure_rate),
            "n_failures": n,
            "total_time": T,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        },
    )


def cheatsheet() -> str:
    return "mtbf_estimate({}) -> Mean Time Between Failures. 'Never underestimate a droid.' -"
