# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Highest Density Interval (HDI)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def bayesian_hdi(
    posterior_samples: np.ndarray | list,
    *,
    prob: float = 0.95,
) -> ESRes:
    """
    Compute the Highest Density Interval (HDI) from posterior samples.

    The HDI is the narrowest interval containing *prob* fraction of
    the posterior mass.

    Parameters
    ----------
    posterior_samples : array-like
        MCMC or other posterior samples.
    prob : float
        Probability mass for the interval.

    Returns
    -------
    ESRes
        estimate = interval width, ci_lower/ci_upper = HDI bounds.

    References
    ----------
    Kruschke, J. K. (2015). *Doing Bayesian Data Analysis*, 2nd ed.
    Academic Press, Ch. 25.
    """
    samples = np.sort(np.asarray(posterior_samples, dtype=float))
    n = len(samples)
    if n < 3:
        raise ValueError("Need at least 3 samples.")
    if not (0 < prob < 1):
        raise ValueError("prob must be in (0, 1).")

    interval_width = int(np.ceil(prob * n))
    if interval_width >= n:
        interval_width = n - 1

    widths = samples[interval_width:] - samples[: n - interval_width]
    best = int(np.argmin(widths))

    hdi_lo = float(samples[best])
    hdi_hi = float(samples[best + interval_width])

    return ESRes(
        measure="HDI",
        estimate=float(hdi_hi - hdi_lo),
        ci_lower=hdi_lo,
        ci_upper=hdi_hi,
        n=n,
        extra={"prob": prob},
    )


bhdi = bayesian_hdi


def cheatsheet() -> str:
    return "bayesian_hdi({}) -> Highest Density Interval (HDI)."
