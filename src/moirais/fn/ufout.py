"""Unfolding outlier detection. 'Big Bang Attack!' -- Vegeta, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def unfolding_outliers(residuals, threshold=2.0):
    """Identify outliers in unfolding residuals.

    Parameters
    ----------
    residuals : array-like
        Residual matrix or vector.
    threshold : float
        Number of standard deviations for outlier cutoff.

    Returns
    -------
    DescriptiveResult
        value = outlier indices (list of tuples or ints).
    """
    import numpy as np

    R = np.asarray(residuals, dtype=float)
    mu = np.mean(R)
    sd = np.std(R)
    if sd < 1e-12:
        return DescriptiveResult(name="unfolding_outliers", value=[], extra={"n_outliers": 0, "threshold": threshold})

    z = np.abs((R - mu) / sd)
    if R.ndim == 2:
        outliers = list(zip(*np.where(z > threshold)))
    else:
        outliers = list(np.where(z > threshold)[0])
    return DescriptiveResult(
        name="unfolding_outliers",
        value=outliers,
        extra={"n_outliers": len(outliers), "threshold": threshold, "mean": float(mu), "std": float(sd)},
    )


ufout = unfolding_outliers


def cheatsheet() -> str:
    return "unfolding_outliers({}) -> Unfolding outlier detection. 'Big Bang Attack!' -- Vegeta, D"
