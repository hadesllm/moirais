"""Single-occasion acute harm risk from BAC levels."""

import numpy as np

from ._containers import DescriptiveResult


def single_use_risk(
    bac_levels: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse single-occasion risk from blood alcohol concentration.

    Risk categories: <0.05 low, 0.05-0.08 moderate, 0.08-0.15 high,
    0.15-0.30 very high, >0.30 life-threatening.

    Parameters
    ----------
    bac_levels : array-like
        BAC levels for individuals on a single occasion.

    Returns
    -------
    DescriptiveResult
    """
    b = np.asarray(bac_levels, dtype=float)
    b = b[~np.isnan(b)]
    if len(b) == 0:
        raise ValueError("No valid BAC levels")

    n = len(b)
    cats = {
        "low": float(np.sum(b < 0.05) / n * 100),
        "moderate": float(np.sum((b >= 0.05) & (b < 0.08)) / n * 100),
        "high": float(np.sum((b >= 0.08) & (b < 0.15)) / n * 100),
        "very_high": float(np.sum((b >= 0.15) & (b < 0.30)) / n * 100),
        "life_threatening": float(np.sum(b >= 0.30) / n * 100),
    }

    return DescriptiveResult(
        name="single_use_risk",
        value=float(np.mean(b)),
        extra={"risk_categories_pct": cats, "n": n, "max_bac": float(np.max(b))},
    )


susngl = single_use_risk


def cheatsheet() -> str:
    return "single_use_risk({}) -> Single-occasion acute harm risk from BAC levels."
