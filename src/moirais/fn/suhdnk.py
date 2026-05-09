"""Heavy/binge drinking prevalence."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def heavy_drinking(
    drinks_per_occasion: list | np.ndarray,
    threshold: int = 5,
    confidence: float = 0.95,
) -> ESRes:
    """Estimate heavy/binge drinking prevalence.

    Binge defined as >= *threshold* drinks on a single occasion.

    Parameters
    ----------
    drinks_per_occasion : array-like
        Number of drinks per occasion for each respondent.
    threshold : int
        Binge drinking threshold (default 5 for males, use 4 for females).
    confidence : float

    Returns
    -------
    ESRes
    """
    d = np.asarray(drinks_per_occasion, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        raise ValueError("No valid data")

    n_heavy = int(np.sum(d >= threshold))
    p = n_heavy / n
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    return ESRes(
        measure="heavy_drinking_prevalence",
        estimate=float(p),
        ci_lower=float(max(0, p - z * se)),
        ci_upper=float(min(1, p + z * se)),
        se=float(se),
        n=n,
        extra={"n_heavy": n_heavy, "threshold": threshold, "mean_drinks": float(np.mean(d))},
    )


suhdnk = heavy_drinking


def cheatsheet() -> str:
    return "heavy_drinking({}) -> Heavy/binge drinking prevalence."
