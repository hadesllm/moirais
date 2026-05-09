"""eBAC distribution analysis."""

import numpy as np

from ._containers import DescriptiveResult


def ebac_dist(
    ebac_values: list | np.ndarray,
    legal_limit: float = 0.08,
) -> DescriptiveResult:
    """Analyse distribution of estimated blood alcohol concentration.

    Parameters
    ----------
    ebac_values : array-like
        eBAC values.
    legal_limit : float
        Legal driving limit (default 0.08 g/dL).

    Returns
    -------
    DescriptiveResult
    """
    v = np.asarray(ebac_values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        raise ValueError("No valid eBAC values")

    return DescriptiveResult(
        name="ebac_distribution",
        value=float(np.mean(v)),
        extra={
            "median": float(np.median(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "max": float(np.max(v)),
            "pct_over_limit": float(np.mean(v > legal_limit) * 100),
            "pct_over_0.05": float(np.mean(v > 0.05) * 100),
            "n": len(v),
            "legal_limit": legal_limit,
        },
    )


suebac = ebac_dist


def cheatsheet() -> str:
    return "ebac_dist({}) -> eBAC distribution analysis."
