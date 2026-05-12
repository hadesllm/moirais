# morie.fn -- function file (hadesllm/morie)
"""Social determinants of health composite score."""

import numpy as np

from ._containers import ESRes


def social_determinants(
    indicator_values: list | np.ndarray,
    weights: list | np.ndarray | None = None,
) -> ESRes:
    """Compute social determinants of health composite score.

    Parameters
    ----------
    indicator_values : array-like
        Values for each social determinant indicator (e.g. education,
        income, housing, employment).
    weights : array-like or None
        Weights per indicator. Default: equal weights.

    Returns
    -------
    ESRes
    """
    v = np.asarray(indicator_values, dtype=float)
    if len(v) == 0:
        raise ValueError("No indicator values provided")

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(v):
            raise ValueError("weights must match indicator_values length")
        composite = float(np.average(v, weights=w))
    else:
        composite = float(np.mean(v))

    return ESRes(
        measure="SDOH_composite",
        estimate=composite,
        extra={"n_indicators": len(v), "min": float(np.min(v)), "max": float(np.max(v))},
    )


ihsoc = social_determinants


def cheatsheet() -> str:
    return "social_determinants({}) -> Social determinants of health composite score."
