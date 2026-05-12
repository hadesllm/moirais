# morie.fn -- function file (hadesllm/morie)
"""Impute missing with mean. 'Full Cowling.' -- Deku, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def impute_missing_mean(D):
    """Replace NaN values in distance matrix with mean of observed values.

    Parameters
    ----------
    D : array-like
        Distance matrix with possible NaN values.

    Returns
    -------
    DescriptiveResult
        value = imputed distance matrix.
    """
    import numpy as np

    D = np.asarray(D, dtype=float).copy()
    mask = np.isnan(D)
    n_missing = int(mask.sum())
    if n_missing > 0:
        mean_val = float(np.nanmean(D))
        D[mask] = mean_val
    else:
        mean_val = float(np.mean(D))
    return DescriptiveResult(
        name="impute_missing_mean", value=D, extra={"n_imputed": n_missing, "impute_value": mean_val}
    )


impme = impute_missing_mean


def cheatsheet() -> str:
    return "impute_missing_mean({}) -> Impute missing with mean. 'Full Cowling.' -- Deku, My Hero A"
