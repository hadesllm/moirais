# morie.fn — function file (hadesllm/morie)
"""MLSMU6 missing data imputation. 'Toguro Punch.' -- Toguro, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def mlsmu6_missing_impute(D, method="mean"):
    """Impute missing values in unfolding distance matrix.

    Parameters
    ----------
    D : array-like
        Distance/preference matrix with NaN for missing.
    method : str
        'mean' (row mean), 'col_mean', or 'global_mean'.

    Returns
    -------
    DescriptiveResult
        value = imputed matrix.
    """
    import numpy as np

    D = np.asarray(D, dtype=float).copy()
    mask = np.isnan(D)
    n_missing = int(mask.sum())

    if method == "col_mean":
        col_means = np.nanmean(D, axis=0)
        for j in range(D.shape[1]):
            D[mask[:, j], j] = col_means[j]
    elif method == "global_mean":
        D[mask] = np.nanmean(D)
    else:
        row_means = np.nanmean(D, axis=1)
        for i in range(D.shape[0]):
            D[i, mask[i]] = row_means[i]

    return DescriptiveResult(
        name="mlsmu6_missing_impute",
        value=D,
        extra={"n_imputed": n_missing, "method": method},
    )


mlsms = mlsmu6_missing_impute


def cheatsheet() -> str:
    return "mlsmu6_missing_impute({}) -> MLSMU6 missing data imputation. 'Toguro Punch.' -- Toguro, Y"
