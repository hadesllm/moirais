# moirais.fn — function file (hadesllm/moirais)
"""Inverse probability of censoring weights (IPCW)."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ._containers import DescriptiveResult


def ipcw(
    censoring_indicator: np.ndarray | list,
    covariates: np.ndarray | list,
) -> DescriptiveResult:
    """
    Compute inverse probability of censoring weights.

    Fits a logistic model for the probability of being uncensored
    and returns stabilised weights.

    Parameters
    ----------
    censoring_indicator : array-like
        1 = observed (uncensored), 0 = censored.
    covariates : array-like
        Covariate matrix (n x p).

    Returns
    -------
    DescriptiveResult
        extra has 'weights', 'mean_weight'.

    References
    ----------
    Robins, J. M., & Finkelstein, D. M. (2000). Correcting for
    noncompliance and dependent censoring in an AIDS clinical trial.
    *Biometrics*, 56(3), 779-788.
    """
    C = np.asarray(censoring_indicator, dtype=float)
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(C) != X.shape[0]:
        raise ValueError("Length mismatch.")

    Xd = np.column_stack([np.ones(len(C)), X])
    from numpy.linalg import lstsq

    beta, _, _, _ = lstsq(Xd, C, rcond=None)
    p_uncensored = np.clip(expit(Xd @ beta), 0.01, 0.99)

    p_marg = np.mean(C)
    weights = np.where(C == 1, p_marg / p_uncensored, 0.0)

    return DescriptiveResult(
        name="IPCW",
        value=float(np.mean(weights[C == 1])),
        extra={
            "weights": weights,
            "mean_weight": float(np.mean(weights[C == 1])),
            "n_uncensored": int(np.sum(C)),
            "n_censored": int(np.sum(1 - C)),
        },
    )


ipctw = ipcw


def cheatsheet() -> str:
    return "ipcw({}) -> Inverse probability of censoring weights (IPCW)."
