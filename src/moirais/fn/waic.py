"""Widely Applicable Information Criterion (WAIC)."""

from __future__ import annotations

__all__ = ["compute_waic", "waic"]

import math
from typing import Any, Union

import numpy as np


def compute_waic(
    log_lik: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Compute the Widely Applicable Information Criterion (WAIC).

    Uses the pointwise log-likelihood matrix to estimate
    out-of-sample predictive accuracy:

    .. math::

        \\text{WAIC} = -2 (\\text{lppd} - p_{\\text{waic}})

    where lppd = sum_i log(mean_s exp(log_lik[s,i])) and
    p_waic = sum_i var_s(log_lik[s,i]).

    Parameters
    ----------
    log_lik : array-like
        Matrix of shape (n_samples, n_obs) with pointwise
        log-likelihood values from posterior samples.

    Returns
    -------
    dict
        waic, lppd, p_waic, se, pointwise_waic

    Raises
    ------
    ValueError
        If log_lik has wrong shape or too few samples.

    References
    ----------
    Watanabe, S. (2010). *JMLR*, 11, 3571--3594.
    Gelman, A., Hwang, J., & Vehtari, A. (2014). *Statistics and
    Computing*, 24(6), 997--1016.
    """
    ll = np.asarray(log_lik, dtype=float)
    if ll.ndim != 2:
        raise ValueError("log_lik must be a 2-D array (n_samples x n_obs).")
    n_samples, n_obs = ll.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 posterior samples.")

    max_ll = np.max(ll, axis=0)
    lppd_pointwise = np.log(np.mean(np.exp(ll - max_ll), axis=0)) + max_ll
    lppd = float(np.sum(lppd_pointwise))

    p_waic_pointwise = np.var(ll, axis=0, ddof=1)
    p_waic = float(np.sum(p_waic_pointwise))

    pw = -2.0 * (lppd_pointwise - p_waic_pointwise)
    waic_val = float(np.sum(pw))

    se = float(math.sqrt(n_obs * np.var(pw, ddof=1)))

    return {
        "waic": waic_val,
        "lppd": lppd,
        "p_waic": p_waic,
        "se": se,
        "pointwise_waic": pw,
    }


waic = compute_waic


def cheatsheet() -> str:
    return "compute_waic(log_lik) -> Widely Applicable Information Criterion (WAIC)."
