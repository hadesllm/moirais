# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""WAIC (widely applicable information criterion)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def compute_waic(
    log_lik_matrix: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Widely Applicable Information Criterion (WAIC).

    :param log_lik_matrix: Matrix of log-likelihoods (n_samples, n_obs).
        Each row is one posterior draw, each column is one observation.
    :return: Dictionary with waic, lppd, p_waic, se.

    References
    ----------
    Watanabe, S. (2010). *JMLR*, 11, 3571--3594.
    Vehtari, A., et al. (2017). *Statistics and Computing*, 27(5), 1413--1432.
    """
    ll = np.asarray(log_lik_matrix, dtype=float)
    if ll.ndim == 1:
        ll = ll.reshape(1, -1)
    n_samples, n_obs = ll.shape

    max_ll = np.max(ll, axis=0)
    lppd_i = np.log(np.mean(np.exp(ll - max_ll), axis=0)) + max_ll
    lppd = float(np.sum(lppd_i))

    p_waic_i = np.var(ll, axis=0, ddof=1)
    p_waic = float(np.sum(p_waic_i))

    elpd_waic_i = lppd_i - p_waic_i
    waic = float(-2.0 * np.sum(elpd_waic_i))

    se = float(np.sqrt(n_obs * np.var(elpd_waic_i, ddof=1)))

    return {
        "waic": waic,
        "lppd": lppd,
        "p_waic": p_waic,
        "elpd_waic": float(np.sum(elpd_waic_i)),
        "se": se,
        "n_obs": n_obs,
        "n_samples": n_samples,
    }


bwaic = compute_waic


def cheatsheet() -> str:
    return "compute_waic({}) -> WAIC (widely applicable information criterion)."
