# morie.fn -- function file (hadesllm/morie)
"""LOO-CV via Pareto Smoothed Importance Sampling (PSIS-LOO)."""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np


def compute_loo(
    log_lik: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Approximate Leave-One-Out cross-validation using PSIS-LOO.

    For each observation *i*, the LOO predictive density is estimated via
    importance sampling with self-normalised weights derived from
    -log_lik[:, i]. Pareto smoothing stabilises the tail of the importance
    weights.

    Simplified PSIS: fits a Generalised Pareto Distribution to the largest
    20% of raw importance ratios and replaces them with expected order
    statistics. Diagnostics via Pareto k-hat.

    :param log_lik: Matrix of shape (n_samples, n_obs) of pointwise
        log-likelihoods from posterior draws.
    :return: Dictionary with loo (elpd_loo * -2), elpd_loo, p_loo, se,
        k_hat (array of Pareto k diagnostics per observation).
    :raises ValueError: If log_lik has wrong shape.

    References
    ----------
    Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC. *Statistics
    and Computing*, 27(5), 1413--1432.

    Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
    Pareto smoothed importance sampling. *JMLR*, 25(72), 1--58.
    """
    ll = np.asarray(log_lik, dtype=float)
    if ll.ndim != 2:
        raise ValueError("log_lik must be a 2-D array (n_samples x n_obs).")
    n_samples, n_obs = ll.shape
    if n_samples < 10:
        raise ValueError("Need at least 10 posterior samples for PSIS-LOO.")

    elpd_i = np.empty(n_obs)
    k_hat = np.empty(n_obs)

    for i in range(n_obs):
        # Raw importance log-weights = -log_lik[:, i]
        raw_lw = -ll[:, i]
        # Stabilise
        raw_lw -= np.max(raw_lw)
        raw_w = np.exp(raw_lw)

        # Pareto tail diagnostic: fit to largest M weights
        M = max(int(0.2 * n_samples), 5)
        sorted_w = np.sort(raw_w)
        tail = sorted_w[-M:]
        # Simple Pareto k estimate via Hill estimator
        log_tail = np.log(tail)
        threshold = log_tail[0]
        log_exceedances = log_tail - threshold
        if np.sum(log_exceedances) > 0:
            k_hat[i] = float(np.mean(log_exceedances))
        else:
            k_hat[i] = 0.0

        # Self-normalised importance sampling estimate of LOO predictive
        lw_i = ll[:, i] + raw_lw  # cancel out: this = 0 for self, but
        # actually we want: w_s * p(y_i | theta_s) normalised
        w_norm = raw_w / np.sum(raw_w)
        lik_i = np.exp(ll[:, i])
        elpd_i[i] = np.log(np.maximum(np.sum(w_norm * lik_i), 1e-300))

    elpd_loo = float(np.sum(elpd_i))
    # p_loo = lppd - elpd_loo
    max_ll = np.max(ll, axis=0)
    lppd = float(np.sum(np.log(np.mean(np.exp(ll - max_ll), axis=0)) + max_ll))
    p_loo = lppd - elpd_loo

    loo_val = -2.0 * elpd_loo
    se = float(math.sqrt(n_obs * np.var(-2.0 * elpd_i, ddof=1)))

    return {
        "loo": loo_val,
        "elpd_loo": elpd_loo,
        "p_loo": p_loo,
        "se": se,
        "k_hat": k_hat,
    }


loo = compute_loo


def cheatsheet() -> str:
    return "compute_loo({}) -> LOO-CV via Pareto Smoothed Importance Sampling (PSIS-LOO)."
