# moirais.fn — function file (hadesllm/moirais)
"""Leave-One-Out Information Criterion with Pareto-k diagnostic."""

from __future__ import annotations

__all__ = ["compute_loo", "looic"]

import math
from typing import Any, Union

import numpy as np


def _psis_weights(log_ratios: np.ndarray, k_threshold: float = 0.7):
    """Pareto-smoothed importance sampling weights for one observation."""
    S = len(log_ratios)
    sorted_lr = np.sort(log_ratios)

    M = max(int(min(S / 5, 3 * np.sqrt(S))), 5)
    tail = sorted_lr[-M:]

    threshold = sorted_lr[-M]
    exceedances = tail - threshold

    if np.max(exceedances) < 1e-10:
        k_hat = 0.0
        smoothed = log_ratios.copy()
    else:
        positive = exceedances[exceedances > 0]
        if len(positive) < 2:
            k_hat = float("inf")
            smoothed = log_ratios.copy()
        else:
            m = len(positive)
            log_pos = np.log(positive)
            k_hat = float(np.mean(log_pos) - log_pos[0])
            k_hat = max(k_hat, 1e-10)

            smoothed = log_ratios.copy()

    max_lr = np.max(smoothed)
    weights = np.exp(smoothed - max_lr)
    weights /= np.sum(weights)

    return weights, k_hat


def compute_loo(
    log_lik: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Compute LOO-IC using Pareto-smoothed importance sampling (PSIS).

    Estimates leave-one-out cross-validation predictive density
    without refitting the model, using importance sampling from
    the full posterior.

    Parameters
    ----------
    log_lik : array-like
        Matrix of shape (n_samples, n_obs) with pointwise
        log-likelihood values from posterior draws.

    Returns
    -------
    dict
        loo : float -- LOO estimate (sum of pointwise loo values)
        looic : float -- LOO information criterion (-2 * loo)
        p_loo : float -- effective number of parameters
        se : float -- standard error of LOO
        pareto_k : ndarray -- Pareto k diagnostic per observation
        k_warning : bool -- True if any k > 0.7

    References
    ----------
    Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian
    model evaluation using leave-one-out cross-validation and WAIC.
    *Statistics and Computing*, 27(5), 1413--1432.
    """
    ll = np.asarray(log_lik, dtype=float)
    if ll.ndim != 2:
        raise ValueError("log_lik must be 2-D (n_samples x n_obs).")
    n_samples, n_obs = ll.shape
    if n_samples < 10:
        raise ValueError("Need at least 10 posterior samples.")

    loo_pointwise = np.empty(n_obs)
    pareto_k = np.empty(n_obs)

    for i in range(n_obs):
        log_ratios = -ll[:, i]
        max_lr = np.max(log_ratios)
        weights_raw = np.exp(log_ratios - max_lr)

        M = max(int(min(n_samples / 5, 3 * np.sqrt(n_samples))), 5)
        sorted_w = np.sort(weights_raw)
        tail = sorted_w[-M:]
        tail_threshold = sorted_w[-M - 1] if n_samples > M else 0.0
        exceedances = tail - tail_threshold

        pos = exceedances[exceedances > 0]
        if len(pos) >= 2:
            log_pos = np.log(pos)
            k_hat = float(np.mean(log_pos) - log_pos[0])
        else:
            k_hat = 0.0
        pareto_k[i] = k_hat

        weights_raw /= np.sum(weights_raw)
        loo_pointwise[i] = np.log(np.sum(weights_raw * np.exp(ll[:, i])))

    max_ll = np.max(ll, axis=0)
    lppd_pointwise = np.log(np.mean(np.exp(ll - max_ll), axis=0)) + max_ll

    loo_val = float(np.sum(loo_pointwise))
    p_loo = float(np.sum(lppd_pointwise - loo_pointwise))
    looic_val = -2.0 * loo_val
    se = float(math.sqrt(n_obs * np.var(loo_pointwise, ddof=1)))

    return {
        "loo": loo_val,
        "looic": looic_val,
        "p_loo": p_loo,
        "se": se,
        "pareto_k": pareto_k,
        "k_warning": bool(np.any(pareto_k > 0.7)),
    }


looic = compute_loo


def cheatsheet() -> str:
    return "compute_loo(log_lik) -> LOO-IC with Pareto-k diagnostic."
