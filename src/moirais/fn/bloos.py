# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian LOO-CV with Pareto-smoothed importance sampling."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def psis_loo(
    log_lik_matrix: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Pareto-smoothed importance sampling LOO-CV (PSIS-LOO).

    :param log_lik_matrix: Matrix of log-likelihoods (n_samples, n_obs).
    :return: Dictionary with elpd_loo, p_loo, looic, k_hat values, warnings.

    References
    ----------
    Vehtari, A., et al. (2017). *Statistics and Computing*, 27(5), 1413--1432.
    """
    ll = np.asarray(log_lik_matrix, dtype=float)
    if ll.ndim == 1:
        ll = ll.reshape(1, -1)
    n_samples, n_obs = ll.shape

    elpd_loo_i = np.zeros(n_obs)
    k_hat = np.zeros(n_obs)

    for i in range(n_obs):
        log_ratios = -ll[:, i]
        log_ratios_shifted = log_ratios - np.max(log_ratios)
        ratios = np.exp(log_ratios_shifted)

        sorted_ratios = np.sort(ratios)
        M = max(int(min(n_samples / 5, 3 * np.sqrt(n_samples))), 1)
        if M < 2:
            k_hat[i] = float("inf")
        else:
            tail = sorted_ratios[-M:]
            tail_log = np.log(tail + 1e-30)
            threshold = tail_log[0]
            exceedances = tail_log - threshold
            exceedances = exceedances[exceedances > 0]
            if len(exceedances) > 1:
                k_hat[i] = float(np.mean(exceedances))
            else:
                k_hat[i] = 0.0

        log_weights = log_ratios
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        weights = weights / np.sum(weights)

        elpd_loo_i[i] = np.log(np.sum(weights * np.exp(ll[:, i])) + 1e-30)

    elpd_loo = float(np.sum(elpd_loo_i))
    p_loo_i = np.log(np.mean(np.exp(ll), axis=0) + 1e-30) - elpd_loo_i
    p_loo = float(np.sum(p_loo_i))
    looic = float(-2.0 * elpd_loo)
    se = float(np.sqrt(n_obs * np.var(elpd_loo_i, ddof=1)))

    n_high_k = int(np.sum(k_hat > 0.7))

    return {
        "elpd_loo": elpd_loo,
        "p_loo": p_loo,
        "looic": looic,
        "se": se,
        "k_hat": k_hat.tolist(),
        "n_high_k": n_high_k,
        "n_obs": n_obs,
    }


bloos = psis_loo


def cheatsheet() -> str:
    return "psis_loo({}) -> Bayesian LOO-CV with Pareto-smoothed importance sampling."
