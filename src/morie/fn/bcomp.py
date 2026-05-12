# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian model comparison (DIC, WAIC, Bayes factor approximation)."""

from __future__ import annotations

import numpy as np


def bayesian_model_compare(
    fit1: dict,
    fit2: dict,
) -> dict:
    """Compare two Bayesian models via DIC, WAIC, and approximate BF.

    Expects each fit dict to contain log-likelihood samples from the
    posterior (``loglik_samples``, n_iter x n array) or summary
    statistics.

    Parameters
    ----------
    fit1 : dict
        First model fit. Must contain ``loglik_samples`` (ndarray,
        n_samples x n_obs) or ``dic`` and ``waic`` directly.
    fit2 : dict
        Second model fit (same format).

    Returns
    -------
    dict
        Keys: ``dic1``, ``dic2``, ``delta_dic``, ``waic1``, ``waic2``,
        ``delta_waic``, ``preferred`` (str, 'model1' or 'model2'),
        ``bf_approx`` (approximate Bayes factor via BIC approximation).

    References
    ----------
    Spiegelhalter, D. J., et al. (2002). Bayesian measures of model
    complexity and fit. *JRSS-B*, 64(4), 583--639.
    Watanabe, S. (2010). Asymptotic equivalence of Bayes cross
    validation and widely applicable information criterion. *JMLR*,
    11, 3571--3594.
    """

    def _compute_ic(fit: dict) -> tuple[float, float]:
        if "dic" in fit and "waic" in fit:
            return fit["dic"], fit["waic"]

        ll = np.asarray(fit.get("loglik_samples", [[]]), dtype=np.float64)
        if ll.size == 0:
            return np.nan, np.nan

        n_samp, n_obs = ll.shape

        # DIC
        mean_ll = np.mean(ll, axis=0)
        D_bar = -2.0 * np.sum(mean_ll)
        D_theta_bar = -2.0 * np.sum(np.mean(ll, axis=0))
        # Effective parameters
        p_D = 2.0 * (np.sum(np.mean(ll, axis=0)) - np.mean(np.sum(ll, axis=1)))
        # Actually: p_D = var of deviance / 2
        p_D = np.var(-2.0 * np.sum(ll, axis=1)) / 2.0
        dic = D_bar + p_D

        # WAIC
        # lppd = sum log(mean(lik))
        lppd = np.sum(np.log(np.mean(np.exp(ll - np.max(ll, axis=0)), axis=0)) + np.max(ll, axis=0))
        # p_waic = sum var(log_lik)
        p_waic = np.sum(np.var(ll, axis=0, ddof=1))
        waic = -2.0 * (lppd - p_waic)

        return dic, waic

    dic1, waic1 = _compute_ic(fit1)
    dic2, waic2 = _compute_ic(fit2)

    delta_dic = dic1 - dic2
    delta_waic = waic1 - waic2

    # Preferred: lower is better
    if np.isfinite(delta_waic):
        preferred = "model1" if delta_waic < 0 else "model2"
    elif np.isfinite(delta_dic):
        preferred = "model1" if delta_dic < 0 else "model2"
    else:
        preferred = "indeterminate"

    # Approximate BF via BIC approximation (Schwarz weight)
    # BF ~ exp(-0.5 * delta_BIC)
    # Use WAIC as proxy
    bf_approx = np.exp(-0.5 * delta_waic) if np.isfinite(delta_waic) else np.nan

    return {
        "dic1": float(dic1) if np.isfinite(dic1) else None,
        "dic2": float(dic2) if np.isfinite(dic2) else None,
        "delta_dic": float(delta_dic) if np.isfinite(delta_dic) else None,
        "waic1": float(waic1) if np.isfinite(waic1) else None,
        "waic2": float(waic2) if np.isfinite(waic2) else None,
        "delta_waic": float(delta_waic) if np.isfinite(delta_waic) else None,
        "preferred": preferred,
        "bf_approx": float(bf_approx) if np.isfinite(bf_approx) else None,
    }


def cheatsheet() -> str:
    return "bayesian_model_compare({}) -> Bayesian model comparison (DIC, WAIC, Bayes factor approxima"
