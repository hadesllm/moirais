# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian causal effect (posterior ATE)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_ate(
    y: Union[list, np.ndarray],
    treatment: Union[list, np.ndarray],
    *,
    prior_mu_diff: float = 0.0,
    prior_sigma_diff: float = 10.0,
    n_iter: int = 5000,
    seed: int = 42,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian ATE estimation via posterior inference on the treatment effect.

    Assumes: y_1 ~ N(mu_1, sigma^2), y_0 ~ N(mu_0, sigma^2), ATE = mu_1 - mu_0.

    :param y: Outcome variable (n,).
    :param treatment: Treatment indicator 0/1 (n,).
    :param prior_mu_diff: Prior mean on ATE.
    :param prior_sigma_diff: Prior SD on ATE.
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :param prob: Credible interval probability.
    :return: Dictionary with ate_samples, posterior mean, HDI.

    References
    ----------
    Rubin, D. B. (1978). *Annals of Statistics*, 6(1), 34--58.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float).ravel()
    t_arr = np.asarray(treatment, dtype=float).ravel()

    y1 = y_arr[t_arr == 1]
    y0 = y_arr[t_arr == 0]
    n1, n0 = len(y1), len(y0)

    mu0 = float(np.mean(y0))
    mu1 = float(np.mean(y1))
    sigma2 = float(np.var(y_arr, ddof=1))

    ate_samples = np.empty(n_iter)

    for it in range(n_iter):
        prec0 = n0 / sigma2
        post_mu0 = (prec0 * float(np.mean(y0))) / (prec0 + 0.01)
        mu0 = rng.normal(post_mu0, 1.0 / np.sqrt(prec0 + 0.01))

        prec1 = n1 / sigma2
        post_mu1 = (prec1 * float(np.mean(y1))) / (prec1 + 0.01)
        mu1 = rng.normal(post_mu1, 1.0 / np.sqrt(prec1 + 0.01))

        ss = float(np.sum((y0 - mu0) ** 2) + np.sum((y1 - mu1) ** 2))
        sigma2 = 1.0 / rng.gamma((n0 + n1) / 2.0, 2.0 / (ss + 1e-10))

        ate_samples[it] = mu1 - mu0

    alpha_half = (1 - prob) / 2
    hdi_lo = float(np.percentile(ate_samples, 100 * alpha_half))
    hdi_hi = float(np.percentile(ate_samples, 100 * (1 - alpha_half)))

    return {
        "ate_samples": ate_samples,
        "posterior_mean": float(np.mean(ate_samples)),
        "posterior_sd": float(np.std(ate_samples, ddof=1)),
        "hdi_lower": hdi_lo,
        "hdi_upper": hdi_hi,
        "prob_positive": float(np.mean(ate_samples > 0)),
    }


bcauz = bayesian_ate


def cheatsheet() -> str:
    return "bayesian_ate({}) -> Bayesian causal effect (posterior ATE)."
