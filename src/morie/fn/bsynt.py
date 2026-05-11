# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian synthetic control."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_synthetic_control(
    y_treat: Union[list, np.ndarray],
    Y_donors: Union[list, np.ndarray],
    t0: int,
    *,
    n_iter: int = 3000,
    seed: int = 42,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian synthetic control method.

    Uses a Dirichlet prior on weights with Gibbs sampling.

    :param y_treat: Treated unit outcomes over time (T,).
    :param Y_donors: Donor unit outcomes (T, J).
    :param t0: Treatment start time index.
    :param n_iter: Number of MCMC iterations.
    :param seed: Random seed.
    :param prob: Credible interval probability.
    :return: Dictionary with weight_samples, effect_samples, posterior summaries.

    References
    ----------
    Brodersen, K. H., et al. (2015). *Annals of Applied Statistics*, 9(1), 247--274.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_treat, dtype=float).ravel()
    D = np.asarray(Y_donors, dtype=float)
    T, J = D.shape

    y_pre = y[:t0]
    D_pre = D[:t0]

    weights = np.ones(J) / J
    sigma2 = 1.0

    weight_samples = np.empty((n_iter, J))
    effect_samples = np.empty((n_iter, T - t0))

    for it in range(n_iter):
        synth_pre = D_pre @ weights
        resid = y_pre - synth_pre
        sigma2 = 1.0 / rng.gamma(t0 / 2.0, 2.0 / (float(resid @ resid) + 1e-10))

        for j in range(J):
            others = D_pre @ weights - D_pre[:, j] * weights[j]
            residual_j = y_pre - others
            mean_j = float(np.sum(residual_j * D_pre[:, j])) / (float(np.sum(D_pre[:, j] ** 2)) + 1e-10)
            var_j = sigma2 / (float(np.sum(D_pre[:, j] ** 2)) + 1e-10)
            w_j = rng.normal(mean_j, np.sqrt(var_j + 1e-10))
            weights[j] = max(w_j, 0.0)

        w_sum = np.sum(weights)
        if w_sum > 1e-10:
            weights = weights / w_sum
        else:
            weights = np.ones(J) / J

        weight_samples[it] = weights
        synth_post = D[t0:] @ weights
        effect_samples[it] = y[t0:] - synth_post

    alpha_half = (1 - prob) / 2
    effect_mean = np.mean(effect_samples, axis=0)
    effect_lo = np.percentile(effect_samples, 100 * alpha_half, axis=0)
    effect_hi = np.percentile(effect_samples, 100 * (1 - alpha_half), axis=0)

    return {
        "effect_mean": effect_mean.tolist(),
        "effect_ci_lower": effect_lo.tolist(),
        "effect_ci_upper": effect_hi.tolist(),
        "weight_mean": np.mean(weight_samples, axis=0).tolist(),
        "cumulative_effect": float(np.sum(effect_mean)),
        "prob_positive_effect": float(np.mean(effect_samples > 0)),
    }


bsynt = bayesian_synthetic_control


def cheatsheet() -> str:
    return "bayesian_synthetic_control({}) -> Bayesian synthetic control."
