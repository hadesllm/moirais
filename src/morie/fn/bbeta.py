# morie.fn -- function file (hadesllm/morie)
"""Bayesian beta-binomial model."""

from __future__ import annotations

__all__ = ["bayesian_beta_binomial", "bbeta"]

from typing import Any, Union

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln


def bayesian_beta_binomial(
    successes: Union[list, np.ndarray],
    trials: Union[list, np.ndarray],
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    n_grid: int = 500,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian beta-binomial model for multiple groups.

    Each group i has its own success probability p_i drawn from a
    common Beta(a, b) population distribution.  Given observed
    (k_i, n_i) for each group, the marginal likelihood of each
    observation is the Beta-Binomial PMF.

    This function estimates the hyperparameters (a, b) via marginal
    maximum likelihood on a grid, then returns the posterior for each
    group's probability using the empirical Bayes posterior
    Beta(a + k_i, b + n_i - k_i).

    Parameters
    ----------
    successes : array-like
        Number of successes per group (G,).
    trials : array-like
        Number of trials per group (G,).
    prior_a : float
        Prior on hyperparameter a (initial grid center).
    prior_b : float
        Prior on hyperparameter b (initial grid center).
    n_grid : int
        Grid resolution for hyperparameter search.
    prob : float
        Credible interval probability.

    Returns
    -------
    dict
        group_means : ndarray (G,)
        group_ci_lower : ndarray (G,)
        group_ci_upper : ndarray (G,)
        hyper_a : float
        hyper_b : float
        marginal_log_lik : float

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 5.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    G = len(k)
    if len(n) != G:
        raise ValueError("successes and trials must have same length.")

    a_grid = np.linspace(0.01, max(20.0, 2 * prior_a), n_grid)
    b_grid = np.linspace(0.01, max(20.0, 2 * prior_b), n_grid)

    best_ll = -np.inf
    best_a, best_b = prior_a, prior_b

    for a_val in a_grid:
        for b_val in b_grid:
            ll = 0.0
            for i in range(G):
                ll += (
                    gammaln(n[i] + 1)
                    - gammaln(k[i] + 1)
                    - gammaln(n[i] - k[i] + 1)
                    + betaln(k[i] + a_val, n[i] - k[i] + b_val)
                    - betaln(a_val, b_val)
                )
            if ll > best_ll:
                best_ll = ll
                best_a, best_b = a_val, b_val

    group_means = np.empty(G)
    ci_lo = np.empty(G)
    ci_hi = np.empty(G)

    for i in range(G):
        pa = best_a + k[i]
        pb = best_b + n[i] - k[i]
        group_means[i] = pa / (pa + pb)
        ci_lo[i] = stats.beta.ppf((1 - prob) / 2, pa, pb)
        ci_hi[i] = stats.beta.ppf(1 - (1 - prob) / 2, pa, pb)

    return {
        "group_means": group_means,
        "group_ci_lower": ci_lo,
        "group_ci_upper": ci_hi,
        "hyper_a": float(best_a),
        "hyper_b": float(best_b),
        "marginal_log_lik": float(best_ll),
    }


bbeta = bayesian_beta_binomial


def cheatsheet() -> str:
    return "bayesian_beta_binomial(successes, trials) -> Beta-binomial hierarchical model."
