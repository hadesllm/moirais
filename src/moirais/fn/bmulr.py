# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian multinomial with Dirichlet conjugate prior."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.special import gammaln


def bayesian_multinomial(
    counts: Union[list, np.ndarray],
    *,
    prior_alpha: Union[list, np.ndarray, None] = None,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Dirichlet-multinomial conjugate Bayesian analysis.

    Prior: theta ~ Dir(alpha_0)
    Posterior: theta | data ~ Dir(alpha_0 + counts)

    :param counts: Observed category counts (k,).
    :param prior_alpha: Dirichlet prior concentration (k,). Default ones.
    :param prob: Credible interval probability.
    :return: Dictionary with posterior mean, mode, and marginal CIs.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 2.
    """
    c = np.asarray(counts, dtype=float)
    k = len(c)
    if prior_alpha is None:
        prior_alpha = np.ones(k)
    else:
        prior_alpha = np.asarray(prior_alpha, dtype=float)

    post_alpha = prior_alpha + c
    alpha_sum = float(np.sum(post_alpha))
    post_mean = post_alpha / alpha_sum

    post_mode = np.zeros(k)
    denom = alpha_sum - k
    if denom > 0:
        post_mode = (post_alpha - 1.0) / denom
        post_mode = np.maximum(post_mode, 0.0)

    from scipy import stats as st

    ci_lower = []
    ci_upper = []
    alpha_half = (1 - prob) / 2
    for j in range(k):
        a_j = post_alpha[j]
        b_j = alpha_sum - a_j
        lo = float(st.beta.ppf(alpha_half, a_j, b_j))
        hi = float(st.beta.ppf(1 - alpha_half, a_j, b_j))
        ci_lower.append(lo)
        ci_upper.append(hi)

    log_ml = float(
        gammaln(np.sum(prior_alpha)) - gammaln(alpha_sum)
        + np.sum(gammaln(post_alpha) - gammaln(prior_alpha))
    )

    return {
        "posterior_alpha": post_alpha.tolist(),
        "posterior_mean": post_mean.tolist(),
        "posterior_mode": post_mode.tolist(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "log_marginal_likelihood": log_ml,
        "k": k,
        "total_count": float(np.sum(c)),
    }


bmulr = bayesian_multinomial


def cheatsheet() -> str:
    return "bayesian_multinomial({}) -> Bayesian multinomial with Dirichlet conjugate prior."
