# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayes factor via Savage-Dickey density ratio."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import stats


def bayes_factor_savage_dickey(
    posterior_samples: Union[list, np.ndarray],
    *,
    null_value: float = 0.0,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> dict[str, Any]:
    """
    Bayes factor (BF01) via the Savage-Dickey density ratio.

    BF01 = p(theta = null_value | data) / p(theta = null_value)

    The posterior density at the null is estimated via KDE.

    :param posterior_samples: MCMC samples from the posterior of the parameter.
    :param null_value: Value of theta under H0.
    :param prior_mean: Mean of the Normal prior.
    :param prior_sd: SD of the Normal prior.
    :return: Dictionary with bf01, bf10, evidence_category.

    References
    ----------
    Wagenmakers, E.-J., et al. (2010). *Cognitive Psychology*, 60(3), 158--189.
    """
    samples = np.asarray(posterior_samples, dtype=float).ravel()

    kde = stats.gaussian_kde(samples)
    posterior_at_null = float(kde(null_value)[0])
    prior_at_null = float(stats.norm.pdf(null_value, loc=prior_mean, scale=prior_sd))

    bf01 = posterior_at_null / prior_at_null if prior_at_null > 1e-30 else float("inf")
    bf10 = 1.0 / bf01 if bf01 > 1e-30 else float("inf")

    if bf01 > 100:
        cat = "Decisive evidence for H0"
    elif bf01 > 10:
        cat = "Strong evidence for H0"
    elif bf01 > 3:
        cat = "Moderate evidence for H0"
    elif bf01 > 1:
        cat = "Anecdotal evidence for H0"
    elif bf01 > 1.0 / 3:
        cat = "Anecdotal evidence for H1"
    elif bf01 > 1.0 / 10:
        cat = "Moderate evidence for H1"
    elif bf01 > 1.0 / 100:
        cat = "Strong evidence for H1"
    else:
        cat = "Decisive evidence for H1"

    return {
        "bf01": float(bf01),
        "bf10": float(bf10),
        "posterior_density_at_null": posterior_at_null,
        "prior_density_at_null": prior_at_null,
        "evidence_category": cat,
    }


bfsdm = bayes_factor_savage_dickey


def cheatsheet() -> str:
    return "bayes_factor_savage_dickey({}) -> Bayes factor via Savage-Dickey density ratio."
