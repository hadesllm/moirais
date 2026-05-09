# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayes factor computation (Savage-Dickey density ratio)."""

from __future__ import annotations

__all__ = ["bayes_factor_savage_dickey", "bfact"]

from typing import Any, Union

import numpy as np
from scipy import stats


def bayes_factor_savage_dickey(
    posterior_samples: Union[list, np.ndarray],
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    null_value: float = 0.0,
    *,
    bw_method: Union[str, float, None] = None,
) -> dict[str, Any]:
    """
    Bayes factor via the Savage-Dickey density ratio.

    For a point null H0: theta = theta_0 nested within a broader
    prior, the Bayes factor in favour of H0 is:

    .. math::

        BF_{01} = \\frac{p(\\theta_0 | y)}{p(\\theta_0)}

    The posterior density at theta_0 is estimated via kernel density
    estimation on the posterior samples.  The prior density is
    evaluated analytically (Normal prior assumed).

    Parameters
    ----------
    posterior_samples : array-like
        MCMC samples from the posterior of the parameter.
    prior_mean : float
        Mean of the Normal prior on the parameter.
    prior_sd : float
        Standard deviation of the Normal prior.
    null_value : float
        Value of the parameter under H0.
    bw_method : str, float, or None
        Bandwidth for scipy.stats.gaussian_kde.

    Returns
    -------
    dict
        bf01 : float -- Bayes factor for H0
        bf10 : float -- Bayes factor for H1
        posterior_density_at_null : float
        prior_density_at_null : float
        evidence_category : str

    References
    ----------
    Dickey, J. M. & Lientz, B. P. (1970). The weighted likelihood
    ratio, sharp hypotheses about chances, the order of a Markov
    chain. *Annals of Mathematical Statistics*, 41(1), 214--226.

    Wagenmakers, E.-J., Lodewyckx, T., Kuriyal, H., & Grasman, R.
    (2010). Bayesian hypothesis testing for psychologists.
    *Psychonomic Bulletin & Review*, 17(3), 367--374.
    """
    samples = np.asarray(posterior_samples, dtype=float).ravel()
    if len(samples) < 10:
        raise ValueError("Need at least 10 posterior samples.")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be positive.")

    kde = stats.gaussian_kde(samples, bw_method=bw_method)
    posterior_at_null = float(kde(null_value)[0])

    prior_at_null = float(stats.norm.pdf(null_value, loc=prior_mean, scale=prior_sd))

    if prior_at_null < 1e-30:
        bf01 = float("inf") if posterior_at_null > 1e-30 else 1.0
    else:
        bf01 = posterior_at_null / prior_at_null

    bf10 = 1.0 / bf01 if bf01 > 1e-30 else float("inf")

    if bf01 > 100:
        cat = "Extreme evidence for H0"
    elif bf01 > 30:
        cat = "Very strong evidence for H0"
    elif bf01 > 10:
        cat = "Strong evidence for H0"
    elif bf01 > 3:
        cat = "Moderate evidence for H0"
    elif bf01 > 1:
        cat = "Anecdotal evidence for H0"
    elif bf01 > 1.0 / 3.0:
        cat = "Anecdotal evidence for H1"
    elif bf01 > 1.0 / 10.0:
        cat = "Moderate evidence for H1"
    elif bf01 > 1.0 / 30.0:
        cat = "Strong evidence for H1"
    elif bf01 > 1.0 / 100.0:
        cat = "Very strong evidence for H1"
    else:
        cat = "Extreme evidence for H1"

    return {
        "bf01": bf01,
        "bf10": bf10,
        "posterior_density_at_null": posterior_at_null,
        "prior_density_at_null": prior_at_null,
        "evidence_category": cat,
    }


bfact = bayes_factor_savage_dickey


def cheatsheet() -> str:
    return "bayes_factor_savage_dickey(samples) -> Savage-Dickey Bayes factor."
