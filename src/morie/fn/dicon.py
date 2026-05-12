# morie.fn -- function file (hadesllm/morie)
"""Deviance Information Criterion (DIC)."""

from __future__ import annotations

__all__ = ["compute_dic", "dicon"]

from typing import Any, Union

import numpy as np


def compute_dic(
    log_lik_at_samples: Union[list, np.ndarray],
    log_lik_at_mean: float,
) -> dict[str, Any]:
    r"""
    Compute the Deviance Information Criterion (DIC).

    DIC = D_bar + p_D, where D_bar is the posterior mean deviance
    and p_D is the effective number of parameters.

    .. math::

        D(\\theta) = -2 \\log p(y | \\theta)

        \\bar{D} = \\mathbb{E}_{\\theta|y}[D(\\theta)]

        p_D = \\bar{D} - D(\\bar{\\theta})

        \\text{DIC} = \\bar{D} + p_D = 2\\bar{D} - D(\\bar{\\theta})

    Parameters
    ----------
    log_lik_at_samples : array-like
        Total log-likelihood evaluated at each posterior sample (S,).
        These are sum_i log p(y_i | theta_s) for each posterior draw s.
    log_lik_at_mean : float
        Total log-likelihood evaluated at the posterior mean theta_bar.

    Returns
    -------
    dict
        dic : float
        p_d : float -- effective number of parameters
        d_bar : float -- posterior mean deviance
        d_at_mean : float -- deviance at posterior mean

    References
    ----------
    Spiegelhalter, D. J., Best, N. G., Carlin, B. P., & van der
    Linde, A. (2002). Bayesian measures of model complexity and fit.
    *JRSS-B*, 64(4), 583--639.
    """
    ll = np.asarray(log_lik_at_samples, dtype=float).ravel()
    if len(ll) < 2:
        raise ValueError("Need at least 2 posterior samples.")

    deviance_samples = -2.0 * ll
    d_bar = float(np.mean(deviance_samples))
    d_at_mean = -2.0 * log_lik_at_mean
    p_d = d_bar - d_at_mean
    dic = d_bar + p_d

    return {
        "dic": dic,
        "p_d": p_d,
        "d_bar": d_bar,
        "d_at_mean": d_at_mean,
    }


dicon = compute_dic


def cheatsheet() -> str:
    return "compute_dic(log_lik_samples, log_lik_mean) -> Deviance Information Criterion."
