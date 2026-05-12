# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian DIC (deviance information criterion)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def bayesian_dic(
    log_likelihood: Callable[[np.ndarray], float],
    posterior_samples: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Deviance Information Criterion (DIC).

    DIC = D_bar + p_D, where D_bar = E[-2 log L(theta)] and
    p_D = D_bar - D(theta_bar) is the effective number of parameters.

    :param log_likelihood: Function returning log-likelihood given parameter vector.
    :param posterior_samples: MCMC samples (n_samples, d).
    :return: Dictionary with dic, d_bar, p_d, d_at_mean.

    References
    ----------
    Spiegelhalter, D. J., et al. (2002). *JRSS-B*, 64(4), 583--639.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    deviances = np.array([-2.0 * log_likelihood(s) for s in samples])
    d_bar = float(np.mean(deviances))

    theta_bar = np.mean(samples, axis=0)
    d_at_mean = float(-2.0 * log_likelihood(theta_bar))

    p_d = d_bar - d_at_mean
    dic = d_bar + p_d

    return {
        "dic": float(dic),
        "d_bar": d_bar,
        "d_at_mean": d_at_mean,
        "p_d": float(p_d),
        "n_samples": len(samples),
    }


bdic = bayesian_dic


def cheatsheet() -> str:
    return "bayesian_dic({}) -> Bayesian DIC (deviance information criterion)."
