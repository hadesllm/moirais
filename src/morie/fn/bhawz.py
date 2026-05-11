# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian hazard estimation."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_hazard(
    times: Union[list, np.ndarray],
    events: Union[list, np.ndarray],
    *,
    n_intervals: int = 10,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian piecewise-constant hazard model with gamma prior.

    :param times: Observed times (n,).
    :param events: Event indicators 0/1 (n,).
    :param n_intervals: Number of piecewise intervals.
    :param prior_a: Gamma prior shape for each interval hazard.
    :param prior_b: Gamma prior rate for each interval hazard.
    :param prob: Credible interval probability.
    :return: Dictionary with hazard estimates, survival function, CIs.

    References
    ----------
    Ibrahim, J. G., et al. (2001). *Bayesian Survival Analysis*, Springer.
    """
    t = np.asarray(times, dtype=float).ravel()
    d = np.asarray(events, dtype=float).ravel()
    n = len(t)

    breaks = np.linspace(0, np.max(t) * 1.01, n_intervals + 1)
    widths = np.diff(breaks)

    d_k = np.zeros(n_intervals)
    e_k = np.zeros(n_intervals)

    for i in range(n):
        for k in range(n_intervals):
            if t[i] > breaks[k + 1]:
                e_k[k] += widths[k]
            elif t[i] > breaks[k]:
                e_k[k] += t[i] - breaks[k]
                d_k[k] += d[i]
                break

    from scipy import stats as st

    post_a = prior_a + d_k
    post_b = prior_b + e_k
    hazard_mean = post_a / (post_b + 1e-30)

    alpha_half = (1 - prob) / 2
    ci_lower = [float(st.gamma.ppf(alpha_half, a=a, scale=1 / (b + 1e-30))) for a, b in zip(post_a, post_b)]
    ci_upper = [float(st.gamma.ppf(1 - alpha_half, a=a, scale=1 / (b + 1e-30))) for a, b in zip(post_a, post_b)]

    cum_hazard = np.cumsum(hazard_mean * widths)
    survival = np.exp(-cum_hazard)

    return {
        "hazard": hazard_mean.tolist(),
        "survival": survival.tolist(),
        "breaks": breaks.tolist(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_events": int(np.sum(d)),
    }


bhawz = bayesian_hazard


def cheatsheet() -> str:
    return "bayesian_hazard({}) -> Bayesian hazard estimation."
