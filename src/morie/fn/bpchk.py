# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Posterior predictive check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def posterior_predictive_check(
    y_obs: Union[list, np.ndarray],
    y_rep: Union[list, np.ndarray],
    *,
    test_stat: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, Any]:
    """
    Posterior predictive check: compare observed test statistic to replicated data.

    :param y_obs: Observed data (n,).
    :param y_rep: Replicated datasets (n_rep, n). Each row is one replicated dataset.
    :param test_stat: Test statistic function applied to each dataset.
    :return: Dictionary with observed_stat, rep_stats, p_value.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 6.
    """
    y = np.asarray(y_obs, dtype=float).ravel()
    rep = np.asarray(y_rep, dtype=float)
    if rep.ndim == 1:
        rep = rep.reshape(1, -1)

    t_obs = float(test_stat(y))
    t_rep = np.array([float(test_stat(row)) for row in rep])

    p_value = float(np.mean(t_rep >= t_obs))

    return {
        "observed_stat": t_obs,
        "rep_stats_mean": float(np.mean(t_rep)),
        "rep_stats_sd": float(np.std(t_rep, ddof=1)) if len(t_rep) > 1 else 0.0,
        "p_value": p_value,
        "n_rep": len(t_rep),
    }


bpchk = posterior_predictive_check


def cheatsheet() -> str:
    return "posterior_predictive_check({}) -> Posterior predictive check."
