# moirais.fn — function file (hadesllm/moirais)
"""Posterior predictive check."""

from __future__ import annotations

__all__ = ["posterior_predictive_check", "ppc"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def posterior_predictive_check(
    observed: Union[list, np.ndarray],
    replications: Union[list, np.ndarray],
    *,
    test_statistic: Union[Callable, str] = "mean",
) -> dict[str, Any]:
    """
    Posterior predictive check (PPC).

    Compares an observed test statistic T(y) to the distribution of
    T(y_rep) computed from posterior predictive replications.

    The Bayesian p-value is P(T(y_rep) >= T(y_obs)).  Values near
    0.5 indicate good model fit; extreme values (near 0 or 1)
    indicate model misspecification.

    Parameters
    ----------
    observed : array-like
        Observed data (n,).
    replications : array-like
        Posterior predictive replications (n_reps, n) or pre-computed
        test statistics (n_reps,) if 1-D.
    test_statistic : callable or str
        Function mapping data array to scalar, or one of 'mean',
        'var', 'max', 'min', 'median'.

    Returns
    -------
    dict
        bayesian_p : float
        T_obs : float
        T_rep_mean : float
        T_rep_sd : float
        T_rep_quantiles : dict

    Raises
    ------
    ValueError
        If test_statistic name is unknown.

    References
    ----------
    Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive
    assessment of model fitness via realized discrepancies.
    *Statistica Sinica*, 6(4), 733--760.
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    Ch. 6.
    """
    stat_map = {
        "mean": np.mean,
        "var": lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0,
        "max": np.max,
        "min": np.min,
        "median": np.median,
    }

    if isinstance(test_statistic, str):
        if test_statistic not in stat_map:
            raise ValueError(
                f"Unknown test_statistic '{test_statistic}'. "
                f"Use one of {list(stat_map.keys())} or a callable."
            )
        stat_fn = stat_map[test_statistic]
    else:
        stat_fn = test_statistic

    obs = np.asarray(observed, dtype=float)
    reps = np.asarray(replications, dtype=float)

    T_obs = float(stat_fn(obs))

    if reps.ndim == 2:
        T_reps = np.array([float(stat_fn(reps[i])) for i in range(reps.shape[0])])
    elif reps.ndim == 1:
        T_reps = reps
    else:
        raise ValueError("replications must be 1-D or 2-D.")

    bayesian_p = float(np.mean(T_reps >= T_obs))

    return {
        "bayesian_p": bayesian_p,
        "T_obs": T_obs,
        "T_rep_mean": float(np.mean(T_reps)),
        "T_rep_sd": float(np.std(T_reps, ddof=1)) if len(T_reps) > 1 else 0.0,
        "T_rep_quantiles": {
            "q025": float(np.percentile(T_reps, 2.5)),
            "q250": float(np.percentile(T_reps, 25)),
            "q500": float(np.percentile(T_reps, 50)),
            "q750": float(np.percentile(T_reps, 75)),
            "q975": float(np.percentile(T_reps, 97.5)),
        },
        "n_reps": len(T_reps),
    }


ppc = posterior_predictive_check


def cheatsheet() -> str:
    return "posterior_predictive_check(observed, reps) -> Posterior predictive check."
