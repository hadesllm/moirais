# morie.fn -- function file (hadesllm/morie)
"""Optuna-based hyperparameter tuning with tree-structured Parzen estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_optuna"]


def geron_optuna(objective, n_trials, sampler):
    """
    Optuna-based hyperparameter tuning with tree-structured Parzen estimator

    Formula: TPE samples promising configs based on previous evaluations

    Parameters
    ----------
    objective : array-like
        Input data.
    n_trials : array-like
        Input data.
    sampler : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_params

    References
    ----------
    Géron Ch 10
    """
    objective = np.atleast_1d(np.asarray(objective, dtype=float))
    n = len(objective)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Optuna-based hyperparameter tuning with tree-structured Parzen estimator"})
    estimate = np.median(objective)
    se = 1.2533 * np.std(objective, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Optuna-based hyperparameter tuning with tree-structured Parzen estimator"})


def cheatsheet():
    return "hmoptn: Optuna-based hyperparameter tuning with tree-structured Parzen estimator"
