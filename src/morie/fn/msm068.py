"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(means, that, the, random, matrix, Z):
    """Time discovers truth. -- Seneca"""
    means = np.atleast_1d(np.asarray(means, dtype=float))
    n = len(means)
    result = float(np.mean(means))
    se = float(np.std(means, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm068: Numbered display equation (6.9) from MVSML chapter 6."
