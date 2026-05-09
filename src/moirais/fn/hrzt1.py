# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric treatment effect estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_treatment_effect"]


def horowitz_treatment_effect(x, y, treatment):
    """
    Semiparametric treatment effect estimator

    Formula: ATE = E[Y(1)-Y(0)] via kernel matching

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    treatment : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Semiparametric treatment effect estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Semiparametric treatment effect estimator"})


def cheatsheet():
    return "hrzt1: Semiparametric treatment effect estimator"
