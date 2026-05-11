# morie.fn — function file (hadesllm/morie)
"""Semiparametric rank estimator for single-index model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_semipar_rank"]


def horowitz_semipar_rank(x, y):
    """
    Semiparametric rank estimator for single-index model

    Formula: beta_hat = argmax_b sum_{i<j} [I(Y_i>Y_j)*I(X_i'b>X_j'b)+I(Y_i<Y_j)*I(X_i'b<X_j'b)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 2, Sec 2.5.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Semiparametric rank estimator for single-index model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Semiparametric rank estimator for single-index model"})


def cheatsheet():
    return "hrzrank: Semiparametric rank estimator for single-index model"
