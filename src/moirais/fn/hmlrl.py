# moirais.fn — function file (hadesllm/moirais)
"""Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_linear_regression_life"]


def geron_linear_regression_life(gdp, theta0, theta1):
    """
    Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example)

    Formula: life_sat = theta0 + theta1 * gdp

    Parameters
    ----------
    gdp : array-like
        Input data.
    theta0 : array-like
        Input data.
    theta1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: life_sat

    References
    ----------
    Géron Ch 1
    """
    gdp = np.atleast_1d(np.asarray(gdp, dtype=float))
    n = len(gdp)
    result = float(np.mean(gdp))
    se = float(np.std(gdp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example)"})


def cheatsheet():
    return "hmlrl: Life satisfaction = theta0 + theta1 * GDP_per_capita (introductory example)"
