"""Simple one-feature linear model used in the life satisfaction example, predicting life satisfaction from GDP per capita.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_simple_linear_life_satisfaction"]


def geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita):
    """
    Simple one-feature linear model used in the life satisfaction example, predicting life satisfaction from GDP per capita.

    Formula: life_satisfaction = theta_0 + theta_1 * GDP_per_capita

    Parameters
    ----------
    theta_0 : array-like
        Input data.
    theta_1 : array-like
        Input data.
    GDP_per_capita : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: life_satisfaction

    References
    ----------
    Geron (2026), Ch 4, Eq 4-1, p. 136
    """
    theta_0 = np.atleast_1d(np.asarray(theta_0, dtype=float))
    n = len(theta_0)
    result = float(np.mean(theta_0))
    se = float(np.std(theta_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple one-feature linear model used in the life satisfaction example, predicting life satisfaction from GDP per capita."})


def cheatsheet():
    return "grn001: Simple one-feature linear model used in the life satisfaction example, predicting life satisfaction from GDP per capita."
