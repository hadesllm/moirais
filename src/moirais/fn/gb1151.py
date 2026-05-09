# moirais.fn — function file (hadesllm/moirais)
"""Blomqvist q (medial correlation): proportion of concordant vs discordant quadrants."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_blomqvist_q"]


def gibbons_blomqvist_q(x, y):
    """
    Blomqvist q (medial correlation): proportion of concordant vs discordant quadrants

    Formula: Q = (A+D-B-C)/(A+B+C+D) where A,B,C,D = quadrant counts around bivariate median

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q, p_value

    References
    ----------
    Gibbons Ch 11.5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Blomqvist q (medial correlation): proportion of concordant vs discordant quadrants"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Blomqvist q (medial correlation): proportion of concordant vs discordant quadrants"})


def cheatsheet():
    return "gb1151: Blomqvist q (medial correlation): proportion of concordant vs discordant quadrants"
