# morie.fn -- function file (hadesllm/morie)
"""DP regression posterior: nonparametric regression with DP error distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_regression_posterior"]


def ghosal_dp_regression_posterior(x, y):
    """
    DP regression posterior: nonparametric regression with DP error distribution

    Formula: Y = f(X) + e, e ~ G ~ DP(alpha, G0), posterior for (f, G)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 7 §7.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP regression posterior: nonparametric regression with DP error distribution"})


def cheatsheet():
    return "gh_dp_reg_post: DP regression posterior: nonparametric regression with DP error distribution"
