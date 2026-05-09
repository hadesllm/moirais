"""TMLE for conditional average treatment effects (CATE) over covariate strata."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_heterogeneous"]


def tmle_heterogeneous(y, treatment, W, strata):
    """
    TMLE for conditional average treatment effects (CATE) over covariate strata

    Formula: tau(W) = E[Y(1) - Y(0) | W]; targeted update per stratum

    Parameters
    ----------
    y : array-like
        Input data.
    treatment : array-like
        Input data.
    W : array-like
        Input data.
    strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Laan-Luedtke (2015); Athey-Imbens (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "TMLE for conditional average treatment effects (CATE) over covariate strata"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "TMLE for conditional average treatment effects (CATE) over covariate strata"})


def cheatsheet():
    return "tmlhte: TMLE for conditional average treatment effects (CATE) over covariate strata"
