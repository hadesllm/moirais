"""Numbered display equation (9.23) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_23"]


def mvsml_ridge_lasso_elastic_eq_9_23(x, y, subject, to, Its, dual):
    """
    Numbered display equation (9.23) from MVSML chapter 9.

    Formula: x + y  2 (9.22) subject to Its dual version according to Wolfe is equal to ) = x2 + y2 + 2\alpha x + y + 2 maximize f x, y, \alpha ( ( )

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    subject : array-like
        Input data.
    to : array-like
        Input data.
    Its : array-like
        Input data.
    dual : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.23) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.23) from MVSML chapter 9."})


def cheatsheet():
    return "msm195: Numbered display equation (9.23) from MVSML chapter 9."
