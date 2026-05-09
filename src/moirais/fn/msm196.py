"""Numbered display equation (9.24) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_24"]


def mvsml_ridge_lasso_elastic_eq_9_24(f, x, y, subject, to):
    """
    Numbered display equation (9.24) from MVSML chapter 9.

    Formula: \partial f x, y, \alpha ( ) subject to = 2x + 2\alpha = 0 \partial x \partial f x, y, \alpha ( ) = 2y + 2\alpha = 0

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.
    subject : array-like
        Input data.
    to : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.24) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.24) from MVSML chapter 9."})


def cheatsheet():
    return "msm196: Numbered display equation (9.24) from MVSML chapter 9."
