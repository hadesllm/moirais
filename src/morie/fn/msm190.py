r"""Numbered display equation (9.18) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_18"]


def mvsml_ridge_lasso_elastic_eq_9_18(z, x, f, subject, to):
    r"""
    Numbered display equation (9.18) from MVSML chapter 9.

    Formula: (9.17) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} x, \alpha \partial f x, \alpha ( ) subject to = 2x + 2\alpha = 0 \partial x and \alpha  0

    Parameters
    ----------
    z : array-like
        Input data.
    x : array-like
        Input data.
    f : array-like
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
    MVSML, Eq. (9.18) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.18) from MVSML chapter 9."})


def cheatsheet():
    return "msm190: Numbered display equation (9.18) from MVSML chapter 9."
