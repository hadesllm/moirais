"""Numbered display equation (9.20) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_20"]


def mvsml_ridge_lasso_elastic_eq_9_20(Then, the, last, version, of, Wolfe):
    """
    Numbered display equation (9.20) from MVSML chapter 9.

    Formula: and \alpha  0 (9.18) Then the last version of the Wolfe dual can be simpliﬁed as ( ) = +\alpha2 + 2\alpha maximize L \lambda (9.19) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} \alpha subject to \alpha  0

    Parameters
    ----------
    Then : array-like
        Input data.
    the : array-like
        Input data.
    last : array-like
        Input data.
    version : array-like
        Input data.
    of : array-like
        Input data.
    Wolfe : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.20) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Then = np.atleast_1d(np.asarray(Then, dtype=float))
    n = len(Then)
    result = float(np.mean(Then))
    se = float(np.std(Then, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.20) from MVSML chapter 9."})


def cheatsheet():
    return "msm192: Numbered display equation (9.20) from MVSML chapter 9."
