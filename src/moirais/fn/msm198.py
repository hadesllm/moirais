"""Numbered display equation (9.26) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_26"]


def mvsml_ridge_lasso_elastic_eq_9_26(The, last, version, of, the, Wolfe):
    """
    Numbered display equation (9.26) from MVSML chapter 9.

    Formula: and \alpha  0 The last version of the Wolfe dual can be simpliﬁed by replacing x = y = \alpha in the dual version, and we obtained: ( ) = +2\alpha2 + 4\alpha maximize L \alpha (9.25) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} \alpha subject to \alpha  0

    Parameters
    ----------
    The : array-like
        Input data.
    last : array-like
        Input data.
    version : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    Wolfe : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.26) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.26) from MVSML chapter 9."})


def cheatsheet():
    return "msm198: Numbered display equation (9.26) from MVSML chapter 9."
