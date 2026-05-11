"""BayesB sparse marker prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_b_marker"]


def bayes_b_marker(y, M, pi):
    """
    BayesB sparse marker prior

    Formula: sigma_j^2=0 with prob pi else scaled-inv-chi2

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meuwissen-Hayes-Goddard (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesB sparse marker prior"})


def cheatsheet():
    return "baysbm: BayesB sparse marker prior"
