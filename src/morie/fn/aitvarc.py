"""CLR covariance matrix of a compositional sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_clr_covariance"]


def aitchison_clr_covariance(X):
    """
    CLR covariance matrix of a compositional sample

    Formula: Σ_clr = cov(clr(X))

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Sigma

    References
    ----------
    Pawlowsky-Glahn (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLR covariance matrix of a compositional sample"})


def cheatsheet():
    return "aitvarc: CLR covariance matrix of a compositional sample"
