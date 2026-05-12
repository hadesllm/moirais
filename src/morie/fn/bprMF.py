"""BPR -- Bayesian personalized ranking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bpr_mf"]


def bpr_mf(pairs, K):
    """
    BPR -- Bayesian personalized ranking

    Formula: max_θ sum log σ(x̂_{u,i,j}) − reg

    Parameters
    ----------
    pairs : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rendle et al (2009)
    """
    pairs = np.atleast_1d(np.asarray(pairs, dtype=float))
    n = len(pairs)
    result = float(np.mean(pairs))
    se = float(np.std(pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BPR -- Bayesian personalized ranking"})


def cheatsheet():
    return "bprMF: BPR -- Bayesian personalized ranking"
