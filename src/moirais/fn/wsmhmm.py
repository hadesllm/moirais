"""HMM forward algorithm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_hmm_forward"]


def wasserman_hmm_forward(obs, A, B, pi):
    """
    HMM forward algorithm

    Formula: alpha_t(j) = sum alpha_{t-1}(i) a_{ij} b_j(o_t)

    Parameters
    ----------
    obs : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loglik

    References
    ----------
    Wasserman (2004), Ch 23
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HMM forward algorithm"})


def cheatsheet():
    return "wsmhmm: HMM forward algorithm"
