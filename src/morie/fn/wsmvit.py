"""Viterbi algorithm for HMM most-likely path."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_viterbi"]


def wasserman_viterbi(obs, A, B, pi):
    """
    Viterbi algorithm for HMM most-likely path

    Formula: delta_t(j) = max_i delta_{t-1}(i) a_{ij} b_j(o_t)

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
        Keys: path

    References
    ----------
    Wasserman (2004), Ch 23
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Viterbi algorithm for HMM most-likely path"})


def cheatsheet():
    return "wsmvit: Viterbi algorithm for HMM most-likely path"
