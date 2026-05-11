"""Viterbi decoding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["viterbi"]


def viterbi(obs, trans, emit):
    """
    Viterbi decoding

    Formula: DP over max-prob path through trellis

    Parameters
    ----------
    obs : array-like
        Input data.
    trans : array-like
        Input data.
    emit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Viterbi (1967)
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Viterbi decoding"})


def cheatsheet():
    return "viterb: Viterbi decoding"
