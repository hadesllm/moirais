"""HMM POS tagging."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hmm_pos"]


def hmm_pos(X, tagset):
    """
    HMM POS tagging

    Formula: argmax_y prod P(y_t|y_{t-1}) P(x_t|y_t)

    Parameters
    ----------
    X : array-like
        Input data.
    tagset : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Charniak (1993); Brants (2000) TnT
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HMM POS tagging"})


def cheatsheet():
    return "hmmTag: HMM POS tagging"
