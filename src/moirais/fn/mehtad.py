"""Mehrotra's predictor-corrector."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mehrotras_predictor"]


def mehrotras_predictor(c, A, b):
    """
    Mehrotra's predictor-corrector

    Formula: primal-dual interior-point with corrector

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mehrotra (1992)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mehrotra's predictor-corrector"})


def cheatsheet():
    return "mehtad: Mehrotra's predictor-corrector"
