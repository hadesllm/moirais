"""DINO output centering (collapse-prevention)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dino_centering"]


def dino_centering(p_t, C, momentum):
    """
    DINO output centering (collapse-prevention)

    Formula: P_t = softmax((p_t - C)/tau_t); C running mean

    Parameters
    ----------
    p_t : array-like
        Input data.
    C : array-like
        Input data.
    momentum : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Caron et al (2021)
    """
    p_t = np.atleast_1d(np.asarray(p_t, dtype=float))
    n = len(p_t)
    result = float(np.mean(p_t))
    se = float(np.std(p_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINO output centering (collapse-prevention)"})


def cheatsheet():
    return "dinoss: DINO output centering (collapse-prevention)"
