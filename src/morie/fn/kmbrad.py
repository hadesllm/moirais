# morie.fn — function file (hadesllm/morie)
"""Bradley-Terry preference pair probability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_bradley_terry_preference"]


def kamath_bradley_terry_preference(r_w, r_l):
    """
    Bradley-Terry preference pair probability

    Formula: P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )

    Parameters
    ----------
    r_w : array-like
        Input data.
    r_l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p_pref

    References
    ----------
    Kamath Ch 5, Bradley-Terry Preference Model section
    """
    r_w = np.atleast_1d(np.asarray(r_w, dtype=float))
    n = len(r_w)
    result = float(np.mean(r_w))
    se = float(np.std(r_w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bradley-Terry preference pair probability"})


def cheatsheet():
    return "kmbrad: Bradley-Terry preference pair probability"
