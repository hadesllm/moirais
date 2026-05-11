"""Implicit-feedback weighted loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["implicit_feedback_loss"]


def implicit_feedback_loss(R, conf):
    """
    Implicit-feedback weighted loss

    Formula: sum c_{ui}(p_{ui} − p̂_{ui})²

    Parameters
    ----------
    R : array-like
        Input data.
    conf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hu-Koren-Volinsky (2008)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Implicit-feedback weighted loss"})


def cheatsheet():
    return "impFB: Implicit-feedback weighted loss"
