# morie.fn -- function file (hadesllm/morie)
"""Pinball (quantile) loss for quantile tau."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_pinball_quantile_loss"]


def joseph_pinball_quantile_loss(y, q, tau):
    """
    Pinball (quantile) loss for quantile tau

    Formula: L_tau(y, q) = max(tau*(y - q), (tau - 1)*(y - q))

    Parameters
    ----------
    y : array-like
        Input data.
    q : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Joseph Ch 17, Quantile / Pinball Loss section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pinball (quantile) loss for quantile tau"})


def cheatsheet():
    return "jopql: Pinball (quantile) loss for quantile tau"
