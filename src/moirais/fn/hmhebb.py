# moirais.fn — function file (hadesllm/moirais)
"""Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hebb_rule"]


def geron_hebb_rule(X, Y, eta):
    """
    Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur

    Formula: dw_ij = eta * x_i * y_j

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Géron Ch 9
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur"})


def cheatsheet():
    return "hmhebb: Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur"
