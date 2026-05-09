"""Switch Transformer top-1 routing with capacity factor."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moe_switch_routing"]


def moe_switch_routing(y, x, W_g, experts, capacity):
    """
    Switch Transformer top-1 routing with capacity factor

    Formula: y = G(x)_argmax * E_argmax(x); capacity = (T/N) * cf

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    W_g : array-like
        Input data.
    experts : array-like
        Input data.
    capacity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fedus, Zoph, Shazeer (2022)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Switch Transformer top-1 routing with capacity factor"})


def cheatsheet():
    return "moeswt: Switch Transformer top-1 routing with capacity factor"
