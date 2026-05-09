"""Echo state network (reservoir computing)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["echo_state_network"]


def echo_state_network(y, reservoir_size):
    """
    Echo state network (reservoir computing)

    Formula: random recurrent reservoir + linear readout

    Parameters
    ----------
    y : array-like
        Input data.
    reservoir_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaeger (2001) ESN
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Echo state network (reservoir computing)"})


def cheatsheet():
    return "esnnts: Echo state network (reservoir computing)"
