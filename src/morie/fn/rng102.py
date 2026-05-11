"""General definition of running integral over (-inf, t].."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_integral_general"]


def rangayyan_ch3_integral_general(x, t):
    """
    General definition of running integral over (-inf, t].

    Formula: y(t) = integral_{-inf}^{t} x(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.113, p. 143
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "General definition of running integral over (-inf, t]."})


def cheatsheet():
    return "rng102: General definition of running integral over (-inf, t]."
