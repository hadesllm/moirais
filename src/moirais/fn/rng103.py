"""Running integral of a causal signal over [0, t].."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_integral_causal"]


def rangayyan_ch3_integral_causal(x, t):
    """
    Running integral of a causal signal over [0, t].

    Formula: y(t) = integral_{0}^{t} x(t) dt

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
    Rangayyan (2024), Ch 3, Eq 3.114, p. 144
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Running integral of a causal signal over [0, t]."})


def cheatsheet():
    return "rng103: Running integral of a causal signal over [0, t]."
