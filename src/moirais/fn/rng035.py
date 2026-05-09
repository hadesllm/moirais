"""Definition of the discrete-time unit step function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_discrete_unit_step"]


def rangayyan_ch3_discrete_unit_step(n):
    """
    Definition of the discrete-time unit step function.

    Formula: u(n) = 1 for n >= 0, 0 otherwise

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.35, p. 109
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of the discrete-time unit step function."})


def cheatsheet():
    return "rng035: Definition of the discrete-time unit step function."
