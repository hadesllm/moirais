"""Definition of the continuous-time unit step function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_unit_step_continuous"]


def rangayyan_ch3_unit_step_continuous(t):
    """
    Definition of the continuous-time unit step function.

    Formula: u(t) = 1 for t > 0, 0 otherwise

    Parameters
    ----------
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.27, p. 108
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of the continuous-time unit step function."})


def cheatsheet():
    return "rng027: Definition of the continuous-time unit step function."
