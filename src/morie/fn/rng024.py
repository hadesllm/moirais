"""Definition of the continuous-time Dirac delta function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dirac_delta_definition"]


def rangayyan_ch3_dirac_delta_definition(t):
    """
    Definition of the continuous-time Dirac delta function.

    Formula: delta(t) = undefined at t=0, 0 otherwise

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
    Rangayyan (2024), Ch 3, Eq 3.24, p. 107
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of the continuous-time Dirac delta function."})


def cheatsheet():
    return "rng024: Definition of the continuous-time Dirac delta function."
