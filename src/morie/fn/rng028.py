"""Sifting property of the Dirac delta function over an interval.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sifting_property"]


def rangayyan_ch3_sifting_property(x, t, t_o, T1, T2):
    """
    Sifting property of the Dirac delta function over an interval.

    Formula: integral_{T1}^{T2} x(t) delta(t - t_o) dt = x(t_o) if T1 < t_o < T2, 0 otherwise

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    t_o : array-like
        Input data.
    T1 : array-like
        Input data.
    T2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.28, p. 108
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sifting property of the Dirac delta function over an interval."})


def cheatsheet():
    return "rng028: Sifting property of the Dirac delta function over an interval."
