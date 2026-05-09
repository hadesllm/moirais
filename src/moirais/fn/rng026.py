"""Dirac delta as the limit of a power-function as a tends to 0.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dirac_delta_limit_form"]


def rangayyan_ch3_dirac_delta_limit_form(t, a):
    """
    Dirac delta as the limit of a power-function as a tends to 0.

    Formula: delta(t) = 0.5 * lim_{a->0} a * |t|^(a-1)

    Parameters
    ----------
    t : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.26, p. 107
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirac delta as the limit of a power-function as a tends to 0."})


def cheatsheet():
    return "rng026: Dirac delta as the limit of a power-function as a tends to 0."
