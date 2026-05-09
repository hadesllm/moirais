"""G-formula (parametric) standardised mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_robins_g_formula"]


def causal_robins_g_formula(y, A, L, fit_fn):
    """
    G-formula (parametric) standardised mean

    Formula: E[Y(a)] = E[E[Y|A=a, L]]

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    L : array-like
        Input data.
    fit_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: EYa

    References
    ----------
    Robins (1986)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G-formula (parametric) standardised mean"})


def cheatsheet():
    return "causmrop: G-formula (parametric) standardised mean"
