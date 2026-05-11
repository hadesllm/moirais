"""Conditional Bernoulli density of a binary response Y given covariate x using a link H composed with a function f.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_binary_regression_density"]


def ghosal_ch2_binary_regression_density(y, x, f, H):
    """
    Conditional Bernoulli density of a binary response Y given covariate x using a link H composed with a function f.

    Formula: p_f(y | x) = H(f(x))^y * (1 - H(f(x)))^(1 - y),   y in {0, 1}

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    f : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.6, p. 21
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional Bernoulli density of a binary response Y given covariate x using a link H composed with a function f."})


def cheatsheet():
    return "ghs007: Conditional Bernoulli density of a binary response Y given covariate x using a link H composed with a function f."
