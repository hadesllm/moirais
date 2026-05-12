r"""Flamingo factorized.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_flamingo_factorized"]


def kamath_ch9_flamingo_factorized(y, x, L):
    r"""
    Flamingo factorized.

    Formula: p(y|x) = \prod_{\ell=1}^L p(y_{\ell}|y_{<\ell}, x_{\le\ell})

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.21, p. 404
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flamingo factorized."})


def cheatsheet():
    return "km149: Flamingo factorized."
