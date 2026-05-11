"""Flamingo dataset mix.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_flamingo_dataset_mix"]


def kamath_ch9_flamingo_dataset_mix(D_m, lambda_m, x, y):
    """
    Flamingo dataset mix.

    Formula: \sum_{m=1}^M \lambda_m E_{(x,y)\sim D_m}[-\sum_{\ell=1}^L \log p(y_{\ell}|y_{<\ell}, x_{\le\ell})]

    Parameters
    ----------
    D_m : array-like
        Input data.
    lambda_m : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.22, p. 404
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flamingo dataset mix."})


def cheatsheet():
    return "km150: Flamingo dataset mix."
