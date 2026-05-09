"""Perplexity.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_perplexity"]


def kamath_ch8_perplexity(X, N, p_theta):
    """
    Perplexity.

    Formula: \mathrm{PPL}(X) = \exp(-\frac{1}{N}\sum_{i=0}^N \log p_{\theta}(x_i|x_{<i}))

    Parameters
    ----------
    X : array-like
        Input data.
    N : array-like
        Input data.
    p_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.1, p. 322
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perplexity."})


def cheatsheet():
    return "km113: Perplexity."
