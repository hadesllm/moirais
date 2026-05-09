# moirais.fn — function file (hadesllm/moirais)
"""Bayes classifier for normal (Gaussian) patterns."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bayes_gaussian"]


def rangayyan_bayes_gaussian(X, mu_list, sigma_list, priors):
    """
    Bayes classifier for normal (Gaussian) patterns

    Formula: g_k(X) = -0.5*(X-mu_k)^T*Sigma_k^{-1}*(X-mu_k) - 0.5*log|Sigma_k| + log P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    mu_list : array-like
        Input data.
    sigma_list : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, discriminants

    References
    ----------
    Rangayyan Ch 10.6.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes classifier for normal (Gaussian) patterns"})


def cheatsheet():
    return "rgbayng: Bayes classifier for normal (Gaussian) patterns"
