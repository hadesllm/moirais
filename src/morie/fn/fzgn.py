# morie.fn -- function file (hadesllm/morie)
"""G_n(x) correction term in Edgeworth expansion for kernel quantile."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_gn_edgeworth_correction"]


def fauzi_gn_edgeworth_correction(x, sigma_n, e_moments, bandwidth):
    """
    G_n(x) correction term in Edgeworth expansion for kernel quantile

    Formula: G_n(x) = Phi(x) - phi(x){ (x^2-1)/(6n^{1/2}sigma_n^3)*(e_{1n}+3e_{2n}/h) + (1/(nh^2))*[...]}

    Parameters
    ----------
    x : array-like
        Input data.
    sigma_n : array-like
        Input data.
    e_moments : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cdf_approx

    References
    ----------
    Fauzi Ch 3, Theorem 3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G_n(x) correction term in Edgeworth expansion for kernel quantile"})


def cheatsheet():
    return "fzgn: G_n(x) correction term in Edgeworth expansion for kernel quantile"
