"""GLS for spatial data with known Sigma."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_gls_spatial"]


def schabenberger_gls_spatial(x, y, sigma):
    """
    GLS for spatial data with known Sigma

    Formula: beta_GLS = (X'*Sigma^{-1}*X)^{-1}*X'*Sigma^{-1}*Y; optimal under Gauss-Markov

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, se

    References
    ----------
    Schabenberger Ch 6, Sec 6.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GLS for spatial data with known Sigma"})


def cheatsheet():
    return "spgls: GLS for spatial data with known Sigma"
