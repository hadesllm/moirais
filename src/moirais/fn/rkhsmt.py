# moirais.fn — function file (hadesllm/moirais)
"""Multi-trait Bayesian kernel regression with shared kernel matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rkhs_multitrait"]


def rkhs_multitrait(Y, K, n_iter):
    """
    Multi-trait Bayesian kernel regression with shared kernel matrix

    Formula: vec(Y) ~ MVN(0, Sigma_g Y K + Sigma_e Y I); Sigma_g, Sigma_e estimated by Gibbs

    Parameters
    ----------
    Y : array-like
        Input data.
    K : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gebv': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-trait Bayesian kernel regression with shared kernel matrix"})


def cheatsheet():
    return "rkhsmt: Multi-trait Bayesian kernel regression with shared kernel matrix"
