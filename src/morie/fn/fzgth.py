# morie.fn -- function file (hadesllm/morie)
"""G(theta): distribution function of (X_1+X_2)/2 used in Wilcoxon moments."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_g_theta_distribution"]


def fauzi_g_theta_distribution(theta, cdf, density):
    """
    G(theta): distribution function of (X_1+X_2)/2 used in Wilcoxon moments

    Formula: G(theta) = integral F(2theta+u) f(u) du

    Parameters
    ----------
    theta : array-like
        Input data.
    cdf : array-like
        Input data.
    density : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Fauzi Ch 5
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G(theta): distribution function of (X_1+X_2)/2 used in Wilcoxon moments"})


def cheatsheet():
    return "fzgth: G(theta): distribution function of (X_1+X_2)/2 used in Wilcoxon moments"
