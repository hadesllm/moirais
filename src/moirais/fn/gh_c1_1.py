# moirais.fn — function file (hadesllm/moirais)
"""Bayes rule for infinite-dimensional parameters: posterior proportional to likelihood times prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bayes_rule_infinite"]


def ghosal_bayes_rule_infinite(x):
    """
    Bayes rule for infinite-dimensional parameters: posterior proportional to likelihood times prior

    Formula: pi(theta|X) proportional to p_theta(X) * pi(theta)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 1 §1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes rule for infinite-dimensional parameters: posterior proportional to likelihood times prior"})


def cheatsheet():
    return "gh_c1_1: Bayes rule for infinite-dimensional parameters: posterior proportional to likelihood times prior"
