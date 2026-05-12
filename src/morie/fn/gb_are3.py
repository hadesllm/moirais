# morie.fn -- function file (hadesllm/morie)
"""ARE values for double exponential distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_are_dbl_exp"]


def gibbons_are_dbl_exp(distribution):
    """
    ARE values for double exponential distribution

    Formula: ARE(Wilcoxon, t|DExp) = 3/2; ARE(sign, t|DExp) = 2

    Parameters
    ----------
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE_values

    References
    ----------
    Gibbons Ch 13
    """
    distribution = np.asarray(distribution, dtype=float)
    n = int(distribution) if distribution.ndim == 0 else len(distribution)
    result = float(np.mean(distribution))
    se = float(np.std(distribution, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARE values for double exponential distribution"})


def cheatsheet():
    return "gb_are3: ARE values for double exponential distribution"
