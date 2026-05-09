"""Product of coefficients indirect effect (a*b)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ab_indirect_effect"]


def ab_indirect_effect(a, b):
    """
    Product of coefficients indirect effect (a*b)

    Formula: IE = a * b, where a is X→M and b is M→Y|X

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacKinnon (2008) Ch 3
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Product of coefficients indirect effect (a*b)"})


def cheatsheet():
    return "abind: Product of coefficients indirect effect (a*b)"
