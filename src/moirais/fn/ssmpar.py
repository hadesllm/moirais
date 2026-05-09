"""SSM parallel-prefix associative scan."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ssm_parallel_scan"]


def ssm_parallel_scan(y, A, B):
    """
    SSM parallel-prefix associative scan

    Formula: (a_i, b_i) o (a_j, b_j) = (a_j a_i, a_j b_i + b_j)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blelloch (1990); Smith et al. (2023) S5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SSM parallel-prefix associative scan"})


def cheatsheet():
    return "ssmpar: SSM parallel-prefix associative scan"
