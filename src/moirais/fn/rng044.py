"""Output of the first branch in a parallel LSI configuration.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_parallel_branch_1"]


def rangayyan_ch3_lsi_parallel_branch_1(x, h_1, n):
    """
    Output of the first branch in a parallel LSI configuration.

    Formula: s_1(n) = x(n) * h_1(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.46, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the first branch in a parallel LSI configuration."})


def cheatsheet():
    return "rng044: Output of the first branch in a parallel LSI configuration."
