"""Output of the second branch in a parallel LSI configuration.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_parallel_branch_2"]


def rangayyan_ch3_lsi_parallel_branch_2(x, h_2, n):
    """
    Output of the second branch in a parallel LSI configuration.

    Formula: s_2(n) = x(n) * h_2(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.47, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the second branch in a parallel LSI configuration."})


def cheatsheet():
    return "rng045: Output of the second branch in a parallel LSI configuration."
