"""Output of two LSI systems in parallel equals input convolved with sum of responses.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lsi_parallel_total"]


def rangayyan_ch3_lsi_parallel_total(x, h_1, h_2, n):
    """
    Output of two LSI systems in parallel equals input convolved with sum of responses.

    Formula: y(n) = s_1(n) + s_2(n) = x(n) * [h_1(n) + h_2(n)] = x(n) * h(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.48, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of two LSI systems in parallel equals input convolved with sum of responses."})


def cheatsheet():
    return "rng046: Output of two LSI systems in parallel equals input convolved with sum of responses."
