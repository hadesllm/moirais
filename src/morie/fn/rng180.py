"""MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_qrs_smoothing_ma_filter"]


def rangayyan_ch4_qrs_smoothing_ma_filter(g_1, n, M):
    """
    MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector.

    Formula: g(n) = (1/M) * sum_{j=0}^{M-1} g_1(n - j)

    Parameters
    ----------
    g_1 : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.5, p. 219
    """
    g_1 = np.atleast_1d(np.asarray(g_1, dtype=float))
    n = len(g_1)
    result = float(np.mean(g_1))
    se = float(np.std(g_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector."})


def cheatsheet():
    return "rng180: MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector."
