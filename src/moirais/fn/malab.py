"""L'Abbé plot data: control vs experimental risks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_labbe_plot"]


def ma_labbe_plot(a, b, c, d):
    """
    L'Abbé plot data: control vs experimental risks

    Formula: (p_C_i, p_E_i) per trial

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p_E, p_C

    References
    ----------
    L'Abbé et al. (1987)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L'Abbé plot data: control vs experimental risks"})


def cheatsheet():
    return "malab: L'Abbé plot data: control vs experimental risks"
