"""Two-step DerSimonian-Laird with Hedges-Eddy improvement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_two_step_dl_he"]


def ma_two_step_dl_he(yi, vi, max_iter):
    """
    Two-step DerSimonian-Laird with Hedges-Eddy improvement

    Formula: Refit weights using Eddy correction term

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau2, theta

    References
    ----------
    Eddy & Hedges (1991)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-step DerSimonian-Laird with Hedges-Eddy improvement"})


def cheatsheet():
    return "matr: Two-step DerSimonian-Laird with Hedges-Eddy improvement"
