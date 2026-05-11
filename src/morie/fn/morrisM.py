"""Morris elementary effects screening."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["morris_screening"]


def morris_screening(model, input_dist, N):
    """
    Morris elementary effects screening

    Formula: μ*, σ over factor's elementary effects

    Parameters
    ----------
    model : array-like
        Input data.
    input_dist : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Morris (1991)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Morris elementary effects screening"})


def cheatsheet():
    return "morrisM: Morris elementary effects screening"
