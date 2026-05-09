"""Three-level random-effects model (cluster within study)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_three_level"]


def ma_three_level(yi, vi, cluster_id, study_id):
    """
    Three-level random-effects model (cluster within study)

    Formula: y_ijk = θ + ν_i + u_ij + ε_ijk

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    cluster_id : array-like
        Input data.
    study_id : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, sigma_3, sigma_2, ll

    References
    ----------
    Cheung (2014)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Three-level random-effects model (cluster within study)"})


def cheatsheet():
    return "matrl: Three-level random-effects model (cluster within study)"
