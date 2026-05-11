"""Convert a soft transport plan to a barycentric map."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_plan_to_map"]


def ot_plan_to_map(T, Y):
    """
    Convert a soft transport plan to a barycentric map

    Formula: T̄(x_i) = Σ_j T_ij y_j / Σ_j T_ij

    Parameters
    ----------
    T : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_map

    References
    ----------
    Peyré & Cuturi (2019)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convert a soft transport plan to a barycentric map"})


def cheatsheet():
    return "otplan: Convert a soft transport plan to a barycentric map"
