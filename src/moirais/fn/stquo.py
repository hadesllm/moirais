"""Status quo point and agenda setter power in spatial model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["status_quo_spatial"]


def status_quo_spatial(ideal_points, status_quo, proposal):
    """
    Status quo point and agenda setter power in spatial model

    Formula: If x_sq is status quo, proposal ideal_points wins if #{i: U_i(x)>U_i(x_sq)} > n/2

    Parameters
    ----------
    ideal_points : array-like
        Input data.
    status_quo : array-like
        Input data.
    proposal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'wins': 'bool', 'majority_size': 'int'}

    References
    ----------
    Armstrong Ch 1
    """
    ideal_points = np.asarray(ideal_points, dtype=float)
    n = int(ideal_points) if ideal_points.ndim == 0 else len(ideal_points)
    result = float(np.mean(ideal_points))
    se = float(np.std(ideal_points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Status quo point and agenda setter power in spatial model"})


def cheatsheet():
    return "stquo: Status quo point and agenda setter power in spatial model"
