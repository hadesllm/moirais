# moirais.fn — function file (hadesllm/moirais)
"""Euclidean distance utility for spatial voting model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["euclidean_utility"]


def euclidean_utility(ideal_point, policy_position):
    """
    Euclidean distance utility for spatial voting model

    Formula: U_i(x_j) = -||x_i* - x_j||^2 = -(sum_d (x_{id} - x_{jd})^2)

    Parameters
    ----------
    ideal_point : array-like
        Input data.
    policy_position : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'utility': 'float'}

    References
    ----------
    Armstrong Ch 1
    """
    ideal_point = np.asarray(ideal_point, dtype=float)
    n = int(ideal_point) if ideal_point.ndim == 0 else len(ideal_point)
    result = float(np.mean(ideal_point))
    se = float(np.std(ideal_point, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Euclidean distance utility for spatial voting model"})


def cheatsheet():
    return "eudst: Euclidean distance utility for spatial voting model"
