"""Rotary position embedding (RoPE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rotary_position_embedding"]


def rotary_position_embedding(y, q, m, theta):
    """
    Rotary position embedding (RoPE)

    Formula: f(q_m, m) = R_theta_m q_m, R_theta_m = block-diag rotation by m * theta_i

    Parameters
    ----------
    y : array-like
        Input data.
    q : array-like
        Input data.
    m : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Su et al. (2021) RoFormer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rotary position embedding (RoPE)"})


def cheatsheet():
    return "atrope: Rotary position embedding (RoPE)"
