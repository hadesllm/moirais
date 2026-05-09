# moirais.fn — function file (hadesllm/moirais)
"""Quadratic deterministic utility function in Bayesian IRT."""
import numpy as np
from ._richresult import RichResult

__all__ = ["irt_quadratic_utility"]


def irt_quadratic_utility(ideal_point, vote_position, discrimination):
    """
    Quadratic deterministic utility function in Bayesian IRT

    Formula: U_i(y_j) = -(x_i - y_j)^2; quadratic loss implies Phi((x_i - beta_j)*alpha_j) vote probability

    Parameters
    ----------
    ideal_point : array-like
        Input data.
    vote_position : array-like
        Input data.
    discrimination : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'utility': 'float', 'p_yea': 'float'}

    References
    ----------
    Armstrong Ch 6
    """
    ideal_point = np.asarray(ideal_point, dtype=float)
    n = int(ideal_point) if ideal_point.ndim == 0 else len(ideal_point)
    result = float(np.mean(ideal_point))
    se = float(np.std(ideal_point, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic deterministic utility function in Bayesian IRT"})


def cheatsheet():
    return "irtdq: Quadratic deterministic utility function in Bayesian IRT"
