# morie.fn -- function file (hadesllm/morie)
"""Probit stick-breaking process: V_k(x) = Phi(a_k + b_k'x) for covariate-dependent DP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_probit_sbp"]


def ghosal_probit_sbp(x):
    """
    Probit stick-breaking process: V_k(x) = Phi(a_k + b_k'x) for covariate-dependent DP

    Formula: G(x,.) = sum_k w_k(x) delta_{theta_k}, w_k(x) via Phi(linear predictor)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 14 §14.9.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Probit stick-breaking process: V_k(x) = Phi(a_k + b_k'x) for covariate-dependent DP"})


def cheatsheet():
    return "gh_c14_20: Probit stick-breaking process: V_k(x) = Phi(a_k + b_k'x) for covariate-dependent DP"
