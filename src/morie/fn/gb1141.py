# morie.fn — function file (hadesllm/morie)
"""Relations between T, E(R), Kendall tau and Spearman rho."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_tau_rho_relation"]


def gibbons_tau_rho_relation(tau, rho):
    """
    Relations between T, E(R), Kendall tau and Spearman rho

    Formula: -1 <= 3T/2 - 1/2 <= r_s <= (1+2T)/2; 3tau-1 <= 2rho <= 1+2tau

    Parameters
    ----------
    tau : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bounds

    References
    ----------
    Gibbons Ch 11.4
    """
    tau = np.asarray(tau, dtype=float)
    n = int(tau) if tau.ndim == 0 else len(tau)
    result = float(np.mean(tau))
    se = float(np.std(tau, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relations between T, E(R), Kendall tau and Spearman rho"})


def cheatsheet():
    return "gb1141: Relations between T, E(R), Kendall tau and Spearman rho"
