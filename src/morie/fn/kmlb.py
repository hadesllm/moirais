# morie.fn — function file (hadesllm/morie)
"""MoE auxiliary load-balancing loss (Shazeer / Switch)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_moe_load_balance_loss"]


def kamath_moe_load_balance_loss(fractions, gate_means, N, alpha):
    """
    MoE auxiliary load-balancing loss (Shazeer / Switch)

    Formula: L_aux = alpha * N * sum_i (f_i * P_i);  f_i = fraction-to-i, P_i = mean-gate-to-i

    Parameters
    ----------
    fractions : array-like
        Input data.
    gate_means : array-like
        Input data.
    N : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 2, Auxiliary Load-Balancing Loss section
    """
    fractions = np.atleast_1d(np.asarray(fractions, dtype=float))
    n = len(fractions)
    result = float(np.mean(fractions))
    se = float(np.std(fractions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MoE auxiliary load-balancing loss (Shazeer / Switch)"})


def cheatsheet():
    return "kmlb: MoE auxiliary load-balancing loss (Shazeer / Switch)"
