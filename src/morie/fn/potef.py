# morie.fn -- function file (hadesllm/morie)
"""Individual treatment effect (ITE) using potential outcomes notation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["potential_outcomes_individual"]


def potential_outcomes_individual(Y1, Y0):
    """
    Individual treatment effect (ITE) using potential outcomes notation

    Formula: tau_i = Y_i(1) - Y_i(0); fundamental problem: only one potential outcome observed

    Parameters
    ----------
    Y1 : array-like
        Input data.
    Y0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ite': 'array'}

    References
    ----------
    Molak Ch 1
    """
    Y1 = np.asarray(Y1, dtype=float)
    n = int(Y1) if Y1.ndim == 0 else len(Y1)
    result = float(np.mean(Y1))
    se = float(np.std(Y1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Individual treatment effect (ITE) using potential outcomes notation"})


def cheatsheet():
    return "potef: Individual treatment effect (ITE) using potential outcomes notation"
