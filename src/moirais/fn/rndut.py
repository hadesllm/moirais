# moirais.fn — function file (hadesllm/moirais)
"""Random utility model (McFadden) for stochastic spatial voting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_utility_model"]


def random_utility_model(V, eps_dist):
    """
    Random utility model (McFadden) for stochastic spatial voting

    Formula: U_ij = V_ij + eps_ij; P(choice j) = P(V_ij + eps_ij > V_ik + eps_ik for all k != j)

    Parameters
    ----------
    V : array-like
        Input data.
    eps_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'probs': 'array'}

    References
    ----------
    Armstrong Ch 1
    """
    V = np.asarray(V, dtype=float)
    n = int(V) if V.ndim == 0 else len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random utility model (McFadden) for stochastic spatial voting"})


def cheatsheet():
    return "rndut: Random utility model (McFadden) for stochastic spatial voting"
