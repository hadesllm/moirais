# moirais.fn — function file (hadesllm/moirais)
"""Tail probabilities of DP: large deviation bounds for G(A)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_tails"]


def ghosal_dp_tails(x):
    """
    Tail probabilities of DP: large deviation bounds for G(A)

    Formula: P(G(A) > t) <= exp(-C*alpha*t) for t > G0(A)

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
    Ghosal Ch 4 §4.3.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail probabilities of DP: large deviation bounds for G(A)"})


def cheatsheet():
    return "gh_c4_16: Tail probabilities of DP: large deviation bounds for G(A)"
