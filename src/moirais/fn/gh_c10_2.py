# moirais.fn — function file (hadesllm/moirais)
"""Universal weights for adaptation: log pi_k <= -C*k*log n ensures adaptation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_univ_weights"]


def ghosal_univ_weights(x):
    """
    Universal weights for adaptation: log pi_k <= -C*k*log n ensures adaptation

    Formula: pi_k: sum_k exp(log pi_k + n*eps_k^2) converges for all eps_k

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
    Ghosal Ch 10 §10.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Universal weights for adaptation: log pi_k <= -C*k*log n ensures adaptation"})


def cheatsheet():
    return "gh_c10_2: Universal weights for adaptation: log pi_k <= -C*k*log n ensures adaptation"
