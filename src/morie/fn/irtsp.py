# morie.fn — function file (hadesllm/morie)
"""IRT-based spatial model (2PL ideal points)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["irt_spatial"]


def irt_spatial(x):
    """
    IRT-based spatial model (2PL ideal points)

    Formula: P(yea) = logit^{-1}(alpha_j*(x_i - beta_j))

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
    Armstrong Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IRT-based spatial model (2PL ideal points)"})


def cheatsheet():
    return "irtsp: IRT-based spatial model (2PL ideal points)"
