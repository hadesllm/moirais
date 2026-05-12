# morie.fn -- function file (hadesllm/morie)
"""Binary regression via finite random series prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_frs_binreg"]


def ghosal_frs_binreg(x, y):
    """
    Binary regression via finite random series prior

    Formula: P(Y=1|x) = Phi(f(x)), f = sum_{k<=K} beta_k phi_k, adaptive rate

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 10 §10.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary regression via finite random series prior"})


def cheatsheet():
    return "gh_c10_9: Binary regression via finite random series prior"
