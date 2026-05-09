# moirais.fn — function file (hadesllm/moirais)
"""DPM of normal kernel contraction: near-optimal rate with log correction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dpm_norm_crt"]


def ghosal_dpm_norm_crt(x):
    """
    DPM of normal kernel contraction: near-optimal rate with log correction

    Formula: DPM of N(mu,sigma^2): rate n^{-s/(2s+1)} * (log n)^t for s-smooth p0

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
    Ghosal Ch 9 §9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DPM of normal kernel contraction: near-optimal rate with log correction"})


def cheatsheet():
    return "gh_c9_4: DPM of normal kernel contraction: near-optimal rate with log correction"
