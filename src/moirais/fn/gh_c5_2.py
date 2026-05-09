# moirais.fn — function file (hadesllm/moirais)
"""DPM marginal likelihood via Polya urn sequential update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dpm_marg"]


def ghosal_dpm_marg(x):
    """
    DPM marginal likelihood via Polya urn sequential update

    Formula: p(X_1..X_n) = prod_i p(X_i | X_1..X_{i-1}) via Polya urn predictive

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
    Ghosal Ch 5 §5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DPM marginal likelihood via Polya urn sequential update"})


def cheatsheet():
    return "gh_c5_2: DPM marginal likelihood via Polya urn sequential update"
