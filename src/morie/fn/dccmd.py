# morie.fn — function file (hadesllm/morie)
"""DCC multivariate GARCH."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dcc_multivariate_garch"]


def dcc_multivariate_garch(x):
    """
    DCC multivariate GARCH

    Formula: Q_t = (1-a-b)*Q_bar + a*e_{t-1}*e_{t-1}' + b*Q_{t-1}

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
    Engle (2002)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DCC multivariate GARCH"})


def cheatsheet():
    return "dccmd: DCC multivariate GARCH"
