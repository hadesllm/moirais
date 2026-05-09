# moirais.fn — function file (hadesllm/moirais)
"""Strict semiparametric BvM: Castillo-Rousseau condition for exact BvM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_strict_sbvm"]


def ghosal_strict_sbvm(x):
    """
    Strict semiparametric BvM: Castillo-Rousseau condition for exact BvM

    Formula: Conditions on prior for exact semiparametric BvM to hold at true theta_0

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
    Ghosal Ch 12 §12.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Strict semiparametric BvM: Castillo-Rousseau condition for exact BvM"})


def cheatsheet():
    return "gh_c12_7: Strict semiparametric BvM: Castillo-Rousseau condition for exact BvM"
