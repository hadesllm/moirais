# morie.fn -- function file (hadesllm/morie)
"""Two-model mixture for adaptation: mixture of parametric and nonparametric prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_two_model_adp"]


def ghosal_two_model_adp(x):
    """
    Two-model mixture for adaptation: mixture of parametric and nonparametric prior

    Formula: Pi = pi_0*Pi_0 + (1-pi_0)*Pi_1, posterior weights adapt to data complexity

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
    Ghosal Ch 10 §10.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-model mixture for adaptation: mixture of parametric and nonparametric prior"})


def cheatsheet():
    return "gh_c10_4: Two-model mixture for adaptation: mixture of parametric and nonparametric prior"
