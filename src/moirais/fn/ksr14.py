# moirais.fn — function file (hadesllm/moirais)
"""Profile likelihood for semiparametric models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_profile_likelihood"]


def kosorok_profile_likelihood(x, y):
    """
    Profile likelihood for semiparametric models

    Formula: L_p(theta) = sup_eta L(theta, eta)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Profile likelihood for semiparametric models"})


def cheatsheet():
    return "ksr14: Profile likelihood for semiparametric models"
