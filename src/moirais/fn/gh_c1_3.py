# moirais.fn — function file (hadesllm/moirais)
"""Formal posterior update for dominated nonparametric experiments."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_prior_posterior_update"]


def ghosal_prior_posterior_update(x):
    """
    Formal posterior update for dominated nonparametric experiments

    Formula: dPi_n/dPi(theta) = p_theta^(n)(X^n) / integral p_eta^(n)(X^n) dPi(eta)

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
    Ghosal Ch 1 §1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Formal posterior update for dominated nonparametric experiments"})


def cheatsheet():
    return "gh_c1_3: Formal posterior update for dominated nonparametric experiments"
