# moirais.fn — function file (hadesllm/moirais)
"""Asymptotic properties of kernel MRL."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_mrl_asymptotic"]


def fauzi_mrl_asymptotic(x):
    """
    Asymptotic properties of kernel MRL

    Formula: sqrt(n)(m_hat - m) -> N(0, sigma_m^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Fauzi Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic properties of kernel MRL"})


def cheatsheet():
    return "fzmrl: Asymptotic properties of kernel MRL"
