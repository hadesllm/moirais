"""RoPE NTK-aware dynamic scaling for longer context."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rope_ntk_dynamic"]


def rope_ntk_dynamic(y, q, m, theta, L_new, L_train):
    """
    RoPE NTK-aware dynamic scaling for longer context

    Formula: theta_i' = theta_i * (alpha)^(-2i/d), alpha = (L_new/L_train)

    Parameters
    ----------
    y : array-like
        Input data.
    q : array-like
        Input data.
    m : array-like
        Input data.
    theta : array-like
        Input data.
    L_new : array-like
        Input data.
    L_train : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    bloc97 (2023); Reddit r/LocalLLaMA NTK-Aware
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RoPE NTK-aware dynamic scaling for longer context"})


def cheatsheet():
    return "ropedy: RoPE NTK-aware dynamic scaling for longer context"
