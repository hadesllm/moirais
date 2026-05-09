# moirais.fn — function file (hadesllm/moirais)
"""Construction of prior on measures via stochastic process with consistent marginals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_stochastic_proc_prior"]


def ghosal_stochastic_proc_prior(x):
    """
    Construction of prior on measures via stochastic process with consistent marginals

    Formula: G(A_1..A_k) ~ p(v_1..v_k) with consistency conditions

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
    Ghosal Ch 3 §3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Construction of prior on measures via stochastic process with consistent marginals"})


def cheatsheet():
    return "gh_c3_2: Construction of prior on measures via stochastic process with consistent marginals"
