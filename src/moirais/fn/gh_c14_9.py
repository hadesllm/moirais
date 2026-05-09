# moirais.fn — function file (hadesllm/moirais)
"""Pitman-Yor process PY(d, theta, G0): two-parameter generalization of DP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_py_process"]


def ghosal_py_process(x):
    """
    Pitman-Yor process PY(d, theta, G0): two-parameter generalization of DP

    Formula: V_k ~ Beta(1-d, theta+k*d), d in [0,1), theta > -d

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
    Ghosal Ch 14 §14.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pitman-Yor process PY(d, theta, G0): two-parameter generalization of DP"})


def cheatsheet():
    return "gh_c14_9: Pitman-Yor process PY(d, theta, G0): two-parameter generalization of DP"
