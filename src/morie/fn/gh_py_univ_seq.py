# morie.fn — function file (hadesllm/morie)
"""PY universal sequence via GEM distribution: residual allocation from PY."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_py_universal_sequence"]


def ghosal_py_universal_sequence(x):
    """
    PY universal sequence via GEM distribution: residual allocation from PY

    Formula: (V_k) iid Beta(1-d, theta+k*d) for PY(d,theta,G0), V_0=1

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PY universal sequence via GEM distribution: residual allocation from PY"})


def cheatsheet():
    return "gh_py_univ_seq: PY universal sequence via GEM distribution: residual allocation from PY"
