# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Party alignment and cohesion metrics."""
import numpy as np
from ._richresult import RichResult

__all__ = ["party_alignment"]


def party_alignment(x):
    """
    Party alignment and cohesion metrics

    Formula: Rice = |%yea_party - %nay_party|

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
    Armstrong Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Party alignment and cohesion metrics"})


def cheatsheet():
    return "algnm: Party alignment and cohesion metrics"
