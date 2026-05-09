# moirais.fn — function file (hadesllm/moirais)
"""Sample entropy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_sample_entropy"]


def rangayyan_sample_entropy(x):
    """
    Sample entropy

    Formula: SampEn = -ln(A/B), A=matches of length m+1, B=m

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
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample entropy"})


def cheatsheet():
    return "rgsam: Sample entropy"
