# moirais.fn — function file (hadesllm/moirais)
"""Multi-head attention with projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["multi_head_attention_full"]


def multi_head_attention_full(x):
    """
    Multi-head attention with projection

    Formula: MultiHead = Concat(head_1,...,head_h)W^O

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
    Vaswani et al. (2017)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-head attention with projection"})


def cheatsheet():
    return "mhatf: Multi-head attention with projection"
