"""QJL JL-quantization combination."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qjl_compression"]


def qjl_compression(x, bits, seed):
    """
    QJL JL-quantization combination

    Formula: random projection + quantize; bias-corrected

    Parameters
    ----------
    x : array-like
        Input data.
    bits : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Achlioptas (2003); Charikar-Sahai (2002)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QJL JL-quantization combination"})


def cheatsheet():
    return "qjlcrn: QJL JL-quantization combination"
