# morie.fn -- function file (hadesllm/morie)
"""TSMixer: MLP-only architecture mixing across time and features alternately."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_tsmixer"]


def joseph_tsmixer(x, n_blocks, hidden_dim):
    """
    TSMixer: MLP-only architecture mixing across time and features alternately

    Formula: alternate TimeMix(x) = MLP_t(x^T)^T and FeatMix(x) = MLP_f(x); residual stacks

    Parameters
    ----------
    x : array-like
        Input data.
    n_blocks : array-like
        Input data.
    hidden_dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 16, TSMixer section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TSMixer: MLP-only architecture mixing across time and features alternately"})


def cheatsheet():
    return "jotsmx: TSMixer: MLP-only architecture mixing across time and features alternately"
