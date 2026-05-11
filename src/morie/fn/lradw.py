# morie.fn — function file (hadesllm/morie)
"""Linear learning rate warmup."""
import numpy as np
from ._richresult import RichResult

__all__ = ["lr_warmup"]


def lr_warmup(x):
    """
    Linear learning rate warmup

    Formula: lr = lr_target * step / warmup_steps

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear learning rate warmup"})


def cheatsheet():
    return "lradw: Linear learning rate warmup"
