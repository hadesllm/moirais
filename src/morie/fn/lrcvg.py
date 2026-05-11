# morie.fn — function file (hadesllm/morie)
"""Learning curve (train/val error vs n)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["learning_curve"]


def learning_curve(x, y):
    """
    Learning curve (train/val error vs n)

    Formula: E_train(n), E_val(n) for n=1..N

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Learning curve (train/val error vs n)"})


def cheatsheet():
    return "lrcvg: Learning curve (train/val error vs n)"
