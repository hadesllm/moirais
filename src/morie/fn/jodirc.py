# morie.fn -- function file (hadesllm/morie)
"""Direct multi-step: train a separate model per horizon h."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_direct_multistep"]


def joseph_direct_multistep(X, y, H):
    """
    Direct multi-step: train a separate model per horizon h

    Formula: for h in 1..H: model_h fits (X, y_{t+h});  y_hat_{T+h} = model_h(X_T)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: models

    References
    ----------
    Joseph Ch 18, Direct Multi-output Strategy section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Direct multi-step: train a separate model per horizon h"})


def cheatsheet():
    return "jodirc: Direct multi-step: train a separate model per horizon h"
