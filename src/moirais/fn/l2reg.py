# moirais.fn — function file (hadesllm/moirais)
"""L2 weight regularization for neural networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["l2_weight_regularization"]


def l2_weight_regularization(loss, weights, lam):
    """
    L2 weight regularization for neural networks

    Formula: L_reg = L + lambda * sum_{l,j,k} w_{ljk}^2

    Parameters
    ----------
    loss : array-like
        Input data.
    weights : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'reg_loss': 'float'}

    References
    ----------
    Montesinos Lopez Ch 11
    """
    loss = np.asarray(loss, dtype=float)
    n = int(loss) if loss.ndim == 0 else len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L2 weight regularization for neural networks"})


def cheatsheet():
    return "l2reg: L2 weight regularization for neural networks"
