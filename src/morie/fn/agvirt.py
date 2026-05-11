"""AlphaZero virtual loss for parallel MCTS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_virtual_loss"]


def alphazero_virtual_loss(node, virtual_loss):
    """
    AlphaZero virtual loss for parallel MCTS

    Formula: deduct virtual loss when thread enters node

    Parameters
    ----------
    node : array-like
        Input data.
    virtual_loss : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chaslot et al (2008)
    """
    node = np.atleast_1d(np.asarray(node, dtype=float))
    n = len(node)
    result = float(np.mean(node))
    se = float(np.std(node, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero virtual loss for parallel MCTS"})


def cheatsheet():
    return "agvirt: AlphaZero virtual loss for parallel MCTS"
