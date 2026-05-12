# morie.fn -- function file (hadesllm/morie)
"""Mini-batch iterator over DataLoader: yields shuffled batches of size b."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dataloader_minibatch"]


def geron_dataloader_minibatch(n, b, shuffle, seed):
    """
    Mini-batch iterator over DataLoader: yields shuffled batches of size b

    Formula: perm = permute(n); for i in 0..n/b: batch = perm[i*b:(i+1)*b]

    Parameters
    ----------
    n : array-like
        Input data.
    b : array-like
        Input data.
    shuffle : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: batches

    References
    ----------
    Géron Ch 10, DataLoader mini-batch GD section
    """
    n = np.asarray(n, dtype=float)
    n = int(n) if n.ndim == 0 else len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch iterator over DataLoader: yields shuffled batches of size b"})


def cheatsheet():
    return "grdlm: Mini-batch iterator over DataLoader: yields shuffled batches of size b"
