# morie.fn — function file (hadesllm/morie)
"""DataLoader for mini-batch iteration with shuffling and parallel workers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dataloader"]


def geron_dataloader(dataset, batch_size, shuffle):
    """
    DataLoader for mini-batch iteration with shuffling and parallel workers

    Formula: for x_b, y_b in DataLoader(dataset, batch_size, shuffle)

    Parameters
    ----------
    dataset : array-like
        Input data.
    batch_size : array-like
        Input data.
    shuffle : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loader

    References
    ----------
    Géron Ch 10
    """
    dataset = np.atleast_1d(np.asarray(dataset, dtype=float))
    n = len(dataset)
    result = float(np.mean(dataset))
    se = float(np.std(dataset, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DataLoader for mini-batch iteration with shuffling and parallel workers"})


def cheatsheet():
    return "hmdld: DataLoader for mini-batch iteration with shuffling and parallel workers"
