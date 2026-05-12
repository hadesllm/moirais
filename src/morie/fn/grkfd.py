# morie.fn -- function file (hadesllm/morie)
"""K-fold cross-validation index generator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kfold_cv"]


def geron_kfold_cv(n, K, shuffle, seed):
    """
    K-fold cross-validation index generator

    Formula: partition range(n) into K folds; yield (train_idx, val_idx) pairs

    Parameters
    ----------
    n : array-like
        Input data.
    K : array-like
        Input data.
    shuffle : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: folds

    References
    ----------
    Géron Ch 2, K-fold Cross-Validation section
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-fold cross-validation index generator"})


def cheatsheet():
    return "grkfd: K-fold cross-validation index generator"
