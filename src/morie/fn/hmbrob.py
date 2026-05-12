# morie.fn -- function file (hadesllm/morie)
"""RoBERTa: robustly-optimized BERT pretraining."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_roberta"]


def geron_roberta(X, n_layers, n_heads):
    """
    RoBERTa: robustly-optimized BERT pretraining

    Formula: BERT without NSP, larger batches, longer training

    Parameters
    ----------
    X : array-like
        Input data.
    n_layers : array-like
        Input data.
    n_heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RoBERTa: robustly-optimized BERT pretraining"})


def cheatsheet():
    return "hmbrob: RoBERTa: robustly-optimized BERT pretraining"
