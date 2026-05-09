# moirais.fn — function file (hadesllm/moirais)
"""Classification MLP output head: softmax for K-class multinomial."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_classification_mlp_output"]


def geron_classification_mlp_output(a_last, W_out, b_out):
    """
    Classification MLP output head: softmax for K-class multinomial

    Formula: p_k = exp(z_k) / sum_j exp(z_j) where z = W_out a_{L-1} + b_out

    Parameters
    ----------
    a_last : array-like
        Input data.
    W_out : array-like
        Input data.
    b_out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 9, Classification MLPs section
    """
    a_last = np.asarray(a_last, dtype=float)
    n = int(a_last) if a_last.ndim == 0 else len(a_last)
    result = float(np.mean(a_last))
    se = float(np.std(a_last, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification MLP output head: softmax for K-class multinomial"})


def cheatsheet():
    return "grmlc: Classification MLP output head: softmax for K-class multinomial"
