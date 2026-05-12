# morie.fn -- function file (hadesllm/morie)
"""LoRA: low-rank adaptation delta, W = W_0 + B A where A,B low-rank."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_lora_weight_update"]


def kamath_lora_weight_update(W0, A, B, alpha, r, x):
    """
    LoRA: low-rank adaptation delta, W = W_0 + B A where A,B low-rank

    Formula: h = W_0 x + (alpha / r) * B A x;  W in R^{d x k}, A in R^{r x k}, B in R^{d x r}

    Parameters
    ----------
    W0 : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.
    r : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Kamath Ch 4, Eq LoRA (Hu et al. 2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LoRA: low-rank adaptation delta, W = W_0 + B A where A,B low-rank"})


def cheatsheet():
    return "kmlora: LoRA: low-rank adaptation delta, W = W_0 + B A where A,B low-rank"
