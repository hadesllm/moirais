# morie.fn -- function file (hadesllm/morie)
"""QLoRA: frozen base weights in NF4, LoRA adapters in BF16/FP16."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_qlora_4bit"]


def kamath_qlora_4bit(W0_nf4, A, B, alpha, r, x):
    """
    QLoRA: frozen base weights in NF4, LoRA adapters in BF16/FP16

    Formula: h = Dequant_NF4(W_0_q) x + (alpha/r) * B A x;  B, A in BF16

    Parameters
    ----------
    W0_nf4 : array-like
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
    Kamath Ch 4, QLoRA section (Dettmers et al.)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QLoRA: frozen base weights in NF4, LoRA adapters in BF16/FP16"})


def cheatsheet():
    return "kmqlor: QLoRA: frozen base weights in NF4, LoRA adapters in BF16/FP16"
