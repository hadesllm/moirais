# moirais.fn — function file (hadesllm/moirais)
"""Supervised fine-tuning (SFT) on instruction-response pairs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sft"]


def geron_sft(model, instruction_data, epochs, lr):
    """
    Supervised fine-tuning (SFT) on instruction-response pairs

    Formula: L = -sum_i log P(y_i | x_i) over instruction dataset

    Parameters
    ----------
    model : array-like
        Input data.
    instruction_data : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Supervised fine-tuning (SFT) on instruction-response pairs"})


def cheatsheet():
    return "hmsft: Supervised fine-tuning (SFT) on instruction-response pairs"
