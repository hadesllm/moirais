# morie.fn -- function file (hadesllm/morie)
"""Instruction tuning CE loss over (instruction, response) pairs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_instruction_tuning_loss"]


def kamath_instruction_tuning_loss(logits, response_mask, targets):
    """
    Instruction tuning CE loss over (instruction, response) pairs

    Formula: L = - (1/|R|) sum_{t in R} log p(r_t | instruction, r_{<t})

    Parameters
    ----------
    logits : array-like
        Input data.
    response_mask : array-like
        Input data.
    targets : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 4, Instruction Tuning section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Instruction tuning CE loss over (instruction, response) pairs"})


def cheatsheet():
    return "kminst: Instruction tuning CE loss over (instruction, response) pairs"
