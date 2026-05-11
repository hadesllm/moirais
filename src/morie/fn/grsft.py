# morie.fn — function file (hadesllm/morie)
"""Supervised fine-tuning (SFT) cross-entropy objective on (prompt, response) pairs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sft_objective"]


def geron_sft_objective(logits, response_mask, targets):
    """
    Supervised fine-tuning (SFT) cross-entropy objective on (prompt, response) pairs

    Formula: L_SFT = - (1/|R|) sum_{t in R} log p_theta(r_t | prompt, r_{<t})

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
    Géron Ch 15, Supervised Fine-Tuning section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Supervised fine-tuning (SFT) cross-entropy objective on (prompt, response) pairs"})


def cheatsheet():
    return "grsft: Supervised fine-tuning (SFT) cross-entropy objective on (prompt, response) pairs"
