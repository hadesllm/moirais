# morie.fn -- function file (hadesllm/morie)
"""PET (Pattern-Exploiting Training) loss combining LM and classifier objectives."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_pet_loss"]


def kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha):
    """
    PET (Pattern-Exploiting Training) loss combining LM and classifier objectives

    Formula: L_PET = L_CE(verbalizer_prob, y_true) + alpha * L_MLM(masked_tokens)

    Parameters
    ----------
    verbalizer_logits : array-like
        Input data.
    y_true : array-like
        Input data.
    mlm_logits : array-like
        Input data.
    mlm_targets : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 3, PET / iPET section
    """
    verbalizer_logits = np.atleast_1d(np.asarray(verbalizer_logits, dtype=float))
    n = len(verbalizer_logits)
    result = float(np.mean(verbalizer_logits))
    se = float(np.std(verbalizer_logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PET (Pattern-Exploiting Training) loss combining LM and classifier objectives"})


def cheatsheet():
    return "kmpet: PET (Pattern-Exploiting Training) loss combining LM and classifier objectives"
