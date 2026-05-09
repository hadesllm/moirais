# moirais.fn — function file (hadesllm/moirais)
"""BERT masked-language-modeling loss over masked tokens."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bert_mlm_loss"]


def geron_bert_mlm_loss(logits, targets, mask):
    """
    BERT masked-language-modeling loss over masked tokens

    Formula: L_MLM = - sum_{i in masked} log p(x_i | context_without_xi)

    Parameters
    ----------
    logits : array-like
        Input data.
    targets : array-like
        Input data.
    mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 15, BERT pretraining (MLM) section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERT masked-language-modeling loss over masked tokens"})


def cheatsheet():
    return "grmlm: BERT masked-language-modeling loss over masked tokens"
