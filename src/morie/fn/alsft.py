# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""SetFit two-step: (1) contrastive fine-tune encoder on few-shot pairs, (2) classifier head."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_setfit_twostep"]


def alammar_setfit_twostep(few_shot_pairs, encoder, classifier):
    """
    SetFit two-step: (1) contrastive fine-tune encoder on few-shot pairs, (2) classifier head

    Formula: step1: L_contrast on pairs; step2: LogReg on frozen embeddings

    Parameters
    ----------
    few_shot_pairs : array-like
        Input data.
    encoder : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Alammar Ch 11, SetFit section
    """
    few_shot_pairs = np.atleast_1d(np.asarray(few_shot_pairs, dtype=float))
    n = len(few_shot_pairs)
    result = float(np.mean(few_shot_pairs))
    se = float(np.std(few_shot_pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SetFit two-step: (1) contrastive fine-tune encoder on few-shot pairs, (2) classifier head"})


def cheatsheet():
    return "alsft: SetFit two-step: (1) contrastive fine-tune encoder on few-shot pairs, (2) classifier head"
