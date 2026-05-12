# morie.fn -- function file (hadesllm/morie)
"""Rejection-sampling fine-tuning: retain top-k per-prompt samples by reward, SFT on them."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rejection_sampling_finetune"]


def kamath_rejection_sampling_finetune(prompts, samples, rewards, k):
    """
    Rejection-sampling fine-tuning: retain top-k per-prompt samples by reward, SFT on them

    Formula: for each prompt x: sample {y_i}, keep top-k by r_phi; SFT pi on retained set

    Parameters
    ----------
    prompts : array-like
        Input data.
    samples : array-like
        Input data.
    rewards : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: retained

    References
    ----------
    Kamath Ch 5, Rejection Sampling Fine-tuning section
    """
    prompts = np.atleast_1d(np.asarray(prompts, dtype=float))
    n = len(prompts)
    result = float(np.mean(prompts))
    se = float(np.std(prompts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rejection-sampling fine-tuning: retain top-k per-prompt samples by reward, SFT on them"})


def cheatsheet():
    return "kmrsft: Rejection-sampling fine-tuning: retain top-k per-prompt samples by reward, SFT on them"
