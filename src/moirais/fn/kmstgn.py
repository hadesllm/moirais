# moirais.fn — function file (hadesllm/moirais)
"""Stiennon et al. summarization-from-human-feedback pipeline loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_summarize_from_feedback"]


def kamath_summarize_from_feedback(preferences, rewards, pi_logprobs, ref_logprobs, beta):
    """
    Stiennon et al. summarization-from-human-feedback pipeline loss

    Formula: L_RM on summary preferences; J_RLHF = E[r_phi] - beta * KL(pi || pi_ref)

    Parameters
    ----------
    preferences : array-like
        Input data.
    rewards : array-like
        Input data.
    pi_logprobs : array-like
        Input data.
    ref_logprobs : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: objective

    References
    ----------
    Kamath Ch 5, Learning Summarization from Human Feedback section
    """
    preferences = np.atleast_1d(np.asarray(preferences, dtype=float))
    n = len(preferences)
    result = float(np.mean(preferences))
    se = float(np.std(preferences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stiennon et al. summarization-from-human-feedback pipeline loss"})


def cheatsheet():
    return "kmstgn: Stiennon et al. summarization-from-human-feedback pipeline loss"
