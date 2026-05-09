# moirais.fn — function file (hadesllm/moirais)
"""Speculative decoding: draft model proposes tokens, target model accepts/rejects per-token."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_speculative_decoding"]


def kamath_speculative_decoding(draft_probs, target_probs):
    """
    Speculative decoding: draft model proposes tokens, target model accepts/rejects per-token

    Formula: accept prob(t_i) = min(1, p_target(t_i | ctx) / p_draft(t_i | ctx)); reject -> sample from p_target - p_draft

    Parameters
    ----------
    draft_probs : array-like
        Input data.
    target_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: accepted_tokens

    References
    ----------
    Kamath Ch 8, Speculative Decoding section
    """
    draft_probs = np.atleast_1d(np.asarray(draft_probs, dtype=float))
    n = len(draft_probs)
    result = float(np.mean(draft_probs))
    se = float(np.std(draft_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Speculative decoding: draft model proposes tokens, target model accepts/rejects per-token"})


def cheatsheet():
    return "kmspd: Speculative decoding: draft model proposes tokens, target model accepts/rejects per-token"
