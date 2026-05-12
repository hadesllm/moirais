# morie.fn -- function file (hadesllm/morie)
"""Bits-per-byte: cross-entropy divided by bytes-per-token averaged over the corpus."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_bits_per_byte"]


def kamath_bits_per_byte(ce_loss, n_tokens, n_bytes):
    """
    Bits-per-byte: cross-entropy divided by bytes-per-token averaged over the corpus

    Formula: bpb = (L_CE * N_tokens) / (ln(2) * N_bytes)

    Parameters
    ----------
    ce_loss : array-like
        Input data.
    n_tokens : array-like
        Input data.
    n_bytes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bpb

    References
    ----------
    Kamath Ch 8, Bits-per-byte section
    """
    ce_loss = np.atleast_1d(np.asarray(ce_loss, dtype=float))
    n = len(ce_loss)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bits-per-byte: cross-entropy divided by bytes-per-token averaged over the corpus"})
    estimate = np.median(ce_loss)
    se = 1.2533 * np.std(ce_loss, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bits-per-byte: cross-entropy divided by bytes-per-token averaged over the corpus"})


def cheatsheet():
    return "kmbpb: Bits-per-byte: cross-entropy divided by bytes-per-token averaged over the corpus"
