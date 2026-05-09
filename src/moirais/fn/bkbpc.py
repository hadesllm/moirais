# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bits-per-character (BPC): cross-entropy per character of an LM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_bits_per_character"]


def burkov_bits_per_character(ce_loss, n_tokens, n_characters):
    """
    Bits-per-character (BPC): cross-entropy per character of an LM

    Formula: BPC = (L_CE * N_tokens) / (ln(2) * N_characters)

    Parameters
    ----------
    ce_loss : array-like
        Input data.
    n_tokens : array-like
        Input data.
    n_characters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bpc

    References
    ----------
    Burkov Ch 2, Bits-per-Character section
    """
    ce_loss = np.atleast_1d(np.asarray(ce_loss, dtype=float))
    n = len(ce_loss)
    result = float(np.mean(ce_loss))
    se = float(np.std(ce_loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bits-per-character (BPC): cross-entropy per character of an LM"})


def cheatsheet():
    return "bkbpc: Bits-per-character (BPC): cross-entropy per character of an LM"
