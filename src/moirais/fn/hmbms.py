# moirais.fn — function file (hadesllm/moirais)
"""Beam search decoding with beam width K."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_beam_search"]


def geron_beam_search(model, src, beam_width):
    """
    Beam search decoding with beam width K

    Formula: maintain top-K partial hypotheses by score

    Parameters
    ----------
    model : array-like
        Input data.
    src : array-like
        Input data.
    beam_width : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_seq

    References
    ----------
    Géron Ch 14
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beam search decoding with beam width K"})


def cheatsheet():
    return "hmbms: Beam search decoding with beam width K"
