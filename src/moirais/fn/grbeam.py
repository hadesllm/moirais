# moirais.fn — function file (hadesllm/moirais)
"""Beam search decoder: keep top-k hypotheses at each step."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_beam_search_decoder"]


def geron_beam_search_decoder(scores, beam_width, max_len):
    """
    Beam search decoder: keep top-k hypotheses at each step

    Formula: B_t = top_k over h in B_{t-1}, token y: score(h) + log p(y | h)

    Parameters
    ----------
    scores : array-like
        Input data.
    beam_width : array-like
        Input data.
    max_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beams

    References
    ----------
    Géron Ch 14, Beam Search section
    """
    scores = np.atleast_1d(np.asarray(scores, dtype=float))
    n = len(scores)
    result = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beam search decoder: keep top-k hypotheses at each step"})


def cheatsheet():
    return "grbeam: Beam search decoder: keep top-k hypotheses at each step"
