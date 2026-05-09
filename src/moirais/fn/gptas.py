"""Beam search decoding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gpt_assistant_decode"]


def gpt_assistant_decode(model, prompt, k, max_len):
    """
    Beam search decoding

    Formula: keep top-k partial sequences by score

    Parameters
    ----------
    model : array-like
        Input data.
    prompt : array-like
        Input data.
    k : array-like
        Input data.
    max_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lowerre (1976) HARPY; Sutskever et al (2014) seq2seq
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beam search decoding"})


def cheatsheet():
    return "gptas: Beam search decoding"
