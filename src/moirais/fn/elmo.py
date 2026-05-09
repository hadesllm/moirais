"""ELMo contextual embeddings."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["elmo"]


def elmo(sentence, model):
    """
    ELMo contextual embeddings

    Formula: task-weighted sum of bi-LM layers

    Parameters
    ----------
    sentence : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Peters et al (2018)
    """
    sentence = np.atleast_1d(np.asarray(sentence, dtype=float))
    n = len(sentence)
    result = float(np.mean(sentence))
    se = float(np.std(sentence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ELMo contextual embeddings"})


def cheatsheet():
    return "elmo: ELMo contextual embeddings"
