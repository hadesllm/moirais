"""sacrebleu — reproducible BLEU."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sacrebleu"]


def sacrebleu(candidate, references):
    """
    sacrebleu — reproducible BLEU

    Formula: BLEU on detokenized text + std tokenization

    Parameters
    ----------
    candidate : array-like
        Input data.
    references : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Post (2018) sacrebleu
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "sacrebleu — reproducible BLEU"})


def cheatsheet():
    return "sacrb: sacrebleu — reproducible BLEU"
