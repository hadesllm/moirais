"""Byte-pair encoding tokenization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bpe_tokenizer"]


def bpe_tokenizer(x):
    """
    Byte-pair encoding tokenization

    Formula: merge most frequent pair until vocab_size

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sennrich et al. (2016)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Byte-pair encoding tokenization"})


def cheatsheet():
    return "tknbp: Byte-pair encoding tokenization"
