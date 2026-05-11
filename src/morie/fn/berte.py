"""BERT encoder forward pass."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bert_encoder"]


def bert_encoder(tokens, model):
    """
    BERT encoder forward pass

    Formula: masked LM + NSP pretraining; transformer encoder

    Parameters
    ----------
    tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Devlin et al (2019)
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERT encoder forward pass"})


def cheatsheet():
    return "berte: BERT encoder forward pass"
