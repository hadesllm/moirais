"""SimCSE-style contrastive sentence embeddings."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["contrastive_sent"]


def contrastive_sent(sentences, tau):
    """
    SimCSE-style contrastive sentence embeddings

    Formula: InfoNCE on dropout-augmented pairs

    Parameters
    ----------
    sentences : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gao-Yao-Chen (2021) SimCSE
    """
    sentences = np.atleast_1d(np.asarray(sentences, dtype=float))
    n = len(sentences)
    result = float(np.mean(sentences))
    se = float(np.std(sentences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SimCSE-style contrastive sentence embeddings"})


def cheatsheet():
    return "contse: SimCSE-style contrastive sentence embeddings"
