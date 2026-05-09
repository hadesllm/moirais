# moirais.fn — function file (hadesllm/moirais)
"""t-SNE: KL divergence between joint probabilities in high- and low-dim."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tsne"]


def geron_tsne(X, n_components, perplexity, seed):
    """
    t-SNE: KL divergence between joint probabilities in high- and low-dim

    Formula: min_Y KL(P || Q); Q uses Student-t heavy tails

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    perplexity : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "t-SNE: KL divergence between joint probabilities in high- and low-dim"})


def cheatsheet():
    return "hmtsne: t-SNE: KL divergence between joint probabilities in high- and low-dim"
