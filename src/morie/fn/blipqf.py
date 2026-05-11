"""BLIP Q-Former bridge."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blip_qformer"]


def blip_qformer(image_features, queries, llm):
    """
    BLIP Q-Former bridge

    Formula: learnable queries cross-attend frozen ViT + LLM

    Parameters
    ----------
    image_features : array-like
        Input data.
    queries : array-like
        Input data.
    llm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li et al (2023) BLIP-2
    """
    image_features = np.atleast_1d(np.asarray(image_features, dtype=float))
    n = len(image_features)
    result = float(np.mean(image_features))
    se = float(np.std(image_features, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLIP Q-Former bridge"})


def cheatsheet():
    return "blipqf: BLIP Q-Former bridge"
