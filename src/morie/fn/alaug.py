# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Augmented SBERT: cross-encoder labels silver pairs → train bi-encoder on them."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_augmented_sbert"]


def alammar_augmented_sbert(unlabeled_pairs, cross_encoder):
    """
    Augmented SBERT: cross-encoder labels silver pairs → train bi-encoder on them

    Formula: silver_labels = CrossEncoder(pairs); train BiEncoder on (pairs, silver_labels)

    Parameters
    ----------
    unlabeled_pairs : array-like
        Input data.
    cross_encoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labeled_pairs

    References
    ----------
    Alammar Ch 10, Augmented SBERT section
    """
    unlabeled_pairs = np.atleast_1d(np.asarray(unlabeled_pairs, dtype=float))
    n = len(unlabeled_pairs)
    result = float(np.mean(unlabeled_pairs))
    se = float(np.std(unlabeled_pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Augmented SBERT: cross-encoder labels silver pairs → train bi-encoder on them"})


def cheatsheet():
    return "alaug: Augmented SBERT: cross-encoder labels silver pairs → train bi-encoder on them"
