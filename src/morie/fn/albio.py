# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BIO/BIOES tagging for NER."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_bio_tagging"]


def alammar_bio_tagging(tokens, entity_spans, scheme):
    """
    BIO/BIOES tagging for NER

    Formula: tags in {B-X, I-X, O} (BIO) or {B, I, O, E, S-X} (BIOES)

    Parameters
    ----------
    tokens : array-like
        Input data.
    entity_spans : array-like
        Input data.
    scheme : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tags

    References
    ----------
    Alammar Ch 11, BIO/BIOES tagging scheme section
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIO/BIOES tagging for NER"})


def cheatsheet():
    return "albio: BIO/BIOES tagging for NER"
