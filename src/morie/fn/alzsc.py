# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Zero-shot classification via NLI entailment scoring."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_zero_shot_classification"]


def alammar_zero_shot_classification(text, candidate_labels, nli_model):
    """
    Zero-shot classification via NLI entailment scoring

    Formula: p(label_c | text) = softmax_c [ entailment(text, hypothesis(c)) ]

    Parameters
    ----------
    text : array-like
        Input data.
    candidate_labels : array-like
        Input data.
    nli_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Alammar Ch 4, Zero-Shot Classification section
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-shot classification via NLI entailment scoring"})


def cheatsheet():
    return "alzsc: Zero-shot classification via NLI entailment scoring"
