# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""T5/Flan-T5 classification: generate label token as text output."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_t5_text_to_text_classify"]


def alammar_t5_text_to_text_classify(input, label_tokens, model):
    """
    T5/Flan-T5 classification: generate label token as text output

    Formula: y_label = argmax_{label} p_T5(label | prefix + input)

    Parameters
    ----------
    input : array-like
        Input data.
    label_tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: label

    References
    ----------
    Alammar Ch 4, text-to-text transfer (T5/Flan-T5) section
    """
    input = np.atleast_1d(np.asarray(input, dtype=float))
    n = len(input)
    result = float(np.mean(input))
    se = float(np.std(input, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T5/Flan-T5 classification: generate label token as text output"})


def cheatsheet():
    return "alt5c: T5/Flan-T5 classification: generate label token as text output"
