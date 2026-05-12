# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Task-specific classification head over the [CLS] token hidden state."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_classification_head"]


def alammar_classification_head(h_cls, W_cls, b):
    """
    Task-specific classification head over the [CLS] token hidden state

    Formula: logits = W_cls * h_CLS + b;  y_hat = argmax(softmax(logits))

    Parameters
    ----------
    h_cls : array-like
        Input data.
    W_cls : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logits

    References
    ----------
    Alammar Ch 4, Task-Specific Classification Head section
    """
    h_cls = np.atleast_1d(np.asarray(h_cls, dtype=float))
    n = len(h_cls)
    result = float(np.mean(h_cls))
    se = float(np.std(h_cls, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Task-specific classification head over the [CLS] token hidden state"})


def cheatsheet():
    return "alclsh: Task-specific classification head over the [CLS] token hidden state"
