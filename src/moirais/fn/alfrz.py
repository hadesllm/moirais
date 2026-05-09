# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Layer freezing / gradual unfreezing during fine-tuning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_layer_freezing"]


def alammar_layer_freezing(model, schedule):
    """
    Layer freezing / gradual unfreezing during fine-tuning

    Formula: for l in L..1: train layers {l, ..., L} while freezing {1, ..., l-1}

    Parameters
    ----------
    model : array-like
        Input data.
    schedule : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trained_model

    References
    ----------
    Alammar Ch 11, Layer Freezing / Gradual Unfreezing section
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Layer freezing / gradual unfreezing during fine-tuning"})


def cheatsheet():
    return "alfrz: Layer freezing / gradual unfreezing during fine-tuning"
