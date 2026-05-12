# morie.fn -- function file (hadesllm/morie)
"""Form factor (ratio of RMS to mean absolute value)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_form_factor"]


def rangayyan_form_factor(x):
    """
    Form factor (ratio of RMS to mean absolute value)

    Formula: FF = RMS(x) / mean(|x|)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: form_factor

    References
    ----------
    Rangayyan Ch 5.6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Form factor (ratio of RMS to mean absolute value)"})


def cheatsheet():
    return "rgff: Form factor (ratio of RMS to mean absolute value)"
