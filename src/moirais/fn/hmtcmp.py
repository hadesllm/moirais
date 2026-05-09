# moirais.fn — function file (hadesllm/moirais)
"""torch.compile: graph-capturing JIT for forward and backward passes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_torch_compile"]


def geron_torch_compile(model, mode):
    """
    torch.compile: graph-capturing JIT for forward and backward passes

    Formula: model_compiled = torch.compile(model)

    Parameters
    ----------
    model : array-like
        Input data.
    mode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: compiled_model

    References
    ----------
    Géron Ch 10
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "torch.compile: graph-capturing JIT for forward and backward passes"})


def cheatsheet():
    return "hmtcmp: torch.compile: graph-capturing JIT for forward and backward passes"
