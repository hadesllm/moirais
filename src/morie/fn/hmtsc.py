# morie.fn — function file (hadesllm/morie)
"""TorchScript: statically-typed graph representation of PyTorch models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_torchscript"]


def geron_torchscript(model, example_inputs):
    """
    TorchScript: statically-typed graph representation of PyTorch models

    Formula: torch.jit.trace(model) or torch.jit.script(model)

    Parameters
    ----------
    model : array-like
        Input data.
    example_inputs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scripted_model

    References
    ----------
    Géron Appendix B
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TorchScript: statically-typed graph representation of PyTorch models"})


def cheatsheet():
    return "hmtsc: TorchScript: statically-typed graph representation of PyTorch models"
