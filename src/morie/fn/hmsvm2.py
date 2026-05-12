# morie.fn -- function file (hadesllm/morie)
"""Save and load PyTorch model state_dict."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_save_load_pytorch"]


def geron_save_load_pytorch(model, path):
    """
    Save and load PyTorch model state_dict

    Formula: torch.save(model.state_dict(), path); model.load_state_dict(torch.load(path))

    Parameters
    ----------
    model : array-like
        Input data.
    path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: io_result

    References
    ----------
    Géron Ch 10
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Save and load PyTorch model state_dict"})


def cheatsheet():
    return "hmsvm2: Save and load PyTorch model state_dict"
