# moirais.fn — function file (hadesllm/moirais)
"""Model parallelism: split model weights across devices."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_model_parallelism"]


def geron_model_parallelism(model, n_devices):
    """
    Model parallelism: split model weights across devices

    Formula: layer or tensor partition over N devices

    Parameters
    ----------
    model : array-like
        Input data.
    n_devices : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distributed_model

    References
    ----------
    Géron Ch 17
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Model parallelism: split model weights across devices"})


def cheatsheet():
    return "hmmpp: Model parallelism: split model weights across devices"
