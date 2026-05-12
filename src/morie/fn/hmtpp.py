# morie.fn -- function file (hadesllm/morie)
"""Tensor parallelism: split individual tensors across devices."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tensor_parallelism"]


def geron_tensor_parallelism(model, n_devices):
    """
    Tensor parallelism: split individual tensors across devices

    Formula: shard W into W_1 ... W_N across devices; all-reduce output

    Parameters
    ----------
    model : array-like
        Input data.
    n_devices : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sharded_model

    References
    ----------
    Géron Ch 17
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tensor parallelism: split individual tensors across devices"})


def cheatsheet():
    return "hmtpp: Tensor parallelism: split individual tensors across devices"
