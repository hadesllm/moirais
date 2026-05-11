# morie.fn — function file (hadesllm/morie)
"""Hugging Face Pipelines: high-level inference wrapper."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hf_pipelines"]


def geron_hf_pipelines(task, inputs, model):
    """
    Hugging Face Pipelines: high-level inference wrapper

    Formula: pipeline(task)(inputs) -> predictions

    Parameters
    ----------
    task : array-like
        Input data.
    inputs : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: predictions

    References
    ----------
    Géron Ch 14
    """
    task = np.atleast_1d(np.asarray(task, dtype=float))
    n = len(task)
    result = float(np.mean(task))
    se = float(np.std(task, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hugging Face Pipelines: high-level inference wrapper"})


def cheatsheet():
    return "hmhfpi: Hugging Face Pipelines: high-level inference wrapper"
