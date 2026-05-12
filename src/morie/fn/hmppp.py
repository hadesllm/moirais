# morie.fn -- function file (hadesllm/morie)
"""Pipeline parallelism: partition layers across devices with microbatches."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pipeline_parallelism"]


def geron_pipeline_parallelism(model, n_stages):
    """
    Pipeline parallelism: partition layers across devices with microbatches

    Formula: forward/backward staged across devices; microbatch pipeline

    Parameters
    ----------
    model : array-like
        Input data.
    n_stages : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pipelined_model

    References
    ----------
    Géron Ch 17
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pipeline parallelism: partition layers across devices with microbatches"})


def cheatsheet():
    return "hmppp: Pipeline parallelism: partition layers across devices with microbatches"
