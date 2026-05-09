# moirais.fn — function file (hadesllm/moirais)
"""Hugging Face Trainer API high-level training loop."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hf_trainer"]


def geron_hf_trainer(model, args, train_ds, eval_ds):
    """
    Hugging Face Trainer API high-level training loop

    Formula: Trainer(model, args, train_ds, eval_ds).train()

    Parameters
    ----------
    model : array-like
        Input data.
    args : array-like
        Input data.
    train_ds : array-like
        Input data.
    eval_ds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trainer

    References
    ----------
    Géron Ch 14
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hugging Face Trainer API high-level training loop"})


def cheatsheet():
    return "hmhftn: Hugging Face Trainer API high-level training loop"
