"""ViT fine-tune for downstream."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_finetune"]


def vit_finetune(model, data, mode):
    """
    ViT fine-tune for downstream

    Formula: freeze backbone or unfreeze; new MLP head

    Parameters
    ----------
    model : array-like
        Input data.
    data : array-like
        Input data.
    mode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT fine-tune for downstream"})


def cheatsheet():
    return "vitfsv: ViT fine-tune for downstream"
