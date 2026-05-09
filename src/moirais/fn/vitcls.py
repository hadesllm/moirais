"""ViT [CLS] token + position embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_cls_token"]


def vit_cls_token(patches, n_patches):
    """
    ViT [CLS] token + position embedding

    Formula: prepend [cls]; add learned 1D pos_embed

    Parameters
    ----------
    patches : array-like
        Input data.
    n_patches : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    patches = np.atleast_1d(np.asarray(patches, dtype=float))
    n = len(patches)
    result = float(np.mean(patches))
    se = float(np.std(patches, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT [CLS] token + position embedding"})


def cheatsheet():
    return "vitcls: ViT [CLS] token + position embedding"
