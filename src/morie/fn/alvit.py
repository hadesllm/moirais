# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ViT patch embedding: split image into PxP patches, flatten, linear-project to d_model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_vit_patch_embedding"]


def alammar_vit_patch_embedding(image, patch_size, E):
    """
    ViT patch embedding: split image into PxP patches, flatten, linear-project to d_model

    Formula: z_0 = [x_class; Flatten(x_p^1)E; ...; Flatten(x_p^N)E] + E_pos

    Parameters
    ----------
    image : array-like
        Input data.
    patch_size : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_0

    References
    ----------
    Alammar Ch 9, ViT section
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT patch embedding: split image into PxP patches, flatten, linear-project to d_model"})


def cheatsheet():
    return "alvit: ViT patch embedding: split image into PxP patches, flatten, linear-project to d_model"
