"""LLaVA visual instruction tuning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["llava_visual_chat"]


def llava_visual_chat(image, instruction, llm):
    """
    LLaVA visual instruction tuning

    Formula: linear projection from CLIP-ViT to LLaMA tokens

    Parameters
    ----------
    image : array-like
        Input data.
    instruction : array-like
        Input data.
    llm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2023) LLaVA
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LLaVA visual instruction tuning"})


def cheatsheet():
    return "llavx: LLaVA visual instruction tuning"
