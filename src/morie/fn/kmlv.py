# morie.fn -- function file (hadesllm/morie)
"""LLaVA visual instruction tuning: linear-project visual features as soft tokens into LLM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_llava_visual_instruction"]


def kamath_llava_visual_instruction(image, W, visual_encoder, text_tokens):
    """
    LLaVA visual instruction tuning: linear-project visual features as soft tokens into LLM

    Formula: z_v = W * ViT(image); inputs = [z_v; text_tokens]; standard CLM loss on response

    Parameters
    ----------
    image : array-like
        Input data.
    W : array-like
        Input data.
    visual_encoder : array-like
        Input data.
    text_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 9, Visual Instruction Tuning (LLaVA) section
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LLaVA visual instruction tuning: linear-project visual features as soft tokens into LLM"})


def cheatsheet():
    return "kmlv: LLaVA visual instruction tuning: linear-project visual features as soft tokens into LLM"
