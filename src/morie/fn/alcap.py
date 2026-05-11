# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Image captioning pipeline: visual encoder → projector → LLM decoder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_image_captioning_pipeline"]


def alammar_image_captioning_pipeline(img, visual_encoder, projector, llm):
    """
    Image captioning pipeline: visual encoder → projector → LLM decoder

    Formula: z = Proj(VisEnc(img));  caption = LLM(inputs=[z; prompt])

    Parameters
    ----------
    img : array-like
        Input data.
    visual_encoder : array-like
        Input data.
    projector : array-like
        Input data.
    llm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: caption

    References
    ----------
    Alammar Ch 9, Image Captioning Pipeline section
    """
    img = np.atleast_1d(np.asarray(img, dtype=float))
    n = len(img)
    result = float(np.mean(img))
    se = float(np.std(img, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Image captioning pipeline: visual encoder → projector → LLM decoder"})


def cheatsheet():
    return "alcap: Image captioning pipeline: visual encoder → projector → LLM decoder"
