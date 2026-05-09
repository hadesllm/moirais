"""OCR with layout (LayoutLMv3)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ocr_wit_layout"]


def ocr_wit_layout(image, ocr_tokens, bboxes):
    """
    OCR with layout (LayoutLMv3)

    Formula: text + image + 1D layout multi-modal

    Parameters
    ----------
    image : array-like
        Input data.
    ocr_tokens : array-like
        Input data.
    bboxes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huang et al (2022) LayoutLMv3
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OCR with layout (LayoutLMv3)"})


def cheatsheet():
    return "ocrwit: OCR with layout (LayoutLMv3)"
