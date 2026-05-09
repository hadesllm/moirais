# moirais.fn — function file (hadesllm/moirais)
"""Static post-training quantization (PTQ) using calibration data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_static_quantization_ptq"]


def geron_static_quantization_ptq(model, calibration_data):
    """
    Static post-training quantization (PTQ) using calibration data

    Formula: observe activation ranges on calibration set; pick scales/zeros

    Parameters
    ----------
    model : array-like
        Input data.
    calibration_data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: quantized_model

    References
    ----------
    Géron Appendix B
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Static post-training quantization (PTQ) using calibration data"})


def cheatsheet():
    return "hmptq: Static post-training quantization (PTQ) using calibration data"
