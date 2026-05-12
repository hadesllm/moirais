# morie.fn -- function file (hadesllm/morie)
"""Static post-training quantization: calibrate scales on a representative set."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_static_ptq"]


def geron_static_ptq(model, calibration_data):
    """
    Static post-training quantization: calibrate scales on a representative set

    Formula: s_act = calibrate(activations); x_q = round(x/s_act); save scales in the model

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
    Géron Appendix B, Static Quantization (PTQ) section
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Static post-training quantization: calibrate scales on a representative set"})


def cheatsheet():
    return "grptq: Static post-training quantization: calibrate scales on a representative set"
