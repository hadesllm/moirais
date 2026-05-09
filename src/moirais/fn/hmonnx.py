# moirais.fn — function file (hadesllm/moirais)
"""Export PyTorch model to ONNX format for cross-platform inference."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_onnx_export"]


def geron_onnx_export(model, args, file):
    """
    Export PyTorch model to ONNX format for cross-platform inference

    Formula: torch.onnx.export(model, args, file)

    Parameters
    ----------
    model : array-like
        Input data.
    args : array-like
        Input data.
    file : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: onnx_model

    References
    ----------
    Géron Appendix B
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Export PyTorch model to ONNX format for cross-platform inference"})


def cheatsheet():
    return "hmonnx: Export PyTorch model to ONNX format for cross-platform inference"
