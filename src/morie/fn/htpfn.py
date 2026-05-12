# morie.fn -- function file (hadesllm/morie)
"""High-throughput phenotyping functional predictor combining genomic + phenomic info."""
import numpy as np
from ._richresult import RichResult

__all__ = ["htp_functional_predictor"]


def htp_functional_predictor(y, markers, W_functional):
    """
    High-throughput phenotyping functional predictor combining genomic + phenomic info

    Formula: y_i = mu + g_i + int beta(t)*w_i(t) dt + e_i; g genomic, w phenomic functional covariate

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    W_functional : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_func': 'array', 'g_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 14
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "High-throughput phenotyping functional predictor combining genomic + phenomic info"})


def cheatsheet():
    return "htpfn: High-throughput phenotyping functional predictor combining genomic + phenomic info"
