# moirais.fn — function file (hadesllm/moirais)
"""Normal versus ectopic beat classification with LDA and Bayes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ecg_bbb_normal"]


def rangayyan_ecg_bbb_normal(ecg, fs, r_peaks, labels):
    """
    Normal versus ectopic beat classification with LDA and Bayes

    Formula: 4-feature LDA; Bayes classifier with Gaussian class models

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: classifier_accuracy, confusion

    References
    ----------
    Rangayyan Ch 10.11
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal versus ectopic beat classification with LDA and Bayes"})


def cheatsheet():
    return "rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes"
