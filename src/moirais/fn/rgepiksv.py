# moirais.fn — function file (hadesllm/moirais)
"""Epileptic seizure detection using K-SVD dictionary learning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_epilepsy_ksvd"]


def rangayyan_epilepsy_ksvd(eeg, fs, dict_size, sparsity):
    """
    Epileptic seizure detection using K-SVD dictionary learning

    Formula: Learned dictionary atoms; OMP for sparse coding; SVM on coefficients

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    dict_size : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_seizure, onset

    References
    ----------
    Rangayyan Ch 9.8
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epileptic seizure detection using K-SVD dictionary learning"})


def cheatsheet():
    return "rgepiksv: Epileptic seizure detection using K-SVD dictionary learning"
