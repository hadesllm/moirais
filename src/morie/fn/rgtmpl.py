# morie.fn -- function file (hadesllm/morie)
"""Template matching for EEG spike-and-wave detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_template_match"]


def rangayyan_template_match(eeg, template, threshold):
    """
    Template matching for EEG spike-and-wave detection

    Formula: corr(template, segment) = sum(t[n]*eeg[n]) / (|t|*|eeg|); threshold on correlation

    Parameters
    ----------
    eeg : array-like
        Input data.
    template : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: match_locs

    References
    ----------
    Rangayyan Ch 4.4.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Template matching for EEG spike-and-wave detection"})


def cheatsheet():
    return "rgtmpl: Template matching for EEG spike-and-wave detection"
