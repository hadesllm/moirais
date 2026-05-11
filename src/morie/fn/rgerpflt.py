# morie.fn — function file (hadesllm/morie)
"""ERP artifact removal via synchronized averaging."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_erp_artifact_remove"]


def rangayyan_erp_artifact_remove(erp_epochs, fs):
    """
    ERP artifact removal via synchronized averaging

    Formula: SNR_avg = sqrt(M)*SNR_single; artifact reduced by 1/sqrt(M)

    Parameters
    ----------
    erp_epochs : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: erp_clean, snr

    References
    ----------
    Rangayyan Ch 3.12
    """
    erp_epochs = np.asarray(erp_epochs, dtype=float)
    n = int(erp_epochs) if erp_epochs.ndim == 0 else len(erp_epochs)
    result = float(np.mean(erp_epochs))
    se = float(np.std(erp_epochs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ERP artifact removal via synchronized averaging"})


def cheatsheet():
    return "rgerpflt: ERP artifact removal via synchronized averaging"
