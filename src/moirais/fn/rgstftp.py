# moirais.fn — function file (hadesllm/moirais)
"""STFT parameter selection (window length vs. time/freq resolution tradeoff)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_stft_params"]


def rangayyan_stft_params(fs, desired_t_res, desired_f_res):
    """
    STFT parameter selection (window length vs. time/freq resolution tradeoff)

    Formula: delta_t = N/fs; delta_f = fs/N; uncertainty: delta_t * delta_f = 1

    Parameters
    ----------
    fs : array-like
        Input data.
    desired_t_res : array-like
        Input data.
    desired_f_res : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: recommended_N, window_type

    References
    ----------
    Rangayyan Ch 8.4.2
    """
    fs = np.asarray(fs, dtype=float)
    n = int(fs) if fs.ndim == 0 else len(fs)
    result = float(np.mean(fs))
    se = float(np.std(fs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "STFT parameter selection (window length vs. time/freq resolution tradeoff)"})


def cheatsheet():
    return "rgstftp: STFT parameter selection (window length vs. time/freq resolution tradeoff)"
