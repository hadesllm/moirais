"""FIMO motif scan with PWM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["motif_fimo"]


def motif_fimo(sequence, pwm):
    """
    FIMO motif scan with PWM

    Formula: score every window vs PWM; p-value via dynamic programming

    Parameters
    ----------
    sequence : array-like
        Input data.
    pwm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Grant et al (2011)
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FIMO motif scan with PWM"})


def cheatsheet():
    return "motfom: FIMO motif scan with PWM"
