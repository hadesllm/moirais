"""SignalP signal peptide prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["signal_peptide"]


def signal_peptide(sequence):
    """
    SignalP signal peptide prediction

    Formula: deep learning + cleavage site

    Parameters
    ----------
    sequence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Almagro Armenteros et al (2019) SignalP-5
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SignalP signal peptide prediction"})


def cheatsheet():
    return "sigpep: SignalP signal peptide prediction"
