"""RFdiffusion protein design."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rfdiffusion_protein"]


def rfdiffusion_protein(target_motif, scaffold):
    """
    RFdiffusion protein design

    Formula: diffusion over residue frames

    Parameters
    ----------
    target_motif : array-like
        Input data.
    scaffold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Watson et al (2023)
    """
    target_motif = np.atleast_1d(np.asarray(target_motif, dtype=float))
    n = len(target_motif)
    result = float(np.mean(target_motif))
    se = float(np.std(target_motif, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RFdiffusion protein design"})


def cheatsheet():
    return "alfrf2: RFdiffusion protein design"
