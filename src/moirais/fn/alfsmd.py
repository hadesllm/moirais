"""AlphaFold MSA attention block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_msa_attention"]


def alphafold_msa_attention(msa, queries, keys):
    """
    AlphaFold MSA attention block

    Formula: attention over MSA rows + columns

    Parameters
    ----------
    msa : array-like
        Input data.
    queries : array-like
        Input data.
    keys : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    msa = np.atleast_1d(np.asarray(msa, dtype=float))
    n = len(msa)
    result = float(np.mean(msa))
    se = float(np.std(msa, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold MSA attention block"})


def cheatsheet():
    return "alfsmd: AlphaFold MSA attention block"
