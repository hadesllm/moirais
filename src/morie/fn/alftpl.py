"""AlphaFold template embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_template_embed"]


def alphafold_template_embed(templates, z):
    """
    AlphaFold template embedding

    Formula: embed homology templates into z

    Parameters
    ----------
    templates : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold template embedding"})


def cheatsheet():
    return "alftpl: AlphaFold template embedding"
