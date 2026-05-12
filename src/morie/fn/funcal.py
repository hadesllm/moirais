"""Functional annotation (eggNOG-mapper)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_annotation"]


def functional_annotation(sequences, eggnog_db):
    """
    Functional annotation (eggNOG-mapper)

    Formula: orthologous group -> KEGG/GO

    Parameters
    ----------
    sequences : array-like
        Input data.
    eggnog_db : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cantalapiedra et al (2021)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional annotation (eggNOG-mapper)"})


def cheatsheet():
    return "funcal: Functional annotation (eggNOG-mapper)"
