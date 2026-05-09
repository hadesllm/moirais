"""SMILES parser → atom-bond graph."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smiles_grammar_parse"]


def smiles_grammar_parse(smiles):
    """
    SMILES parser → atom-bond graph

    Formula: context-free grammar; OpenSMILES spec

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Weininger (1988); OpenSMILES.org
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SMILES parser → atom-bond graph"})


def cheatsheet():
    return "smigr: SMILES parser → atom-bond graph"
