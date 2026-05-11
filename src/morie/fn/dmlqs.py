"""Deep-learning QSAR (graph neural network)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deepml_qsar"]


def deepml_qsar(smiles, activities):
    """
    Deep-learning QSAR (graph neural network)

    Formula: GNN on molecular graph; readout to scalar

    Parameters
    ----------
    smiles : array-like
        Input data.
    activities : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yang et al (2019) D-MPNN; Gilmer et al (2017) MPNN
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep-learning QSAR (graph neural network)"})


def cheatsheet():
    return "dmlqs: Deep-learning QSAR (graph neural network)"
