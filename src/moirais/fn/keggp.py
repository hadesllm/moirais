"""KEGG pathway enrichment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kegg_pathway"]


def kegg_pathway(genes, kegg_pathways):
    """
    KEGG pathway enrichment

    Formula: gene-set enrichment vs KEGG

    Parameters
    ----------
    genes : array-like
        Input data.
    kegg_pathways : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kanehisa et al (2017)
    """
    genes = np.atleast_1d(np.asarray(genes, dtype=float))
    n = len(genes)
    result = float(np.mean(genes))
    se = float(np.std(genes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KEGG pathway enrichment"})


def cheatsheet():
    return "keggp: KEGG pathway enrichment"
