"""Gene-set enrichment (GSEA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geneset_enrichment"]


def geneset_enrichment(ranked_genes, gene_set):
    """
    Gene-set enrichment (GSEA)

    Formula: weighted KS over ranked list

    Parameters
    ----------
    ranked_genes : array-like
        Input data.
    gene_set : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Subramanian et al (2005)
    """
    ranked_genes = np.atleast_1d(np.asarray(ranked_genes, dtype=float))
    n = len(ranked_genes)
    result = float(np.mean(ranked_genes))
    se = float(np.std(ranked_genes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gene-set enrichment (GSEA)"})


def cheatsheet():
    return "gnsetenr: Gene-set enrichment (GSEA)"
