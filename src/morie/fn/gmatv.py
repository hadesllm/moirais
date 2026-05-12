# morie.fn -- function file (hadesllm/morie)
"""Genomic relationship matrix (VanRaden 2008, methods 1 and 2)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["grm_vanraden"]


def grm_vanraden(markers, method: int = 1):
    """Genomic relationship matrix per VanRaden (2008).

    Method 1 (default, "G_VR1"):
        Centre Z = M - 2P, where P_j = column allele frequency p_j.
        G = Z Z' / [2 * sum_j p_j (1 - p_j)]

    Method 2 ("G_VR2"):
        Z is centred AND each column scaled by sqrt(2 p_j (1-p_j)).
        G = Z Z' / m, where m is the number of markers.

    Parameters
    ----------
    markers : array-like, shape (n, m)
        Genotype matrix, coded {0,1,2} (number of reference alleles).
    method : {1, 2}, default 1

    Returns
    -------
    RichResult with payload keys:
        estimate : (n,n) ndarray, the GRM
        diag_mean, off_mean : diagnostic averages
        p : allele frequencies, length m
        n, m : sample / marker counts
        method : description string

    References
    ----------
    VanRaden, P. M. (2008). Efficient methods to compute genomic predictions.
        J Dairy Sci, 91(11), 4414-4423.
    Montesinos Lopez et al. (2022), Ch. 3.
    """
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2:
        raise ValueError("`markers` must be a 2D (n × m) array")
    n, m = M.shape
    # Allele frequencies (assume coding 0/1/2)
    p = M.mean(axis=0) / 2.0
    Z = M - 2.0 * p  # centred
    if method == 2:
        scale = np.sqrt(2.0 * p * (1.0 - p))
        scale = np.where(scale > 0, scale, 1.0)
        Z = Z / scale
        denom = float(m)
        method_str = "VanRaden method 2 (per-locus scaled)"
    else:
        denom = float(2.0 * np.sum(p * (1.0 - p)))
        denom = denom if denom > 0 else 1.0
        method_str = "VanRaden method 1 (sum-2pq)"
    G = (Z @ Z.T) / denom
    diag_mean = float(np.mean(np.diag(G)))
    off = G - np.diag(np.diag(G))
    off_mean = float(np.sum(off) / (n * (n - 1))) if n > 1 else 0.0
    return RichResult(
        title="VanRaden Genomic Relationship Matrix",
        summary_lines=[
            ("n (individuals)", n),
            ("m (markers)", m),
            ("mean diag(G)", diag_mean),
            ("mean off-diag(G)", off_mean),
        ],
        payload={
            "estimate": G,
            "diag_mean": diag_mean,
            "off_mean": off_mean,
            "p": p,
            "n": n,
            "m": m,
            "method": method_str,
        },
    )


def cheatsheet():
    return "gmatv: Genomic relationship matrix (VanRaden methods 1 and 2)"


# CANONICAL TEST
# markers = np.array([[0,1,2,0],[2,1,0,1],[1,2,1,2],[0,0,2,1]])
# G = grm_vanraden(markers, method=1).estimate
# expected diag_mean ≈ 1.0 ± slack; G is symmetric, trace(G) ≈ n.
