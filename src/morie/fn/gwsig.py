# morie.fn -- function file (hadesllm/morie)
"""Genome-wide significance threshold (Bonferroni / BH)."""

__all__ = ["gwsig"]

import numpy as np

from ._containers import GenomicsResult


def gwsig(
    p_values: np.ndarray,
    *,
    method: str = "bonferroni",
    alpha: float = 0.05,
    n_effective: int | None = None,
) -> GenomicsResult:
    """Apply genome-wide significance correction.

    Parameters
    ----------
    p_values : array, shape (p,)
        Raw p-values from GWAS.
    method : {'bonferroni', 'bh', 'sidak'}
        Correction method:
        - 'bonferroni': alpha / m
        - 'bh': Benjamini-Hochberg FDR control
        - 'sidak': 1 - (1 - alpha)^(1/m)
    alpha : float
        Family-wise or FDR-controlled error rate.
    n_effective : int, optional
        Effective number of independent tests (for Bonferroni/Sidak).
        If None, uses total number of tests.

    Returns
    -------
    GenomicsResult
        statistic = number of significant hits,
        extra has 'threshold', 'significant_indices',
        'adjusted_pvalues'.

    References
    ----------
    Pe'er, I., et al. (2008). Estimation of the multiple testing
        burden for genomewide association studies of nearly all
        common variants. Genetic Epidemiology, 32(4), 381-385.
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
        discovery rate. JRSS-B, 57(1), 289-300.
    """
    pv = np.asarray(p_values, dtype=float).ravel()
    m = len(pv)

    if m == 0:
        return GenomicsResult(name="GW_significance", statistic=0.0, n=0)

    m_eff = n_effective if n_effective is not None else m

    if method == "bonferroni":
        threshold = alpha / m_eff
        adjusted = np.minimum(pv * m_eff, 1.0)
        significant = pv < threshold

    elif method == "sidak":
        threshold = 1.0 - (1.0 - alpha) ** (1.0 / m_eff)
        adjusted = 1.0 - (1.0 - pv) ** m_eff
        adjusted = np.minimum(adjusted, 1.0)
        significant = pv < threshold

    elif method == "bh":
        order = np.argsort(pv)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)

        adjusted = np.minimum(pv * m / ranks, 1.0)

        sorted_adj = adjusted[order]
        for i in range(m - 2, -1, -1):
            sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
        adjusted[order] = sorted_adj

        significant = adjusted < alpha
        threshold = float(np.max(pv[significant])) if np.any(significant) else 0.0

    else:
        raise ValueError(f"Unknown method: {method}")

    sig_idx = np.where(significant)[0]
    n_sig = int(np.sum(significant))

    return GenomicsResult(
        name="GW_significance",
        statistic=float(n_sig),
        n=m,
        extra={
            "threshold": float(threshold),
            "significant_indices": sig_idx.tolist(),
            "adjusted_pvalues": adjusted.tolist(),
            "method": method,
            "alpha": alpha,
            "n_effective": m_eff,
        },
    )


def cheatsheet() -> str:
    return "gwsig(pvals) -> Genome-wide significance threshold."
