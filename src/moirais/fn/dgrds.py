# moirais.fn — function file (hadesllm/moirais)
"""Degree distribution and power-law test."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def degree_distribution(adj_matrix: np.ndarray) -> DescriptiveResult:
    """Compute degree distribution and test for power-law scaling.

    The power-law exponent is estimated via maximum likelihood for the
    discrete Pareto distribution (Clauset et al., 2009):

    .. math::

        \\hat{\\alpha} = 1 + n
        \\left[\\sum_{i=1}^{n} \\ln \\frac{k_i}{k_{\\min} - 0.5}\\right]^{-1}

    A Kolmogorov-Smirnov test compares the empirical distribution to
    the fitted power-law.

    Parameters
    ----------
    adj_matrix : ndarray
        Square adjacency matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean degree.  ``extra`` has ``degrees``,
        ``degree_counts``, ``alpha_hat`` (MLE exponent), ``ks_stat``,
        and ``ks_pvalue``.

    References
    ----------
    Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
    Power-law distributions in empirical data.
    *SIAM Review*, 51(4), 661--703.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square.")

    A_bin = (A != 0).astype(np.float64)
    np.fill_diagonal(A_bin, 0)
    degrees = A_bin.sum(axis=1).astype(int)

    unique, counts = np.unique(degrees, return_counts=True)
    degree_counts = dict(zip(unique.tolist(), counts.tolist()))
    mean_deg = float(np.mean(degrees))

    pos = degrees[degrees > 0]
    alpha_hat = np.nan
    ks_stat = np.nan
    ks_pval = np.nan

    if len(pos) >= 5:
        k_min = float(pos.min())
        alpha_hat = 1.0 + len(pos) / np.sum(np.log(pos / (k_min - 0.5)))

        fitted_cdf = 1 - (np.sort(pos) / k_min) ** (1 - alpha_hat)
        empirical_cdf = np.arange(1, len(pos) + 1) / len(pos)
        ks_stat = float(np.max(np.abs(fitted_cdf - empirical_cdf)))
        _, ks_pval = _st.kstest(pos, "pareto", args=(alpha_hat - 1, 0, k_min))
        ks_pval = float(ks_pval)

    return DescriptiveResult(
        name="DegreeDistribution",
        value=mean_deg,
        extra={
            "degrees": degrees,
            "degree_counts": degree_counts,
            "alpha_hat": float(alpha_hat),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "n_nodes": A.shape[0],
        },
    )


dgrds = degree_distribution


def cheatsheet() -> str:
    return "degree_distribution({}) -> Degree distribution and power-law test."
