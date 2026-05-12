"""Co-occurrence matrix analysis. 'We are Venom.' -- Venom"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cooccurrence_matrix(
    data: np.ndarray,
    *,
    binary: bool = True,
    normalize: bool = True,
) -> DescriptiveResult:
    r"""Compute a co-occurrence (symbiosis) matrix from binary or count data.

    Given a (samples x features) matrix, computes pairwise co-occurrence
    counts :math:`C_{ij} = \\sum_k x_{ki} x_{kj}` and derived metrics:
    Jaccard similarity, Dice coefficient, and pointwise mutual information.

    Parameters
    ----------
    data : np.ndarray
        (n x p) matrix of observations.
    binary : bool
        If True, binarize input (> 0 -> 1).
    normalize : bool
        If True, return normalised (Jaccard) co-occurrence.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``cooccurrence`` (p x p), ``jaccard`` (p x p),
        ``dice`` (p x p), ``pmi`` (p x p).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2D")
    n, p = X.shape

    if binary:
        X = (X > 0).astype(float)

    C = X.T @ X

    row_sums = np.diag(C)
    union = row_sums[:, None] + row_sums[None, :] - C
    union[union == 0] = 1
    jaccard = C / union

    sum_pairs = row_sums[:, None] + row_sums[None, :]
    sum_pairs[sum_pairs == 0] = 1
    dice = 2 * C / sum_pairs

    marginal = row_sums / n
    expected = marginal[:, None] * marginal[None, :] * n
    expected[expected == 0] = 1e-10
    pmi = np.log2(C / expected + 1e-30)
    np.fill_diagonal(pmi, 0)

    if normalize:
        result_matrix = jaccard
    else:
        result_matrix = C

    return DescriptiveResult(
        name="cooccurrence_matrix",
        value={
            "cooccurrence": C,
            "jaccard": jaccard,
            "dice": dice,
            "pmi": pmi,
        },
        extra={"n": n, "p": p, "binary": binary},
    )


venmm = cooccurrence_matrix


def cheatsheet() -> str:
    return "cooccurrence_matrix({}) -> Co-occurrence matrix analysis. 'We are Venom.' -- Venom"
