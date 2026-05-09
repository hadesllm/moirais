"""VC index (Vapnik-Chervonenkis dimension) computation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class VCIndexResult:
    """Result of VC dimension computation.

    Attributes
    ----------
    vc_dim : int
        The VC dimension (largest set size shattered).
    shatter_counts : dict[int, tuple[int, int]]
        Mapping from set size k to (shattered, total) subsets tested.
    n_points : int
        Number of data points used.
    d_features : int
        Number of features.
    theoretical_bound : int | None
        Theoretical VC dimension for linear classifiers (d + 1),
        or None if not applicable.
    """

    vc_dim: int
    shatter_counts: dict
    n_points: int
    d_features: int
    theoretical_bound: int | None


def vcidx(
    X: np.ndarray,
    *,
    classifier_type: str = "linear",
    max_k: int | None = None,
    max_subsets: int = 1000,
    seed: int | None = None,
) -> VCIndexResult:
    r"""
    Estimate the VC (Vapnik-Chervonenkis) dimension of a function class.

    The VC dimension :math:`V(\mathcal{F})` is the largest integer :math:`k`
    such that there exist :math:`k` points shattered by :math:`\mathcal{F}`:

    .. math::

        V(\mathcal{F}) = \max\bigl\{k : \exists\, x_1,\ldots,x_k \text{ shattered
        by } \mathcal{F}\bigr\}

    For linear classifiers in :math:`\mathbb{R}^d`, the VC dimension is
    :math:`d + 1`. This function verifies shattering empirically for small
    :math:`k` and provides the theoretical bound for linear classifiers.

    The VC dimension controls uniform convergence rates via the
    Vapnik-Chervonenkis inequality:

    .. math::

        P\!\left(\sup_{f \in \mathcal{F}} |P_n f - Pf| > \varepsilon\right)
        \le 8\,S(\mathcal{F}, n)\,e^{-n\varepsilon^2/32}

    where :math:`S(\mathcal{F}, n)` is the shattering coefficient.

    :param X: Data matrix of shape (n, d).
    :param classifier_type: Type of classifier class. Currently ``"linear"``
        (halfspaces). Default ``"linear"``.
    :param max_k: Maximum subset size to test. Default min(d+2, n, 10).
    :param max_subsets: Maximum subsets to sample per size k. Default 1000.
    :param seed: Random seed.
    :return: VCIndexResult with estimated VC dimension and shatter diagnostics.
    :raises ValueError: If X has fewer than 2 rows or unknown classifier_type.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 9.2 (VC classes). Springer.
    DOI:10.1007/978-0-387-74978-5

    Vapnik, V.N. & Chervonenkis, A.Ya. (1971). On the uniform convergence
    of relative frequencies of events to their probabilities.
    *Theory of Probability and its Applications*, 16(2), 264--280.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    if n < 2:
        raise ValueError("X must have at least 2 rows.")
    if classifier_type != "linear":
        raise ValueError(f"Unknown classifier_type '{classifier_type}'. Use 'linear'.")

    rng = np.random.default_rng(seed)
    theoretical = d + 1

    if max_k is None:
        max_k = min(d + 2, n, 10)

    def _can_shatter_linear(points: np.ndarray) -> bool:
        """Check if k points in R^d can be shattered by linear halfspaces."""
        k = points.shape[0]
        ones = np.ones((k, 1))
        aug = np.hstack([ones, points])
        return np.linalg.matrix_rank(aug) == k

    shatter_counts: dict[int, tuple[int, int]] = {}
    vc_dim = 0

    for k in range(1, max_k + 1):
        if k > n:
            break

        total_possible = 1
        for i in range(k):
            total_possible = total_possible * (n - i) // (i + 1)

        if total_possible <= max_subsets:
            subsets = list(combinations(range(n), k))
        else:
            subsets = []
            for _ in range(max_subsets):
                idx = tuple(sorted(rng.choice(n, size=k, replace=False)))
                subsets.append(idx)

        shattered = 0
        for idx in subsets:
            pts = X[list(idx), :]
            if _can_shatter_linear(pts):
                shattered += 1

        shatter_counts[k] = (shattered, len(subsets))
        if shattered > 0:
            vc_dim = k

    return VCIndexResult(
        vc_dim=vc_dim,
        shatter_counts=shatter_counts,
        n_points=n,
        d_features=d,
        theoretical_bound=theoretical,
    )


def cheatsheet() -> str:
    return "vcidx({X}) -> VC dimension / VC index computation."
