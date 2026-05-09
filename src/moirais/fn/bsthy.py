# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Basic space dimensionality test via eigenvalue ratios."""

from __future__ import annotations

from ._containers import DescriptiveResult


def basic_space_dim_test(data, max_dims: int = 5) -> DescriptiveResult:
    """Test dimensionality of the basic space via eigenvalue ratios.

    :param data: Respondent x issue data matrix.
    :param max_dims: Maximum dimensions to examine.
    :return: DescriptiveResult with eigenvalue ratios.

    .. epigraph:: "The world shall know pain." -- Nagato, Naruto
    """
    import numpy as np

    X = np.asarray(data, dtype=float)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    k = min(max_dims, len(eigvals))
    eigvals = eigvals[:k]
    total = eigvals.sum()
    ratios = (eigvals / total).tolist() if total > 0 else [0.0] * k
    return DescriptiveResult(
        name="basic_space_dim_test",
        value=int(np.argmax(np.diff(np.append(ratios, 0)) < -0.1) + 1) if len(ratios) > 1 else 1,
        extra={"eigenvalues": eigvals.tolist(), "variance_ratios": ratios, "max_dims": max_dims},
    )


bsthy = basic_space_dim_test


def cheatsheet() -> str:
    return "basic_space_dim_test({}) -> Basic space dimensionality test via eigenvalue ratios."
