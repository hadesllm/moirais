"""Marked point pattern summary statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def marked_point_summary(points, marks):
    """Compute summary statistics for a marked point pattern.

    Includes mark correlation and nearest-neighbor mark tests.

    .. epigraph:: "Place of Power. Gotta be." -- Geralt, The Witcher

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    marks : array_like
        Mark values, shape ``(n,)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    pts = np.asarray(points, dtype=np.float64)
    m = np.asarray(marks, dtype=np.float64).ravel()
    n = pts.shape[0]

    D = squareform(pdist(pts))
    np.fill_diagonal(D, np.inf)

    nn_idx = D.argmin(axis=1)
    nn_marks = m[nn_idx]

    corr = float(np.corrcoef(m, nn_marks)[0, 1]) if n > 2 else 0.0

    mark_mean = float(m.mean())
    mark_var = float(m.var())

    idx_upper = np.triu_indices(n, k=1)
    mark_products = m[idx_upper[0]] * m[idx_upper[1]]
    expected_product = mark_mean**2
    kmm = float(mark_products.mean() / expected_product) if expected_product > 0 else 0.0

    return DescriptiveResult(
        name="marked_point_summary",
        value=corr,
        extra={
            "mark_correlation": corr,
            "mark_mean": mark_mean,
            "mark_variance": mark_var,
            "kmm_ratio": kmm,
            "n_points": n,
        },
    )


sgmkd = marked_point_summary


def cheatsheet() -> str:
    return "marked_point_summary({}) -> Marked point pattern summary statistics."
