# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Blackbox scaling for ideal point estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def blackbox_scaling_basic(
    response_matrix,
    n_dims: int = 2,
    *,
    missing_val: float = float("nan"),
) -> DescriptiveResult:
    """Time discovers truth. — Seneca"""
    R = np.asarray(response_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("response_matrix must be 2D.")
    n_resp, n_items = R.shape

    if np.isnan(missing_val):
        mask = ~np.isnan(R)
    else:
        mask = missing_val != R

    col_means = np.nanmean(R, axis=0)
    for j in range(n_items):
        R[~mask[:, j], j] = col_means[j]

    row_means = R.mean(axis=1, keepdims=True)
    col_means_arr = R.mean(axis=0, keepdims=True)
    grand_mean = R.mean()
    centered = R - row_means - col_means_arr + grand_mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    ideal_points = U[:, :n_dims] * S[:n_dims]
    explained = (S[:n_dims] ** 2).sum() / max((S ** 2).sum(), 1e-14)

    return DescriptiveResult(
        name="blackbox_scaling",
        value={"ideal_points": ideal_points, "explained_variance": float(explained)},
        extra={
            "eigenvalues": S[:min(10, len(S))].tolist(),
            "n_respondents": n_resp,
            "n_items": n_items,
            "n_dims": n_dims,
        },
    )


blkbt = blackbox_scaling_basic


def cheatsheet() -> str:
    return "blackbox_scaling_basic({}) -> Blackbox scaling for ideal points."
