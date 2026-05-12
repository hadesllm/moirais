# morie.fn -- function file (hadesllm/morie)
"""Decompose disparity (Blinder-Oaxaca)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def disparity_decompose(
    outcomes_a: np.ndarray | list[float],
    X_a: np.ndarray,
    outcomes_b: np.ndarray | list[float],
    X_b: np.ndarray,
) -> DescriptiveResult:
    """Blinder-Oaxaca decomposition of outcome disparity between two groups.

    Decomposes gap into 'explained' (due to covariate differences)
    and 'unexplained' (due to coefficient differences) components.

    Parameters
    ----------
    outcomes_a : array-like
        Outcomes for group A.
    X_a : ndarray
        Covariate matrix for group A (n_a x k).
    outcomes_b : array-like
        Outcomes for group B.
    X_b : ndarray
        Covariate matrix for group B (n_b x k).

    Returns
    -------
    DescriptiveResult
    """
    ya = np.asarray(outcomes_a, dtype=float)
    yb = np.asarray(outcomes_b, dtype=float)
    Xa = np.asarray(X_a, dtype=float)
    Xb = np.asarray(X_b, dtype=float)

    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xb.ndim == 1:
        Xb = Xb.reshape(-1, 1)
    if len(ya) < 3 or len(yb) < 3:
        raise ValueError("Need at least 3 observations per group")

    Xa_aug = np.column_stack([np.ones(len(ya)), Xa])
    Xb_aug = np.column_stack([np.ones(len(yb)), Xb])

    beta_a = np.linalg.lstsq(Xa_aug, ya, rcond=None)[0]
    beta_b = np.linalg.lstsq(Xb_aug, yb, rcond=None)[0]

    mean_Xa = Xa_aug.mean(axis=0)
    mean_Xb = Xb_aug.mean(axis=0)

    total_gap = float(np.mean(ya) - np.mean(yb))
    explained = float((mean_Xa - mean_Xb) @ beta_b)
    unexplained = total_gap - explained

    return DescriptiveResult(
        name="blinder_oaxaca",
        value=total_gap,
        extra={
            "total_gap": total_gap,
            "explained": explained,
            "unexplained": unexplained,
            "pct_explained": explained / total_gap if abs(total_gap) > 1e-12 else 0.0,
            "n_a": len(ya),
            "n_b": len(yb),
        },
    )


eqdsp = disparity_decompose


def cheatsheet() -> str:
    return "disparity_decompose({}) -> Decompose disparity (Blinder-Oaxaca)."
