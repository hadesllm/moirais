# morie.fn -- function file (hadesllm/morie)
"""Parallel forms reliability."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def parallel_form_reliability(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> ESRes:
    """Parallel forms reliability via Pearson correlation.

    Parameters
    ----------
    scores_a : ndarray
        Scores on form A (n,).
    scores_b : ndarray
        Scores on form B (n,).

    Returns
    -------
    ESRes
        measure="parallel_forms".

    References
    ----------
    Crocker, L. & Algina, J. (2006). Introduction to Classical and
    Modern Test Theory. Cengage Learning.
    """
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    n = len(a)
    if n != len(b):
        raise ValueError("Form A and B must have same length.")

    r = float(np.corrcoef(a, b)[0, 1])
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z = np.arctanh(r)
    ci_lo = float(np.tanh(z - 1.96 * se))
    ci_hi = float(np.tanh(z + 1.96 * se))

    return ESRes(
        measure="parallel_forms",
        estimate=r,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n=n,
        extra={"mean_a": float(a.mean()), "mean_b": float(b.mean())},
    )


parallel_forms = parallel_form_reliability


def cheatsheet() -> str:
    return "parallel_form_reliability({}) -> Parallel forms reliability."
