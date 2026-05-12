# morie.fn -- function file (hadesllm/morie)
"""Partial correlation. 'In my experience there is no such thing as luck. --'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def partial_correlation(
    X: np.ndarray,
    i: int = 0,
    j: int = 1,
    controls: list[int] | None = None,
) -> DescriptiveResult:
    """Partial correlation between variables i and j controlling for others.

    Computes the Pearson correlation between the residuals of
    regressing X[:, i] and X[:, j] on the control variables.

    :param X: (n, p) data matrix.
    :param i: Column index of first variable.
    :param j: Column index of second variable.
    :param controls: Column indices of control variables. If None, all other columns.
    :return: DescriptiveResult with partial correlation.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if controls is None:
        controls = [k for k in range(p) if k != i and k != j]
    if len(controls) == 0:
        r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
        return DescriptiveResult(name="pcorp", value=r, extra={"n": n, "controls": []})

    Z = X[:, controls]
    Z1 = np.column_stack([np.ones(n), Z])

    def resid(y):
        beta = np.linalg.lstsq(Z1, y, rcond=None)[0]
        return y - Z1 @ beta

    ri = resid(X[:, i])
    rj = resid(X[:, j])
    denom = np.sqrt(np.sum(ri**2) * np.sum(rj**2)) + 1e-12
    r_partial = float(np.sum(ri * rj) / denom)

    return DescriptiveResult(
        name="pcorp",
        value=r_partial,
        extra={"n": n, "i": i, "j": j, "controls": controls},
    )


pcorp = partial_correlation


def cheatsheet() -> str:
    return "partial_correlation({}) -> Partial correlation. 'In my experience there is no such thin"
