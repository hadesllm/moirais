# morie.fn — function file (hadesllm/morie)
"""Log-linear model for multi-way tables. 'Truly wonderful, the mind of a child is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def loglinear_model(table: np.ndarray, max_iter: int = 100, tol: float = 1e-8, cdf=None) -> DescriptiveResult:
    r"""
    Fit a log-linear independence model to a two-way contingency table
    using iterative proportional fitting (IPF).

    Under independence:
    :math:`\\log m_{ij} = \\mu + \\alpha_i + \\beta_j`.

    :param table: (r, c) contingency table of non-negative counts.
    :type table: numpy.ndarray
    :param max_iter: Maximum IPF iterations. Default 100.
    :type max_iter: int
    :param tol: Convergence tolerance. Default 1e-8.
    :type tol: float
    :return: DescriptiveResult with fitted values, G2 statistic, p-value.
    :rtype: DescriptiveResult

    References
    ----------
    Bishop Y.M.M., Fienberg S.E. & Holland P.W. (1975). *Discrete
    Multivariate Analysis: Theory and Practice*. MIT Press.
    """
    T = np.asarray(table, dtype=float)
    if T.ndim != 2:
        raise ValueError(f"Expected 2-D table, got {T.ndim}-D.")
    if np.any(T < 0):
        raise ValueError("Table must contain non-negative values.")
    r, c = T.shape
    row_totals = T.sum(axis=1)
    col_totals = T.sum(axis=0)
    grand = T.sum()
    fitted = np.ones_like(T)
    for _ in range(max_iter):
        fitted *= row_totals[:, None] / np.maximum(fitted.sum(axis=1, keepdims=True), 1e-12)
        fitted *= col_totals[None, :] / np.maximum(fitted.sum(axis=0, keepdims=True), 1e-12)
        if np.max(np.abs(fitted.sum(axis=1) - row_totals)) < tol:
            break
    mask = (T > 0) & (fitted > 0)
    g2 = 2.0 * float(np.sum(T[mask] * np.log(T[mask] / fitted[mask])))
    dof = (r - 1) * (c - 1)
    from scipy import stats as _st

    p_value = float(1.0 - _st.chi2.cdf(g2, df=dof)) if dof > 0 else 1.0
    return DescriptiveResult(
        name="loglinear_model",
        value=g2,
        extra={
            "fitted": fitted,
            "G2": g2,
            "dof": dof,
            "p_value": p_value,
        },
    )


logln = loglinear_model


def cheatsheet() -> str:
    return "loglinear_model({}) -> Log-linear model for multi-way tables. 'Truly wonderful, the"
