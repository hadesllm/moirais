# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Breusch-Pagan Lagrange multiplier test."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def breusch_pagan_lm(
    y: np.ndarray,
    X: np.ndarray,
    entity: np.ndarray,
) -> DescriptiveResult:
    r"""Breusch-Pagan LM test for individual (random) effects.

    Tests :math:`H_0: \\sigma^2_u = 0` (pooled OLS adequate) against
    :math:`H_1: \\sigma^2_u > 0` (RE model needed).

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    entity : (n,) entity identifiers

    Returns
    -------
    DescriptiveResult
        ``value`` is the LM statistic (chi2 with 1 df).

    References
    ----------
    Breusch, T. S. & Pagan, A. R. (1980). The Lagrange multiplier test
    and its applications to model specification in econometrics. *RES*,
    47(1), 239--253.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    entity = np.asarray(entity).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)

    X_int = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    e = y - X_int @ beta

    unique_ent = np.unique(entity)
    N = len(unique_ent)

    sum_e_sq = float(e @ e)
    sum_ebar_sq = 0.0
    for ent in unique_ent:
        idx = entity == ent
        Ti = np.sum(idx)
        sum_ebar_sq += (np.sum(e[idx])) ** 2

    LM = (n / (2.0 * (N - 1))) * (sum_ebar_sq / sum_e_sq - 1.0) ** 2
    p_value = float(_st.chi2.sf(LM, df=1))

    return DescriptiveResult(
        name="Breusch-Pagan LM Test",
        value=float(LM),
        extra={
            "p_value": p_value,
            "df": 1,
            "n": n,
            "n_entities": N,
            "reject_H0": p_value < 0.05,
            "interpretation": "RE effects present" if p_value < 0.05 else "Pooled OLS adequate",
        },
    )


bplag = breusch_pagan_lm


def cheatsheet() -> str:
    return "breusch_pagan_lm({}) -> BP LM test for random effects."
