# moirais.fn — function file (hadesllm/moirais)
"""Uniform DIF detection via logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_uniform(responses: np.ndarray | pd.DataFrame, total_score: np.ndarray, group: np.ndarray | list, cdf=None, *, item_names: list[str] | None = None, alpha: float = 0.05) -> DIFResult:
    """Detect uniform DIF via logistic regression (group main effect).

    Fits logit(P) = b0 + b1*total + b2*group per item.
    Uniform DIF indicated by significant b2.

    Parameters
    ----------
    responses : ndarray or DataFrame
        Binary response matrix (n x k).
    total_score : ndarray
        Matching variable (total score, length n).
    group : array-like
        Group variable (0/1).
    item_names : list[str], optional
    alpha : float

    Returns
    -------
    DIFResult
        method="Uniform".

    References
    ----------
    Swaminathan, H. & Rogers, H. J. (1990). Detecting differential
    item functioning using logistic regression procedures. Journal
    of Educational Measurement, 27(4), 361-370.
    """
    X = np.asarray(responses, dtype=np.float64)
    ts = np.asarray(total_score, dtype=np.float64)
    g = np.asarray(group, dtype=np.float64).ravel()
    n, k = X.shape
    X = np.where(np.isnan(X), 0, X)

    if item_names is None:
        item_names = list(responses.columns) if isinstance(responses, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    design = np.column_stack([np.ones(n), ts, g])
    rows = []
    flagged = []
    for j in range(k):
        y = X[:, j]
        beta = np.zeros(3)
        for _ in range(30):
            logit = design @ beta
            P = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
            P = np.clip(P, 1e-10, 1 - 1e-10)
            W = P * (1 - P)
            grad = design.T @ (y - P)
            H = design.T @ (design * W[:, None])
            try:
                delta = np.linalg.solve(H + np.eye(3) * 1e-8, grad)
            except np.linalg.LinAlgError:
                break
            beta += delta
            if np.max(np.abs(delta)) < 1e-6:
                break

        P_f = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -500, 500)))
        P_f = np.clip(P_f, 1e-10, 1 - 1e-10)
        W_f = P_f * (1 - P_f)
        H_f = design.T @ (design * W_f[:, None])
        try:
            cov = np.linalg.inv(H_f + np.eye(3) * 1e-8)
            se_b2 = np.sqrt(max(cov[2, 2], 1e-10))
        except np.linalg.LinAlgError:
            se_b2 = 1.0
        z = beta[2] / se_b2
        p_val = 2 * (1 - sp.norm.cdf(abs(z)))

        rows.append(
            {
                "item": item_names[j],
                "group_coef": float(beta[2]),
                "se": float(se_b2),
                "z": float(z),
                "p_value": float(p_val),
            }
        )
        if p_val < alpha:
            flagged.append(item_names[j])

    return DIFResult(method="Uniform", items=pd.DataFrame(rows), flagged=flagged)


uniform_dif = dif_uniform


def cheatsheet() -> str:
    return "dif_uniform({}) -> Uniform DIF detection via logistic regression."
