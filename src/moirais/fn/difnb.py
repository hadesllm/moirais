# moirais.fn — function file (hadesllm/moirais)
"""Non-uniform DIF detection via interaction term."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_nonuniform(responses: np.ndarray | pd.DataFrame, ability: np.ndarray, group: np.ndarray | list, cdf=None, *, item_names: list[str] | None = None, alpha: float = 0.05) -> DIFResult:
    """Detect non-uniform DIF via ability x group interaction.

    Fits logistic: logit(P) = b0 + b1*ability + b2*group + b3*ability*group.
    Non-uniform DIF is indicated by a significant b3 (interaction).

    Parameters
    ----------
    responses : ndarray or DataFrame
        Binary response matrix (n x k).
    ability : ndarray
        Total score or theta estimate (length n).
    group : array-like
        Group membership (0/1).
    item_names : list[str], optional
    alpha : float
        Significance level.

    Returns
    -------
    DIFResult
        method="NonUniform".

    References
    ----------
    Swaminathan, H. & Rogers, H. J. (1990). Detecting differential
    item functioning using logistic regression procedures. Journal
    of Educational Measurement, 27(4), 361-370.
    """
    X = np.asarray(responses, dtype=np.float64)
    ability = np.asarray(ability, dtype=np.float64)
    g = np.asarray(group, dtype=np.float64).ravel()
    n, k = X.shape
    X = np.where(np.isnan(X), 0, X)

    if item_names is None:
        item_names = list(responses.columns) if isinstance(responses, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    design = np.column_stack([np.ones(n), ability, g, ability * g])

    rows = []
    flagged = []
    for j in range(k):
        y = X[:, j]
        beta = np.zeros(4)
        for _ in range(30):
            logit = design @ beta
            P = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
            P = np.clip(P, 1e-10, 1 - 1e-10)
            W = P * (1 - P)
            gradient = design.T @ (y - P)
            H = design.T @ (design * W[:, None])
            try:
                delta = np.linalg.solve(H + np.eye(4) * 1e-8, gradient)
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
            cov = np.linalg.inv(H_f + np.eye(4) * 1e-8)
            se_b3 = np.sqrt(max(cov[3, 3], 1e-10))
        except np.linalg.LinAlgError:
            se_b3 = 1.0
        z = beta[3] / se_b3
        p_val = 2 * (1 - sp.norm.cdf(abs(z)))

        rows.append(
            {
                "item": item_names[j],
                "interaction_coef": float(beta[3]),
                "se": float(se_b3),
                "z": float(z),
                "p_value": float(p_val),
            }
        )
        if p_val < alpha:
            flagged.append(item_names[j])

    return DIFResult(method="NonUniform", items=pd.DataFrame(rows), flagged=flagged)


nonuniform_dif = dif_nonuniform


def cheatsheet() -> str:
    return "dif_nonuniform({}) -> Non-uniform DIF detection via interaction term."
