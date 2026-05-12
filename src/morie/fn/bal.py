# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Balance diagnostics via standardized mean differences."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def balance_diagnostics(
    covariates: Union[list, np.ndarray],
    treatment: Union[list, np.ndarray],
    *,
    covariate_names: list[str] | None = None,
    weights: Union[list, np.ndarray, None] = None,
) -> list[dict[str, Any]]:
    r"""
    Compute standardised mean differences (SMD) for covariate balance.

    For each covariate, the SMD is:

    .. math::

        \\text{SMD} = \\frac{\\bar{X}_1 - \\bar{X}_0}
            {\\sqrt{(s_1^2 + s_0^2) / 2}}

    Conventionally, |SMD| < 0.1 indicates adequate balance (Austin, 2011).

    If weights are provided, weighted means and variances are used for the
    "adjusted" SMD.

    :param covariates: Matrix of shape (n, p).
    :param treatment: Binary treatment indicator (0/1).
    :param covariate_names: Optional names for each covariate column.
    :param weights: Optional weights (e.g., IPW) for adjusted balance.
    :return: List of dicts, one per covariate, each with keys: variable,
        smd_raw, smd_weighted (None if no weights), balanced_raw,
        balanced_weighted.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Austin, P. C. (2011). An introduction to propensity score methods for
    reducing the effects of confounding in observational studies.
    *Multivariate Behavioral Research*, 46(3), 399--424.
    """
    X = np.asarray(covariates, dtype=float)
    T = np.asarray(treatment, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if len(T) != n:
        raise ValueError("covariates and treatment must have same number of rows.")

    treated = T == 1
    control = T == 0
    names = covariate_names or [f"X{j}" for j in range(p)]

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != n:
            raise ValueError("weights must have same length as treatment.")

    results = []
    for j in range(p):
        xj = X[:, j]
        m1 = float(np.mean(xj[treated]))
        m0 = float(np.mean(xj[control]))
        s1 = float(np.var(xj[treated], ddof=1))
        s0 = float(np.var(xj[control], ddof=1))
        pooled_sd = np.sqrt((s1 + s0) / 2.0)
        smd_raw = (m1 - m0) / pooled_sd if pooled_sd > 0 else float("nan")

        smd_w = None
        bal_w = None
        if w is not None:
            # Weighted means
            wt = w * T
            wc = w * (1 - T)
            sum_wt = np.sum(wt)
            sum_wc = np.sum(wc)
            if sum_wt > 0 and sum_wc > 0:
                wm1 = float(np.sum(wt * xj) / sum_wt)
                wm0 = float(np.sum(wc * xj) / sum_wc)
                smd_w = (wm1 - wm0) / pooled_sd if pooled_sd > 0 else float("nan")
                bal_w = abs(smd_w) < 0.1 if np.isfinite(smd_w) else None

        results.append(
            {
                "variable": names[j] if j < len(names) else f"X{j}",
                "smd_raw": float(smd_raw),
                "smd_weighted": float(smd_w) if smd_w is not None else None,
                "balanced_raw": abs(smd_raw) < 0.1 if np.isfinite(smd_raw) else None,
                "balanced_weighted": bal_w,
            }
        )

    return results


bal = balance_diagnostics


def cheatsheet() -> str:
    return "balance_diagnostics({}) -> Balance diagnostics via standardized mean differences."
