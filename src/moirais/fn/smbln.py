"""The man who moves a mountain begins by carrying away small stones. — Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def smd_balance(
    X: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> DescriptiveResult:
    """
    Compute standardized mean differences (SMD) for covariate balance.

    .. math::

        SMD_j = \\frac{\\bar{X}_{j,1} - \\bar{X}_{j,0}}
                     {\\sqrt{(s_{j,1}^2 + s_{j,0}^2) / 2}}

    :param X: Covariate matrix (n, p).
    :param treatment: Binary treatment indicator (n,), 0/1.
    :param weights: Optional weights (n,). If provided, computes weighted SMD.
    :return: DescriptiveResult with max absolute SMD as value.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the distribution
    of baseline covariates between treatment groups in propensity-score
    matched samples. Statistics in Medicine, 28(25), 3083--3107.
    doi:10.1002/sim.3697
    """
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != treatment.shape[0]:
        raise ValueError("X and treatment must have the same number of rows.")

    t1 = treatment == 1
    t0 = treatment == 0

    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != X.shape[0]:
            raise ValueError("weights must have the same length as treatment.")
    else:
        w = np.ones(X.shape[0])

    p = X.shape[1]
    smds = np.zeros(p)
    for j in range(p):
        xj = X[:, j]
        w1 = w[t1]
        w0 = w[t0]
        sw1 = np.sum(w1)
        sw0 = np.sum(w0)

        mean1 = np.sum(w1 * xj[t1]) / sw1 if sw1 > 0 else 0.0
        mean0 = np.sum(w0 * xj[t0]) / sw0 if sw0 > 0 else 0.0

        var1 = np.sum(w1 * (xj[t1] - mean1) ** 2) / sw1 if sw1 > 0 else 0.0
        var0 = np.sum(w0 * (xj[t0] - mean0) ** 2) / sw0 if sw0 > 0 else 0.0

        pooled_sd = np.sqrt((var1 + var0) / 2.0)
        smds[j] = (mean1 - mean0) / pooled_sd if pooled_sd > 1e-12 else 0.0

    max_abs_smd = float(np.max(np.abs(smds)))
    balanced = bool(max_abs_smd < 0.1)

    return DescriptiveResult(
        name="SMD Balance",
        value=float(np.round(max_abs_smd, 4)),
        extra={
            "smds": smds.tolist(),
            "max_abs_smd": float(np.round(max_abs_smd, 4)),
            "mean_abs_smd": float(np.round(np.mean(np.abs(smds)), 4)),
            "balanced": balanced,
            "threshold": 0.1,
            "n_covariates": p,
            "n_treated": int(np.sum(t1)),
            "n_control": int(np.sum(t0)),
            "weighted": weights is not None,
        },
    )


smbln = smd_balance


def cheatsheet() -> str:
    return "smd_balance({}) -> Standardized mean difference balance check. 'Great shot, kid"
