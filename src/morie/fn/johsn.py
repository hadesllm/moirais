# morie.fn -- function file (hadesllm/morie)
"""Johansen cointegration test (trace + max-eigenvalue)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["johansen_cointegration"]


# Asymptotic critical values for the trace statistic, no det. trend
# (det_order = 0). Table from Osterwald-Lenum (1992), columns 90/95/99 %.
_TRACE_CRIT = {
    1: [2.7055, 3.8415, 6.6349],
    2: [13.4294, 15.4943, 19.9349],
    3: [27.0669, 29.7961, 35.4628],
    4: [44.4929, 47.8545, 54.6815],
    5: [65.8202, 69.8189, 77.8202],
    6: [91.1090, 95.7542, 104.9637],
    7: [120.3673, 125.6185, 136.0541],
    8: [153.6341, 159.5290, 171.0905],
    9: [190.8714, 197.3711, 210.0366],
    10: [232.1054, 239.2468, 253.2526],
}


def johansen_cointegration(x, k_ar_diff=1):
    r"""Johansen (1991) reduced-rank VAR test for cointegration.

    Reports the trace statistic together with critical values for
    each rank ``r`` from 0 up to ``k-1``.

    .. math::

        \mathrm{LR}_{trace}(r) = -T \sum_{i=r+1}^{k} \log(1 - \hat\lambda_i)

    Parameters
    ----------
    x : array-like, shape (T, k)
        Stationary panel of I(1) candidate series.
    k_ar_diff : int, default 1
        Number of lagged differences (Δy lags) in the VAR.

    Returns
    -------
    RichResult
        keys: ``eigenvalues``, ``trace_stat`` (length k), ``crit_values``
        (k × 3 matrix at 90/95/99%), ``rank`` (largest r where trace
        rejects at 5%), ``n``, ``k``, ``method``.

    References
    ----------
    Johansen S (1991). Estimation and Hypothesis Testing of Cointegration
    Vectors in Gaussian VAR Models. *Econometrica* 59(6), 1551-1580.
    """
    Y = np.atleast_2d(np.asarray(x, dtype=float))
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    T, k = Y.shape
    if T < 20 or k < 2:
        raise ValueError(f"Need T>=20 and k>=2; got T={T}, k={k}.")

    # Prefer statsmodels if available -- robust treatment of trends/lags.
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        jres = coint_johansen(Y, det_order=0, k_ar_diff=k_ar_diff)
        eig = np.asarray(jres.eig)
        trace = np.asarray(jres.lr1)
        cvt = np.asarray(jres.cvt)
        rank = int(np.sum(trace > cvt[:, 1]))
        return RichResult(payload={
            "eigenvalues": eig,
            "trace_stat": trace,
            "crit_values": cvt,
            "rank": rank,
            "n": int(T), "k": int(k),
            "method": "Johansen trace test via statsmodels (det_order=0)",
        })
    except Exception:
        pass

    # ---- Pure-NumPy reduced-rank regression fallback ----------------------
    dY = np.diff(Y, axis=0)
    # Build lagged-diff matrix Z2 and levels regressor Z1.
    Z0 = dY[k_ar_diff:]
    rows = Z0.shape[0]
    Z1 = Y[k_ar_diff: k_ar_diff + rows]
    if k_ar_diff > 0:
        Z2 = np.column_stack([dY[k_ar_diff - i - 1 : k_ar_diff - i - 1 + rows]
                              for i in range(k_ar_diff)])
        # Regress Z0 and Z1 on Z2 to get residuals R0 and R1.
        Z2c = np.column_stack([np.ones(rows), Z2])
    else:
        Z2c = np.ones((rows, 1))
    Pi = Z2c @ np.linalg.pinv(Z2c.T @ Z2c) @ Z2c.T
    R0 = Z0 - Pi @ Z0
    R1 = Z1 - Pi @ Z1
    S00 = (R0.T @ R0) / rows
    S01 = (R0.T @ R1) / rows
    S11 = (R1.T @ R1) / rows
    S10 = S01.T
    # Solve generalised eigenproblem |λ S11 − S10 S00⁻¹ S01| = 0.
    M = np.linalg.solve(S11, S10 @ np.linalg.solve(S00, S01))
    eig = np.linalg.eigvals(M).real
    eig = np.clip(np.sort(eig)[::-1], 0.0, 1.0 - 1e-12)
    trace = np.array([-rows * np.sum(np.log(1.0 - eig[r:])) for r in range(k)])
    cvt = np.array([_TRACE_CRIT.get(k - r, [np.nan, np.nan, np.nan])
                    for r in range(k)])
    rank = int(np.sum(trace > cvt[:, 1]))
    return RichResult(payload={
        "eigenvalues": eig,
        "trace_stat": trace,
        "crit_values": cvt,
        "rank": rank,
        "n": int(T), "k": int(k),
        "method": "Johansen trace test (reduced-rank regression, numpy)",
    })


def cheatsheet():
    return "johsn: Johansen cointegration test (Johansen 1991)."
