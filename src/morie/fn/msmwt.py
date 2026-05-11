# morie.fn — function file (hadesllm/morie)
"""MSM stabilized inverse-probability weights.

Computes and diagnostics stabilized IPW weights for marginal
structural models with time-varying treatments.

References
----------
Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal
structural models and causal inference in epidemiology.
*Epidemiology*, 11(5), 550-560.

Cole, S. R., & Hernan, M. A. (2008). Constructing inverse probability
weights for marginal structural models.
*American Journal of Epidemiology*, 168(6), 656-664.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

__all__ = ["msmwt"]


def msmwt(
    T_seq: np.ndarray,
    L_seq: np.ndarray,
    *,
    ps_trim: float = 0.01,
    stabilize: bool = True,
) -> dict[str, Any]:
    r"""Compute stabilized IPW weights for time-varying treatment.

    For :math:`K` time points, stabilized weights are:

    .. math::

        SW_i = \prod_{k=1}^{K}
            \frac{P(A_k = a_{ik} \mid \bar{A}_{k-1})}
                 {P(A_k = a_{ik} \mid \bar{A}_{k-1}, \bar{L}_k)}

    Parameters
    ----------
    T_seq : np.ndarray
        Treatment matrix, shape ``(n, K)``.
    L_seq : np.ndarray
        Time-varying covariate matrix, shape ``(n, K)`` or
        ``(n, K, p)`` (multiple covariates per time).
    ps_trim : float
        Clip propensity scores to ``[ps_trim, 1 - ps_trim]``.
    stabilize : bool
        If True, divide by marginal treatment probabilities.

    Returns
    -------
    dict
        ``weights`` (stabilized SW, shape ``(n,)``),
        ``weight_numerator``, ``weight_denominator``,
        ``effective_n``, ``max_weight``, ``mean_weight``,
        ``n``, ``K``, ``method``.

    References
    ----------
    Cole & Hernan (2008). AJE, 168(6), 656-664.
    """
    T_seq = np.asarray(T_seq, dtype=float)
    if T_seq.ndim == 1:
        T_seq = T_seq[:, None]
    L_seq = np.asarray(L_seq, dtype=float)
    if L_seq.ndim == 2:
        L_seq = L_seq[:, :, None]
    n, K = T_seq.shape
    if L_seq.shape[0] != n or L_seq.shape[1] != K:
        raise ValueError("L_seq shape must be (n, K) or (n, K, p).")

    log_num = np.zeros(n)
    log_den = np.zeros(n)

    for k in range(K):
        t_k = T_seq[:, k]
        # Denominator: P(T_k | T_{<k}, L_{<=k})
        cum_T = T_seq[:, :k].reshape(n, -1) if k > 0 else np.zeros((n, 0))
        cum_L = L_seq[:, : k + 1, :].reshape(n, -1)
        Xd = np.column_stack([np.ones(n), cum_T, cum_L])
        ps_d = np.clip(_irls(Xd, t_k), ps_trim, 1.0 - ps_trim)

        # Numerator: P(T_k | T_{<k})  [marginal: only past T]
        if stabilize:
            Xn = np.column_stack([np.ones(n), cum_T]) if k > 0 else np.ones((n, 1))
            ps_n = np.clip(_irls(Xn, t_k), ps_trim, 1.0 - ps_trim)
        else:
            ps_n = np.ones(n) * t_k.mean() if t_k.mean() > 0 else np.full(n, 0.5)
            ps_n = np.clip(ps_n, ps_trim, 1.0 - ps_trim)

        p_den = np.where(t_k == 1, ps_d, 1.0 - ps_d)
        p_num = np.where(t_k == 1, ps_n, 1.0 - ps_n)
        log_den += np.log(p_den)
        log_num += np.log(p_num)

    weights = np.exp(log_num - log_den)
    eff_n = float(np.sum(weights) ** 2 / np.sum(weights**2))

    return {
        "weights": weights,
        "weight_numerator": np.exp(log_num),
        "weight_denominator": np.exp(log_den),
        "effective_n": eff_n,
        "max_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "n": n,
        "K": K,
        "method": "MSM-stabilized-IPW" if stabilize else "MSM-raw-IPW",
    }


def _irls(X, y, max_iter=25):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        A = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        b = (X.T * W) @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "msmwt(T_seq, L_seq) -> MSM stabilized weights (Cole & Hernan 2008, AJE)."
