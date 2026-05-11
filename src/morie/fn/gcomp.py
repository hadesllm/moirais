# morie.fn — function file (hadesllm/morie)
"""G-computation for time-varying treatments.

Extends the parametric g-formula to the time-varying setting by
sequentially fitting outcome models at each time point and computing
the counterfactual outcome under a specified treatment regime.

References
----------
Robins, J. M. (1986). A new approach to causal inference in
mortality studies with a sustained exposure period.
*Mathematical Modelling*, 7(9-12), 1393-1512.

Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*.
Chapman & Hall/CRC. Part III.

Daniel, R. M., Cousens, S. N., De Stavola, B. L., Kenward, M. G.,
& Sterne, J. A. C. (2013). Methods for dealing with time-dependent
confounding. *Statistics in Medicine*, 32(9), 1584-1618.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm as _norm

__all__ = ["gcomp"]


def gcomp(
    Y: np.ndarray,
    T_seq: np.ndarray,
    L_seq: np.ndarray,
    *,
    regime: str = "always_treat",
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    r"""G-computation for time-varying treatment regimes.

    Sequentially models the outcome under the observed data, then
    applies the g-formula by replacing treatment with a counterfactual
    regime and iterating forward through time.

    For ``K`` time points, at each step :math:`k`:

    .. math::

        \bar{L}_k = (L_0, \ldots, L_k), \quad
        \hat{Y}_k = \hat{\mu}_k(\bar{A}_k^*, \bar{L}_k)

    Parameters
    ----------
    Y : np.ndarray
        Final outcome, shape ``(n,)``.
    T_seq : np.ndarray
        Treatment matrix, shape ``(n, K)``; one column per time point.
    L_seq : np.ndarray
        Time-varying covariate matrix, shape ``(n, K, p)`` or
        ``(n, K)`` for a single covariate per time point.
    regime : str
        ``"always_treat"`` (:math:`\bar{A}^*=1`),
        ``"never_treat"`` (:math:`\bar{A}^*=0`), or
        ``"natural"`` (observed treatment).
    alpha : float
        Significance level.
    n_boot : int
        Bootstrap replicates for standard error.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``ate`` (difference always vs never), ``mean_treated``,
        ``mean_control``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.

    References
    ----------
    Robins (1986). Mathematical Modelling, 7(9-12), 1393-1512.
    Daniel et al. (2013). Statistics in Medicine, 32(9), 1584-1618.
    """
    Y = np.asarray(Y, dtype=float)
    T_seq = np.asarray(T_seq, dtype=float)
    if T_seq.ndim == 1:
        T_seq = T_seq[:, None]
    L_seq = np.asarray(L_seq, dtype=float)
    if L_seq.ndim == 2:
        L_seq = L_seq[:, :, None]

    n, K = T_seq.shape
    if len(Y) != n:
        raise ValueError("Y and T_seq must have the same number of rows.")
    if L_seq.shape[0] != n or L_seq.shape[1] != K:
        raise ValueError("L_seq shape must be (n, K) or (n, K, p).")

    rng = np.random.default_rng(seed)

    def _estimate(idx):
        y_ = Y[idx]
        t_ = T_seq[idx]
        l_ = L_seq[idx]
        n_ = len(idx)

        # Build cumulative feature matrix and fit OLS at each time step
        betas = []
        for k in range(K):
            cum_T = t_[:, : k + 1].reshape(n_, -1)
            cum_L = l_[:, : k + 1, :].reshape(n_, -1)
            Xk = np.column_stack([np.ones(n_), cum_T, cum_L])
            bk = np.linalg.lstsq(Xk, y_, rcond=None)[0]
            betas.append(bk)

        def _predict_regime(t_regime):
            t_fill = np.full((n_, K), t_regime, dtype=float)
            preds = np.zeros(n_)
            for k in range(K):
                cum_T = t_fill[:, : k + 1]
                cum_L = l_[:, : k + 1, :].reshape(n_, -1)
                Xk = np.column_stack([np.ones(n_), cum_T, cum_L])
                preds = Xk @ betas[k]
            return preds

        if regime == "natural":
            return float(np.mean(y_)), None, None

        mu1 = float(np.mean(_predict_regime(1.0)))
        mu0 = float(np.mean(_predict_regime(0.0)))
        return mu1 - mu0, mu1, mu0

    ate_obs, mu1_obs, mu0_obs = _estimate(np.arange(n))

    # Bootstrap SE
    boot_ates = np.empty(n_boot)
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        boot_ates[b] = _estimate(boot_idx)[0]
    se = float(np.std(boot_ates, ddof=1))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": float(ate_obs),
        "mean_treated": float(mu1_obs) if mu1_obs is not None else None,
        "mean_control": float(mu0_obs) if mu0_obs is not None else None,
        "se": se,
        "ci_lower": float(ate_obs) - z * se,
        "ci_upper": float(ate_obs) + z * se,
        "n": n,
        "method": "g-computation",
    }


def cheatsheet() -> str:
    return "gcomp(Y, T_seq, L_seq) -> G-computation for time-varying treatments (Robins 1986)."
