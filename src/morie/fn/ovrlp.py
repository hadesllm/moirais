# morie.fn — function file (hadesllm/morie)
"""Overlap (matching) weights for causal inference.

Overlap weights are the propensity-score weights that maximize the
effective sample size for the overlap population — the target
population in which all subjects have positive probability of
receiving either treatment.

Also exposes the Szymkiewicz-Simpson overlap coefficient for sets.

References
----------
Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing
covariates via propensity score weighting.
*Journal of the American Statistical Association*, 113(521), 390-400.

Zhou, Y., Matsouaka, R. A., & Thomas, L. E. (2020). Propensity score
weighting under limited overlap and model misspecification.
*Statistical Methods in Medical Research*, 29(12), 3721-3756.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

__all__ = ["ovrlp"]


def ovrlp(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    ps_model: str = "logistic",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate the ATE for the overlap population via overlap weights.

    Overlap weights are:

    .. math::

        w_i = \begin{cases}
            1 - \hat{e}(X_i) & \text{if } T_i = 1 \\
            \hat{e}(X_i)     & \text{if } T_i = 0
        \end{cases}

    These down-weight units with extreme propensity scores and target
    the population where both treatment arms are well-represented.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    ps_model : str
        Propensity score model (``"logistic"``).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``weights``, ``propensity``, ``effective_n``,
        ``n``, ``method``.

    References
    ----------
    Li et al. (2018). JASA, 113(521), 390-400.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if not (len(T) == n == X.shape[0]):
        raise ValueError("Y, T, X must share first dimension.")

    Xa = np.column_stack([np.ones(n), X])
    ps = np.clip(_irls(Xa, T), 1e-6, 1.0 - 1e-6)

    # Overlap weights: w = e(x)(1-e(x)); treated get (1-e), controls get e
    w = np.where(T == 1, 1.0 - ps, ps)

    # Hajek-style weighted ATE estimator
    w1 = w * T
    w0 = w * (1.0 - T)
    mu1_hat = np.sum(w1 * Y) / np.sum(w1)
    mu0_hat = np.sum(w0 * Y) / np.sum(w0)
    ate = float(mu1_hat - mu0_hat)

    # Influence-function based SE
    ic1 = w1 * (Y - mu1_hat) / np.sum(w1)
    ic0 = w0 * (Y - mu0_hat) / np.sum(w0)
    ic = ic1 - ic0
    se = float(np.sqrt(n * np.var(ic, ddof=1)))

    eff_n = float(np.sum(w) ** 2 / np.sum(w**2))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "weights": w,
        "propensity": ps,
        "effective_n": eff_n,
        "n": n,
        "method": "overlap-weights",
    }


def overlap_coefficient(set_a, set_b) -> float:
    """Szymkiewicz-Simpson overlap coefficient (set similarity).

    .. math::

        \\text{overlap}(A, B) = |A \\cap B| / \\min(|A|, |B|)
    """
    if isinstance(set_a, np.ndarray):
        set_a = set(np.where(set_a.ravel().astype(bool))[0])
    else:
        set_a = set(set_a)
    if isinstance(set_b, np.ndarray):
        set_b = set(np.where(set_b.ravel().astype(bool))[0])
    else:
        set_b = set(set_b)
    m = min(len(set_a), len(set_b))
    return 0.0 if m == 0 else len(set_a & set_b) / m


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
    return "ovrlp(Y, T, X) -> Overlap weights ATE (Li et al. 2018, JASA)."
