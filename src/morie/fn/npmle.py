# morie.fn — function file (hadesllm/morie)
"""Nonparametric Maximum Likelihood Estimator (NPMLE).

Implements the NPMLE for distribution functions and mixing
distributions using the EM algorithm (Turnbull, 1976) and the
self-consistency equation.

References
----------
Turnbull, B. W. (1976). The empirical distribution function with
arbitrarily grouped, censored and truncated data. *Journal of the
Royal Statistical Society, Series B*, 38(3), 290--295.

Kiefer, J. & Wolfowitz, J. (1956). Consistency of the maximum
likelihood estimator in the presence of infinitely many incidental
parameters. *Annals of Mathematical Statistics*, 27(4), 887--906.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 9, 13.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def npmle(data: np.ndarray, cdf=None, *, left: np.ndarray | None = None, right: np.ndarray | None = None, max_iter: int = 500, tol: float = 1e-8) -> dict[str, Any]:
    r"""Compute the NPMLE for a distribution function.

    For uncensored data, this reduces to the empirical CDF.
    For interval-censored data :math:`(L_i, R_i]`, computes the Turnbull
    NPMLE via EM:

    .. math::

        \hat{F}^{(t+1)}(s_j) = \frac{1}{n} \sum_{i=1}^{n}
            \frac{\hat{p}_j^{(t)} \cdot \mathbf{1}(s_j \in (L_i, R_i])}
                 {\sum_{k} \hat{p}_k^{(t)} \cdot \mathbf{1}(s_k \in (L_i, R_i])}

    Parameters
    ----------
    data : np.ndarray
        Observed data values, shape ``(n,)``.  For uncensored data, pass
        the observed values.  For interval-censored data, pass midpoints
        or left endpoints (and supply *left*/*right*).
    left : np.ndarray | None
        Left endpoints of intervals (for interval censoring).
    right : np.ndarray | None
        Right endpoints of intervals (for interval censoring).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on weight change.

    Returns
    -------
    dict[str, Any]
        ``support`` (array of mass points), ``weights`` (probability masses),
        ``cdf_values``, ``log_likelihood``, ``n_iter``, ``converged``.
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)

    if left is not None and right is not None:
        return _npmle_interval(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            max_iter=max_iter,
            tol=tol,
        )

    support = np.sort(np.unique(data))
    m = len(support)
    weights = np.full(m, 1.0 / m)

    for j, s in enumerate(support):
        weights[j] = float(np.sum(data == s)) / n

    cdf = np.cumsum(weights)

    ll = float(np.sum(np.log(np.maximum(weights[np.searchsorted(support, data, side="right") - 1], 1e-300))))

    return {
        "support": support,
        "weights": weights,
        "cdf_values": cdf,
        "log_likelihood": ll,
        "n_iter": 1,
        "converged": True,
    }


def _npmle_interval(left, right, *, max_iter, tol):
    """Turnbull EM for interval-censored data."""
    n = len(left)
    endpoints = np.sort(np.unique(np.concatenate([left, right])))
    endpoints = endpoints[np.isfinite(endpoints)]
    m = len(endpoints)
    weights = np.full(m, 1.0 / m)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        membership = np.zeros((n, m))
        for j, s in enumerate(endpoints):
            membership[:, j] = ((left < s) | np.isclose(left, s, atol=1e-12)) & (s <= right)

        denom = membership @ weights
        denom = np.maximum(denom, 1e-300)

        new_weights = np.zeros(m)
        for j in range(m):
            new_weights[j] = np.sum(membership[:, j] * weights[j] / denom) / n

        new_weights = np.maximum(new_weights, 0.0)
        total = new_weights.sum()
        if total > 0:
            new_weights /= total

        change = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        n_iter = iteration + 1

        if change < tol:
            converged = True
            break

    cdf = np.cumsum(weights)
    ll = float(np.sum(np.log(np.maximum(membership @ weights, 1e-300))))

    return {
        "support": endpoints,
        "weights": weights,
        "cdf_values": cdf,
        "log_likelihood": ll,
        "n_iter": n_iter,
        "converged": converged,
    }


npmle_fn = npmle


def cheatsheet() -> str:
    return "npmle(data) -> Nonparametric MLE via Turnbull EM (Kosorok 2008, Ch. 9, 13)."
