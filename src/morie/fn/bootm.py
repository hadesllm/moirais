# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap for M-estimators.

Implements the nonparametric bootstrap for M-estimators, providing
valid inference under regularity conditions. Includes the multiplier
(weighted) bootstrap for computational efficiency.

References
----------
Arcones, M. A. & Gine, E. (1992). On the bootstrap of M-estimators
and other statistical functionals. In *Exploring the Limits of
Bootstrap* (pp. 13--47). Wiley.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapter 10.

van der Vaart, A. W. & Wellner, J. A. (1996). *Weak Convergence and
Empirical Processes*. Springer. Chapter 3.6.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


def bootm(
    data: np.ndarray,
    estimator: Callable[[np.ndarray], float] | None = None,
    *,
    n_boot: int = 999,
    method: str = "nonparametric",
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    r"""Bootstrap inference for an M-estimator.

    Given data :math:`\{X_1, \ldots, X_n\}` and an estimator
    :math:`T_n = T(X_1, \ldots, X_n)`, the bootstrap distribution of

    .. math::

        \sqrt{n}(T_n^* - T_n)

    consistently estimates the distribution of :math:`\sqrt{n}(T_n - \theta)`
    under standard M-estimator regularity conditions (Kosorok, 2008, Thm 10.6).

    For the multiplier bootstrap, weights :math:`\xi_i \sim \mathrm{Exp}(1)`
    are drawn and the weighted estimator is computed.

    Parameters
    ----------
    data : np.ndarray
        Data array, shape ``(n,)`` or ``(n, p)``.
    estimator : callable | None
        Function mapping data array to a scalar estimate.
        Defaults to ``np.mean`` applied to the first column.
    n_boot : int
        Number of bootstrap replicates.
    method : str
        ``"nonparametric"`` (resample rows) or ``"multiplier"``
        (exponential weights).
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper`` (percentile),
        ``ci_lower_normal``, ``ci_upper_normal``, ``boot_estimates``
        (array), ``n_boot``, ``method``.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)

    if estimator is None:
        if data.ndim == 1:
            estimator = np.mean
        else:
            estimator = lambda x: float(np.mean(x[:, 0]))

    point = float(estimator(data))
    n = data.shape[0]
    boot_est = np.empty(n_boot)

    if method == "multiplier":
        for b in range(n_boot):
            w = rng.exponential(1.0, size=n)
            w = w / w.sum() * n
            if data.ndim == 1:
                boot_est[b] = float(np.average(data, weights=w))
            else:
                idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
                boot_est[b] = float(estimator(data[idx]))
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_est[b] = float(estimator(data[idx]))

    se = float(np.std(boot_est, ddof=1))
    ci_pct = np.percentile(boot_est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "estimate": point,
        "se": se,
        "ci_lower": float(ci_pct[0]),
        "ci_upper": float(ci_pct[1]),
        "ci_lower_normal": point - z * se,
        "ci_upper_normal": point + z * se,
        "boot_estimates": boot_est,
        "n_boot": n_boot,
        "method": f"bootstrap_{method}",
    }


bootm_fn = bootm


def cheatsheet() -> str:
    return "bootm(data, estimator) -> Bootstrap for M-estimators (Kosorok 2008, Ch. 10)."
