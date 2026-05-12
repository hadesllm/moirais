# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap confidence interval (percentile, BCa, studentized).

Implements three resampling-based CIs from Efron & Tibshirani (1993,
*An Introduction to the Bootstrap*, Chapters 12-14):

* **percentile**  CI = [theta*_{alpha/2}, theta*_{1-alpha/2}]
* **BCa**         bias-corrected & accelerated; see Efron (1987 JASA)
* **studentized** Hall (1988) bootstrap-t using nested resamples for SE

Default statistic is the mean; pass `statistic=np.median` etc.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["bootstrap_ci"]


def bootstrap_ci(x, statistic=None, B: int = 2000, alpha: float = 0.05,
                 method: str = "percentile", seed: int = 42):
    """Bootstrap confidence interval (percentile, BCa, studentized).

    Parameters
    ----------
    x : array-like
        Sample.
    statistic : callable, optional
        ``statistic(sample) -> scalar``.  Default = sample mean.
    B : int
        Number of bootstrap resamples (default 2000).
    alpha : float
        Two-sided level (default 0.05 -> 95% CI).
    method : {"percentile", "bca", "studentized"}
    seed : int

    Returns
    -------
    RichResult
        Keys: ``estimate, se, ci_lower, ci_upper, method, B, n``.

    References
    ----------
    Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
    Efron, B. (1987). Better bootstrap confidence intervals. JASA 82, 171.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return RichResult(payload={"estimate": float("nan"), "se": float("nan"),
                                   "ci_lower": float("nan"), "ci_upper": float("nan"),
                                   "n": int(n), "method": method,
                                   "warning": "Need n >= 2."})
    if statistic is None:
        statistic = np.mean
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    idx = rng.integers(0, n, size=(B, n))
    boot = np.array([statistic(x[i]) for i in idx], dtype=float)
    se = float(np.std(boot, ddof=1))

    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    elif method == "bca":
        # bias correction
        z0 = stats.norm.ppf(np.mean(boot < theta_hat))
        # acceleration via jackknife
        jk = np.array([statistic(np.delete(x, i)) for i in range(n)])
        jk_mean = jk.mean()
        num = np.sum((jk_mean - jk) ** 3)
        den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
        a = num / den if den > 0 else 0.0
        z_lo = stats.norm.ppf(alpha / 2)
        z_hi = stats.norm.ppf(1 - alpha / 2)
        alpha1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
        lo, hi = np.quantile(boot, [alpha1, alpha2])
    elif method == "studentized":
        # nested SE per resample; B2 inner replicates
        B2 = max(50, B // 10)
        t_stars = np.empty(B)
        for b in range(B):
            xb = x[idx[b]]
            theta_b = statistic(xb)
            inner = np.array([statistic(xb[rng.integers(0, n, size=n)])
                              for _ in range(B2)])
            se_b = inner.std(ddof=1)
            t_stars[b] = (theta_b - theta_hat) / se_b if se_b > 0 else 0.0
        t_lo, t_hi = np.quantile(t_stars, [alpha / 2, 1 - alpha / 2])
        lo = theta_hat - t_hi * se
        hi = theta_hat - t_lo * se
    else:
        raise ValueError(f"Unknown method {method!r}; "
                         "use percentile / bca / studentized.")

    return RichResult(payload={
        "estimate": theta_hat, "se": se,
        "ci_lower": float(lo), "ci_upper": float(hi),
        "alpha": alpha, "B": int(B), "n": int(n),
        "method": f"Bootstrap CI ({method})",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.normal(0, 1, 100)
# >>> res = bootstrap_ci(x, B=500, seed=0)
# >>> assert abs(res["estimate"] - x.mean()) < 1e-9
# >>> assert res["ci_lower"] < res["estimate"] < res["ci_upper"]


def cheatsheet():
    return "btsrp(x, B=2000, alpha=0.05, method='percentile'): bootstrap CI."
