# moirais.fn — function file (hadesllm/moirais)
"""Hotelling T^2 (one-sample) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import f as _f


def hotelt2(X: Union[Sequence, np.ndarray],
            mu0: Union[Sequence, np.ndarray]):
    """Hotelling T^2 for testing mu = mu0."""
    from ._richresult import hypothesis_test_result
    X = np.asarray(X, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    n, p = X.shape
    if mu0.size != p:
        raise ValueError(f"mu0 length ({mu0.size}) must match X cols ({p}).")
    if n <= p:
        raise ValueError(f"need n>p; got n={n}, p={p}.")
    xbar = X.mean(axis=0)
    S = np.cov(X.T, ddof=1)
    diff = xbar - mu0
    t2 = n * float(diff @ np.linalg.pinv(S) @ diff)
    f_stat = ((n - p) / (p * (n - 1))) * t2
    p_val = float(1 - _f.cdf(f_stat, p, n - p))
    warnings = []
    if n < 5 * p:
        warnings.append(f"n={n} small relative to p={p}; multivariate normality critical.")
    return hypothesis_test_result(
        test_name="Hotelling T^2 (one-sample multivariate)",
        statistic=f_stat, df=(p, n - p), pvalue=p_val,
        extra_summary=[("T^2 statistic", t2), ("F-equivalent", f_stat),
                       ("df1 (between)", p), ("df2 (within)", n - p),
                       ("n", n), ("p (variables)", p),
                       ("Mean diff norm", float(np.linalg.norm(diff)))],
        warnings=warnings,
        extra_payload={"T2": t2, "F": f_stat, "df1": p, "df2": n - p,
                       "xbar": xbar.tolist(), "mu0": mu0.tolist()},
    )
