"""
Spatial hierarchical 2-level

Category: SpatReg2
"""

import numpy as np


def srhm2(X=None, y=None, w=None, n=50, k=3):
    """Spatial hierarchical 2-level

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if X is None:
        X = np.random.default_rng(0).standard_normal((n, k))
    if y is None:
        y = X @ np.ones(X.shape[1]) + np.random.default_rng(1).standard_normal(X.shape[0]) * 0.5
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    stat = float(1 - np.sum(resid**2) / np.sum((y - np.mean(y)) ** 2))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "coefficients": beta.tolist(),
            "residual_var": float(np.var(resid)),
            "r_squared": float(stat),
            "n": len(y),
            "k": X.shape[1],
        },
    )


short = "srhm2"
alias = "srhm2"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
srhm2 = srhm2


def cheatsheet() -> str:
    return "srhm2({}) -> Spatial hierarchical 2-level"
