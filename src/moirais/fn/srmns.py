"""
Spatial Manski model

Category: SpatReg2
"""

import numpy as np


def srmns(X=None, y=None, w=None, n=50, k=3):
    """Spatial Manski model

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


short = "srmns"
alias = "srmns"
quote = "I will take a potato chip and eat it! -- Light"
srmns = srmns


def cheatsheet() -> str:
    return "srmns({}) -> Spatial Manski model"
