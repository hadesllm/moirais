"""
Spatial quasi-ML estimator

Category: SpatReg2
"""

import numpy as np


def srqml(X=None, y=None, w=None, n=50, k=3):
    """Spatial quasi-ML estimator

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


short = "srqml"
alias = "srqml"
quote = "Not all those who wander are lost. -- Gandalf"
srqml = srqml


def cheatsheet() -> str:
    return "srqml({}) -> Spatial quasi-ML estimator"
