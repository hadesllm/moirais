"""
Spatial panel random effects

Category: SpatReg2
"""

import numpy as np


def srpre(X=None, y=None, w=None, n=50, k=3):
    """Spatial panel random effects

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


short = "srpre"
alias = "srpre"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
srpre = srpre


def cheatsheet() -> str:
    return "srpre({}) -> Spatial panel random effects"
