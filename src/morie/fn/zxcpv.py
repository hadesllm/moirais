"""Vine copula spatial"""

import numpy as np

from ._containers import DescriptiveResult


def copula_vine_sp(data, *, method="default"):
    """Vine copula spatial

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="zxcpv",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


copu = copula_vine_sp


def cheatsheet() -> str:
    return "copula_vine_sp({}) -> Vine copula spatial"
