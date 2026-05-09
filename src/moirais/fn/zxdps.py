"""Dirichlet process spatial"""

import numpy as np

from ._containers import DescriptiveResult


def dirichlet_proc_sp(data, *, method="default"):
    """Dirichlet process spatial

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
        name="zxdps",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


diri = dirichlet_proc_sp


def cheatsheet() -> str:
    return "dirichlet_proc_sp({}) -> Dirichlet process spatial"
