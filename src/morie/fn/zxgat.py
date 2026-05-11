"""Graph attention spatial"""

import numpy as np

from ._containers import DescriptiveResult


def graph_attention_sp(data, *, method="default"):
    """Graph attention spatial

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
        name="zxgat",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


grap = graph_attention_sp


def cheatsheet() -> str:
    return "graph_attention_sp({}) -> Graph attention spatial"
