# moirais.fn — function file (hadesllm/moirais)
"""Embedding quality measure"""

import numpy as np

from ._containers import DescriptiveResult


def embedding_qual(data, *, method="default"):
    """Embedding quality measure

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
        name="msemb",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


embe = embedding_qual


def cheatsheet() -> str:
    return "embedding_qual({}) -> Embedding quality measure"
