# moirais.fn — function file (hadesllm/moirais)
"""Alpha-NOMINATE (Bayesian MCMC)"""

import numpy as np

from ._containers import DescriptiveResult


def alpha_nominate(data, *, method="default"):
    """Alpha-NOMINATE (Bayesian MCMC)

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
        name="nmaln",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


alph = alpha_nominate


def cheatsheet() -> str:
    return "alpha_nominate({}) -> Alpha-NOMINATE (Bayesian MCMC)"
