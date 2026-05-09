# moirais.fn — function file (hadesllm/moirais)
"""Alpha-NOMINATE acceptance rate"""

import numpy as np

from ._containers import DescriptiveResult


def alpha_nom_accept(data, *, method="default"):
    """Alpha-NOMINATE acceptance rate

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
        name="nmala",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


alph = alpha_nom_accept


def cheatsheet() -> str:
    return "alpha_nom_accept({}) -> Alpha-NOMINATE acceptance rate"
