# morie.fn — function file (hadesllm/morie)
"""OC Coombs mesh"""

import numpy as np

from ._containers import DescriptiveResult


def oc_coombs_mesh(data, *, method="default"):
    """OC Coombs mesh

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
        name="nmocm",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


oc_c = oc_coombs_mesh


def cheatsheet() -> str:
    return "oc_coombs_mesh({}) -> OC Coombs mesh"
