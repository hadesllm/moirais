"""Network spatial clustering"""

import numpy as np

from ._containers import DescriptiveResult


def network_cluster_sp(observed, *, expected=None):
    """Network spatial clustering

    Returns
    -------
    DescriptiveResult
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float) if expected is not None else np.ones_like(observed) * observed.mean()
    ratio = observed / (expected + 1e-10)
    stat = float(np.max(ratio))
    idx = int(np.argmax(ratio))
    return DescriptiveResult(
        name="zxncl",
        value=float(stat) if isinstance(stat, (int, float)) else stat,
        extra={},
    )


netw = network_cluster_sp


def cheatsheet() -> str:
    return "network_cluster_sp({}) -> Network spatial clustering"
