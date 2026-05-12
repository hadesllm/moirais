# morie.fn -- function file (hadesllm/morie)
"""Classical metric multidimensional scaling."""

from __future__ import annotations

import numpy as np

from ._containers import MdsRes


def classical_metric_mds(D, n_dims: int = 2) -> MdsRes:
    """Classical (Torgerson) metric MDS via eigendecomposition.

    :param D: Square distance matrix.
    :param n_dims: Number of dimensions to retain.
    :return: MdsRes with coordinates and eigenvalues.

    .. epigraph:: "A process cannot be understood by stopping it." -- First Law of Mentat, Dune
    """
    from morie._spatial_voting import classical_mds as _fn

    result = _fn(D, n_dims=n_dims)
    return MdsRes(
        coordinates=result["coordinates"],
        stress=result.get("stress", 0.0),
        eigenvalues=result.get("eigenvalues", np.array([])),
    )


cmds = classical_metric_mds


def cheatsheet() -> str:
    return "classical_metric_mds({}) -> Classical metric multidimensional scaling."
