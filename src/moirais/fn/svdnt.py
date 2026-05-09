"""SVD entropy."""

import numpy as np

from ._containers import ESRes


def svd_entropy(x, m: int = 10, delay: int = 1, **kwargs) -> ESRes:
    """
    Compute SVD entropy of a time series.

    Constructs an embedding matrix and computes the normalised entropy
    of its singular value spectrum.

    .. math::

        H_{SVD} = -\\sum_i \\bar{\\sigma}_i \\log_2 \\bar{\\sigma}_i

    where bar{sigma}_i are the normalised singular values.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 10).
    :param delay: Time delay (default 1).
    :return: ESRes with normalised SVD entropy in [0,1].

    References
    ----------
    Roberts SJ et al. (1999). Temporal and spatial complexity measures
    for electroencephalogram based brain-computer interfacing.
    Medical & Biological Engineering & Computing, 37(1), 93-98.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    n_embed = n - (m - 1) * delay
    if n_embed < 2:
        raise ValueError(f"Need at least {(m - 1) * delay + 2} observations.")

    embedded = np.array([x[i:i + m * delay:delay] for i in range(n_embed)])
    sv = np.linalg.svd(embedded, compute_uv=False)
    sv_norm = sv / sv.sum()
    sv_norm = sv_norm[sv_norm > 0]
    h = -float(np.sum(sv_norm * np.log2(sv_norm)))
    h_max = np.log2(len(sv))
    h_norm = h / h_max if h_max > 0 else 0.0

    return ESRes(
        measure="svd_entropy",
        estimate=h_norm,
        n=n,
        extra={"raw_entropy": h, "max_entropy": h_max, "n_sv": len(sv), "m": m},
    )


svdnt = svd_entropy


def cheatsheet() -> str:
    return "svd_entropy(x, m=10) -> Normalised SVD entropy."
