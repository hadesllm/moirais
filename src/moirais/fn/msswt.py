# moirais.fn — function file (hadesllm/moirais)
"""Missing data weight matrix. '200% Power.' -- Deku, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def missing_data_weights(D):
    """Create weight matrix: 0 for NaN, 1 for observed.

    Parameters
    ----------
    D : array-like
        Distance matrix with possible NaN values.

    Returns
    -------
    DescriptiveResult
        value = weight matrix (0/1), extra has n_missing.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    W = (~np.isnan(D)).astype(float)
    n_missing = int(np.isnan(D).sum())
    return DescriptiveResult(name="missing_data_weights", value=W, extra={"n_missing": n_missing, "shape": D.shape})


msswt = missing_data_weights


def cheatsheet() -> str:
    return "missing_data_weights({}) -> Missing data weight matrix. '200% Power.' -- Deku, My Hero A"
