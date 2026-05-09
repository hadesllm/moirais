"""Transfer entropy."""

import numpy as np

from ._containers import ESRes


def transfer_entropy(x, y, k: int = 1, bins: int = 20, **kwargs) -> ESRes:
    """
    Compute transfer entropy from Y to X (TE_{Y->X}).

    Measures the reduction in uncertainty of X given its own past
    and the past of Y. A nonlinear Granger-like causality measure.

    .. math::

        TE_{Y \\to X} = H(X_t | X_{t-k}) - H(X_t | X_{t-k}, Y_{t-k})

    :param x: 1-D array-like, target time series.
    :param y: 1-D array-like, source time series.
    :param k: History length (default 1).
    :param bins: Number of histogram bins.
    :return: ESRes with transfer entropy in bits.

    References
    ----------
    Schreiber T (2000). Measuring information transfer. Physical
    Review Letters, 85(2), 461-464.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    if n < k + 2:
        raise ValueError(f"Need at least {k + 2} observations.")
    x = x[:n]
    y = y[:n]

    x_now = x[k:]
    x_past = x[:n - k]
    y_past = y[:n - k]

    def _h(data):
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_xnow_xpast = _h(np.column_stack([x_now, x_past]))
    h_xpast = _h(x_past.reshape(-1, 1))
    h_xnow_xpast_ypast = _h(np.column_stack([x_now, x_past, y_past]))
    h_xpast_ypast = _h(np.column_stack([x_past, y_past]))

    te = (h_xnow_xpast - h_xpast) - (h_xnow_xpast_ypast - h_xpast_ypast)

    return ESRes(
        measure="transfer_entropy",
        estimate=float(te),
        n=n,
        extra={"k": k, "direction": "Y->X"},
    )


trsfn = transfer_entropy


def cheatsheet() -> str:
    return "transfer_entropy(x, y, k=1) -> Transfer entropy Y->X."
