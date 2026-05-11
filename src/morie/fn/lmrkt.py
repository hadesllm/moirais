# morie.fn — function file (hadesllm/morie)
"""Local Markov transition matrices for spatial regime dynamics."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def local_markov(
    values: np.ndarray,
    W: np.ndarray,
    n_classes: int = 4,
    quantile_breaks: np.ndarray | None = None,
) -> SpatialResult:
    r"""
    Local (spatial) Markov chain transition matrices (Rey, 2001).

    Discretizes a panel of continuous values into *k* classes (via quantiles)
    and estimates separate :math:`k \times k` transition matrices conditioned
    on the spatial lag class at time *t*:

    .. math::

        P^{(c)}_{ij} = \frac{n^{(c)}_{ij}}{\sum_j n^{(c)}_{ij}}

    where :math:`c \in \{1, \ldots, k\}` is the spatial lag class and
    :math:`n^{(c)}_{ij}` counts transitions from class *i* to *j* among
    units whose lag is in class *c*.

    Parameters
    ----------
    values : np.ndarray
        (n, T) panel data.
    W : np.ndarray
        (n, n) row-standardized spatial weight matrix.
    n_classes : int
        Number of quantile classes (default 4).
    quantile_breaks : np.ndarray or None
        Custom breakpoints. If None, uses equal-frequency quantiles.

    Returns
    -------
    SpatialResult
        statistic = chi-squared test for spatial independence of transitions,
        extra has ``transition_matrices`` (dict of k arrays of shape (k, k)),
        ``global_transition``, ``classes``.

    References
    ----------
    Rey SJ (2001). Spatial empirics for economic growth and convergence.
    *Geographical Analysis*, 33(3), 195--214.
    doi:10.1111/j.1538-4632.2001.tb00444.x

    Rey SJ (2004). Spatial dependence in the evolution of regional income
    distributions. In *Spatially Integrated Social Science*, M. Goodchild
    & D. Janelle (eds.), Oxford UP, 193--213.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n, T = 30, 10
    >>> W = np.ones((n, n)) / (n - 1)
    >>> np.fill_diagonal(W, 0)
    >>> vals = rng.normal(0, 1, (n, T)).cumsum(axis=1)
    >>> res = local_markov(vals, W, n_classes=3)
    >>> len(res.extra["transition_matrices"]) == 3
    True
    """
    values = np.asarray(values, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, T = values.shape
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")
    if T < 2:
        raise ValueError("Need at least 2 time periods.")

    if quantile_breaks is None:
        all_vals = values.ravel()
        quantile_breaks = np.percentile(all_vals, np.linspace(0, 100, n_classes + 1))
        quantile_breaks[0] = -np.inf
        quantile_breaks[-1] = np.inf

    def _classify(v: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(v, quantile_breaks[1:]), 0, n_classes - 1)

    classes = np.empty((n, T), dtype=int)
    for tt in range(T):
        classes[:, tt] = _classify(values[:, tt])

    lag_vals = np.empty((n, T))
    for tt in range(T):
        lag_vals[:, tt] = W @ values[:, tt]
    lag_classes = np.empty((n, T), dtype=int)
    for tt in range(T):
        lag_classes[:, tt] = _classify(lag_vals[:, tt])

    global_trans = np.zeros((n_classes, n_classes))
    local_trans = {c: np.zeros((n_classes, n_classes)) for c in range(n_classes)}

    for tt in range(T - 1):
        for i in range(n):
            fr = classes[i, tt]
            to = classes[i, tt + 1]
            lc = lag_classes[i, tt]
            global_trans[fr, to] += 1
            local_trans[lc][fr, to] += 1

    def _normalize(mat: np.ndarray) -> np.ndarray:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return mat / row_sums

    global_prob = _normalize(global_trans)
    local_prob = {c: _normalize(local_trans[c]) for c in range(n_classes)}

    chi2 = 0.0
    for c in range(n_classes):
        for i in range(n_classes):
            for j in range(n_classes):
                obs = local_trans[c][i, j]
                row_total = local_trans[c][i, :].sum()
                if row_total > 0 and global_prob[i, j] > 0:
                    exp = row_total * global_prob[i, j]
                    if exp > 0:
                        chi2 += (obs - exp) ** 2 / exp

    return SpatialResult(
        name="local_markov",
        statistic=chi2,
        extra={
            "transition_matrices": local_prob,
            "global_transition": global_prob,
            "classes": classes,
            "n_classes": n_classes,
            "n": n,
            "T": T,
        },
    )


lmrkt = local_markov


def cheatsheet() -> str:
    return "local_markov({}) -> Local Markov transition matrices for spatial regime dynamics"
