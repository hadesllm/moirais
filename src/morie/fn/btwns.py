# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Betweenness centrality for unweighted graphs."""

from __future__ import annotations

from collections import deque

import numpy as np

from ._containers import DescriptiveResult


def betweenness(adj_matrix: np.ndarray) -> DescriptiveResult:
    r"""Betweenness centrality via Brandes' algorithm.

    .. math::

        C_B(v) = \\sum_{s \\neq v \\neq t}
                 \\frac{\\sigma_{st}(v)}{\\sigma_{st}}

    where :math:`\\sigma_{st}` is the number of shortest paths from *s*
    to *t* and :math:`\\sigma_{st}(v)` is the number passing through *v*.

    Parameters
    ----------
    adj_matrix : ndarray
        Square adjacency matrix (n x n), treated as unweighted (nonzero
        entries are edges).

    Returns
    -------
    DescriptiveResult
        ``value`` is a 1-D array of betweenness scores (unnormalized).
        ``extra`` has ``n_nodes`` and ``n_edges``.

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163--177.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square.")

    n = A.shape[0]
    cb = np.zeros(n)

    for s in range(n):
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        d = np.full(n, -1)
        d[s] = 0
        Q: deque[int] = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            for w in range(n):
                if A[v, w] == 0:
                    continue
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    n_edges = int(np.count_nonzero(A))

    return DescriptiveResult(
        name="Betweenness",
        value=cb,
        extra={"n_nodes": n, "n_edges": n_edges},
    )


btwns = betweenness


def cheatsheet() -> str:
    return "betweenness({}) -> Betweenness centrality for unweighted graphs."
