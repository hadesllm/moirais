"""Tests for moirais.fn.ssefn -- SSE for MDS."""

import numpy as np
from moirais.fn.ssefn import sse_mds, ssefn


def test_ssefn_perfect():
    X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    D = np.zeros((3, 3))
    D[0, 1] = D[1, 0] = 1.0
    D[0, 2] = D[2, 0] = 1.0
    D[1, 2] = D[2, 1] = np.sqrt(2)
    r = ssefn(X, D)
    assert r.name == "sse_mds"
    assert r.value < 1e-10


def test_ssefn_alias():
    assert ssefn is sse_mds
