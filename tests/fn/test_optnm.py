"""Tests for moirais.fn.optnm -- Nelder-Mead MDS."""

import numpy as np
from moirais.fn.optnm import nelder_mead_mds, optnm


def test_optnm_smoke():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    r = optnm(D, n_dims=2, maxiter=200)
    assert r.name == "nelder_mead_mds"
    assert r.value.shape == (3, 2)
    assert "stress" in r.extra


def test_optnm_alias():
    assert optnm is nelder_mead_mds
