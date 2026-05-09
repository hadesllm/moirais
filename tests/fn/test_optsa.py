"""Tests for moirais.fn.optsa -- simulated annealing MDS."""

import numpy as np
from moirais.fn.optsa import simulated_anneal_mds, optsa


def test_optsa_smoke():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    r = optsa(D, n_dims=2, n_iter=50)
    assert r.name == "simulated_anneal_mds"
    assert r.value.shape == (3, 2)
    assert "stress" in r.extra


def test_optsa_alias():
    assert optsa is simulated_anneal_mds
