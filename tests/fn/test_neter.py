"""Tests for erdos_renyi."""
import numpy as np, pytest
from moirais.fn.neter import erdos_renyi

class TestER:
    def test_basic(self):
        r = erdos_renyi(20, 0.3, seed=0)
        assert r.extra["n_nodes"] == 20
        assert r.extra["n_edges"] > 0

    def test_empty(self):
        r = erdos_renyi(10, 0.0, seed=0)
        assert r.extra["n_edges"] == 0

    def test_complete(self):
        r = erdos_renyi(5, 1.0, seed=0)
        assert r.extra["n_edges"] == 10
