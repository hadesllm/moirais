"""Tests for morie.fn.ibs — identity by state."""
import numpy as np
import pytest
from morie.fn.ibs import identity_by_state


class TestIBS:
    def test_identical(self):
        G = np.array([[0, 1, 2], [0, 1, 2]])
        res = identity_by_state(G)
        assert res.extra["ibs_matrix"][0, 1] == pytest.approx(1.0)

    def test_diagonal_is_one(self):
        G = np.random.default_rng(42).choice([0, 1, 2], size=(5, 10))
        res = identity_by_state(G)
        np.testing.assert_allclose(np.diag(res.extra["ibs_matrix"]), 1.0)
